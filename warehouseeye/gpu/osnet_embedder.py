"""OSNet feature extractor for person re-identification.

Backed by torchreid (Kaiyang Zhou et al., MIT licensed). Designed to
plug into ReIDEngine in place of the multimodal Qwen3-VL embedding path
when stronger same-person discrimination is needed.

The class exposes the same minimal interface used by ReIDEngine:
    - compute_embedding(image_path: str | Path | np.ndarray) -> np.ndarray
    - compute_embeddings_batch(image_paths) -> np.ndarray  (N, D)
    - average_latency_ms property
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# OSNet weights pre-trained on MSMT17 (Apache-2.0/MIT compatible) hosted on
# kaiyangzhou's HuggingFace mirror. ImageNet weights are good for a generic
# vision warm-up but produce weak person-identity cosine clusters; MSMT17
# weights are trained for person re-identification specifically and lift the
# typical same-person cosine into the 0.85-0.95 range.
HF_OSNET_REID_WEIGHTS = {
    "osnet_x0_25": "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    "osnet_x0_5": "osnet_x0_5_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    "osnet_x0_75": "osnet_x0_75_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    "osnet_x1_0": "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    "osnet_ain_x1_0": "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth",
    "osnet_ibn_x1_0": "osnet_ibn_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
}


def _download_msmt17_weights(model_name: str) -> str | None:
    """Best-effort download of MSMT17-trained weights for OSNet variants."""
    weights_file = HF_OSNET_REID_WEIGHTS.get(model_name)
    if not weights_file:
        logger.info("osnet_msmt17_unsupported_model", extra={"model_name": model_name})
        return None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:  # pragma: no cover - huggingface_hub is a transitive dep
        logger.info("osnet_huggingface_hub_missing")
        return None
    try:
        weights_path = hf_hub_download(repo_id="kaiyangzhou/osnet", filename=weights_file)
        logger.info(
            "osnet_msmt17_weights_loaded",
            extra={"model_name": model_name, "weights_path": weights_path},
        )
        return weights_path
    except Exception as exc:  # pragma: no cover - network/IO best-effort
        logger.warning("osnet_msmt17_weights_download_failed: %s", exc)
        return None


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Embedding norm is zero; cannot normalize.")
    return vector / norm


class OSNetEmbedder:
    """Wrap torchreid's OSNet as a drop-in ReID embedder.

    OSNet is small (~2 MB for x0_25), fast on CPU, and trained explicitly
    for person re-identification, giving tighter same-person cosine
    distributions than general-purpose multimodal embeddings.
    """

    def __init__(
        self,
        model_name: str = "osnet_x0_25",
        model_path: str | None = None,
        device: str | None = None,
        embedding_dimension: int = 512,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment without torch
            msg = "torch is required for OSNetEmbedder."
            raise RuntimeError(msg) from exc

        FeatureExtractor = None
        # torchreid (Kaiyang Zhou, MIT) and torchreid-pip (kadirnar, MIT) expose
        # FeatureExtractor under different module paths; try both.
        for module_path in ("torchreid.utils", "torchreid.reid.utils"):
            try:
                module = __import__(module_path, fromlist=["FeatureExtractor"])
                FeatureExtractor = getattr(module, "FeatureExtractor", None)
                if FeatureExtractor is not None:
                    break
            except ImportError:
                continue
        if FeatureExtractor is None:
            msg = (
                "torchreid is required for OSNetEmbedder. "
                "Install with `pip install torchreid tensorboard` or set REID_BACKEND=qwen."
            )
            raise RuntimeError(msg)

        self.model_name = os.getenv("OSNET_MODEL_NAME", model_name)
        self.model_path = os.getenv("OSNET_MODEL_PATH", model_path or "")
        # If the caller did not pin an explicit model_path, try to auto-download
        # the MSMT17 weights for the chosen variant; ImageNet weights are a fallback.
        if not self.model_path:
            downloaded_path = _download_msmt17_weights(self.model_name)
            if downloaded_path:
                self.model_path = downloaded_path
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                # MPS works for OSNet inference on Apple Silicon.
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.embedding_dimension = embedding_dimension
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.latencies_ms: list[float] = []

        extractor_kwargs: dict[str, Any] = {
            "model_name": self.model_name,
            "device": self.device,
        }
        if self.model_path:
            extractor_kwargs["model_path"] = self.model_path
        self._extractor = FeatureExtractor(**extractor_kwargs)
        logger.info(
            "osnet_embedder_ready",
            extra={
                "model_name": self.model_name,
                "model_path": self.model_path or "<auto>",
                "device": self.device,
            },
        )

    @staticmethod
    def _load_image_array(image_input: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(image_input, np.ndarray):
            return np.ascontiguousarray(image_input.astype(np.uint8))
        image = Image.open(image_input).convert("RGB")
        return np.asarray(image, dtype=np.uint8)

    def compute_embedding(self, image_path: str | Path | np.ndarray) -> np.ndarray:
        """Compute one normalized embedding from a person crop."""
        self.request_count += 1
        started_at = time.perf_counter()
        try:
            image = self._load_image_array(image_input=image_path)
            features = self._extractor([image])
            vector = features[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
            normalized = _normalize(vector)
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            self.latencies_ms.append(elapsed_ms)
            self.success_count += 1
            logger.info(
                "osnet_embedding_timing",
                extra={
                    "elapsed_ms": round(elapsed_ms, 2),
                    "dim": int(normalized.shape[0]),
                    "device": self.device,
                },
            )
            return normalized
        except Exception:
            self.failure_count += 1
            raise

    def compute_embeddings_batch(self, image_paths: list[str | Path | np.ndarray]) -> np.ndarray:
        """Compute a batch of normalized embeddings as an (N, D) array."""
        if not image_paths:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)
        self.request_count += 1
        started_at = time.perf_counter()
        try:
            images = [self._load_image_array(image_input=item) for item in image_paths]
            features = self._extractor(images)
            matrix = features.detach().cpu().numpy().astype(np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            normalized = matrix / norms
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            self.latencies_ms.append(elapsed_ms)
            self.success_count += 1
            return normalized
        except Exception:
            self.failure_count += 1
            raise

    @property
    def average_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)
