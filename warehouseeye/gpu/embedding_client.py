"""OpenAI-compatible client for Qwen3-VL embedding inference."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for multimodal embedding calls against a vLLM endpoint."""

    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        api_key: str = "EMPTY",
        timeout_sec: float = 45.0,
        max_attempts: int = 3,
        initial_backoff_sec: float = 0.5,
        embedding_dimension: int = 1536,
        client: Any | None = None,
    ) -> None:
        self.base_url = os.getenv("EMBEDDING_URL", base_url).rstrip("/")
        self.model_name = os.getenv("EMBEDDING_MODEL", model_name)
        self.api_key = api_key or os.getenv("AMD_API_KEY", "EMPTY")
        self.timeout_sec = timeout_sec
        self.max_attempts = max_attempts
        self.initial_backoff_sec = initial_backoff_sec
        self.embedding_dimension = embedding_dimension
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.latencies_ms: list[float] = []
        if client is not None:
            self._client = client
            return
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._client = httpx.Client(
            timeout=self.timeout_sec,
            headers=headers,
        )

    @staticmethod
    def _image_to_data_url(image_path: str | Path) -> str:
        path = Path(image_path)
        raw = path.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        detected_type, _ = mimetypes.guess_type(path.name)
        mime_type = detected_type or "image/jpeg"
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Embedding norm is zero; cannot normalize.")
        return vector / norm

    def _coerce_dimension(self, vector: np.ndarray) -> np.ndarray:
        if vector.shape == (self.embedding_dimension,):
            return vector
        if vector.shape[0] > self.embedding_dimension:
            logger.info(
                "embedding_dimension_truncated",
                extra={
                    "original_dim": int(vector.shape[0]),
                    "target_dim": self.embedding_dimension,
                },
            )
            return vector[: self.embedding_dimension]
        msg = (
            "Embedding dimension is smaller than requested output dimension: "
            f"{vector.shape[0]} < {self.embedding_dimension}."
        )
        raise ValueError(msg)

    def _build_input(self, image_path: str | Path) -> dict[str, Any]:
        image_url = self._image_to_data_url(image_path)
        return {
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
                {
                    "type": "text",
                    "text": "Represent the given image.",
                },
            ],
        }

    def _post_v1_embeddings(self, messages_batch: list[dict[str, Any]]) -> dict[str, Any]:
        response = self._client.post(
            f"{self.base_url}/embeddings",
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": item["content"]} for item in messages_batch],
                "encoding_format": "float",
            },
        )
        response.raise_for_status()
        return response.json()

    def _extract_vectors_v1(self, payload: dict[str, Any]) -> list[np.ndarray]:
        data = payload.get("data", [])
        if not isinstance(data, list):
            msg = f"Unexpected /v1/embeddings payload: {json.dumps(payload)[:400]}"
            raise RuntimeError(msg)
        vectors: list[np.ndarray] = []
        for item in data:
            vector = np.asarray(item["embedding"], dtype=np.float32)
            vector = self._coerce_dimension(vector)
            vectors.append(self._normalize(vector))
        return vectors

    def _request_embeddings(self, messages_batch: list[dict[str, Any]]) -> list[np.ndarray]:
        self.request_count += 1
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            started_at = time.perf_counter()
            try:
                payload = self._post_v1_embeddings(messages_batch)
                vectors = self._extract_vectors_v1(payload)
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                self.latencies_ms.append(elapsed_ms)
                logger.info(
                    "embedding_request_timing",
                    extra={
                        "attempt": attempt,
                        "elapsed_ms": round(elapsed_ms, 2),
                        "batch_size": len(messages_batch),
                        "status": "ok",
                    },
                )
                self.success_count += 1
                return vectors
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.HTTPStatusError,
                TimeoutError,
                ConnectionError,
                ValueError,
            ) as exc:
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                self.latencies_ms.append(elapsed_ms)
                last_error = exc
                logger.info(
                    "embedding_request_retry",
                    extra={
                        "attempt": attempt,
                        "elapsed_ms": round(elapsed_ms, 2),
                        "batch_size": len(messages_batch),
                        "error": str(exc),
                    },
                )
                if attempt >= self.max_attempts:
                    break
                backoff = self.initial_backoff_sec * (2 ** (attempt - 1))
                time.sleep(backoff)
            except Exception as exc:  # pragma: no cover - final guard.
                last_error = exc
                break
        self.failure_count += 1
        raise RuntimeError(
            f"Failed embedding request after retries (base_url={self.base_url}, model={self.model_name})."
        ) from last_error

    def compute_embedding(self, image_path: str | Path) -> np.ndarray:
        """Compute one L2-normalized embedding vector of shape (1536,)."""
        vectors = self._request_embeddings([self._build_input(image_path)])
        vector = vectors[0]
        if vector.shape != (self.embedding_dimension,):
            msg = f"Unexpected embedding shape: {vector.shape}"
            raise ValueError(msg)
        return vector

    def compute_embeddings_batch(self, image_paths: list[str | Path]) -> np.ndarray:
        """Compute a batch of normalized embeddings with shape (N, 1536)."""
        if not image_paths:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)
        inputs = [self._build_input(path) for path in image_paths]
        vectors = self._request_embeddings(inputs)
        if len(vectors) != len(image_paths):
            msg = "Embedding response length does not match input batch size."
            raise RuntimeError(msg)
        return np.vstack(vectors).astype(np.float32)

    @property
    def average_latency_ms(self) -> float:
        """Return mean request latency in milliseconds."""
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)
