"""Tests for embedding endpoint client behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from warehouseeye.gpu.embedding_client import EmbeddingClient


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (12, 12), color)
    image.save(path)


class _DummyOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = self

    def create(self, **kwargs):
        raise AssertionError(f"Unexpected live API call in test with kwargs={kwargs}")


def test_compute_embeddings_batch_with_two_images(tmp_path, monkeypatch) -> None:
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    _write_image(image_a, (255, 0, 0))
    _write_image(image_b, (0, 255, 0))

    client = EmbeddingClient(base_url="http://localhost:8001/v1", client=_DummyOpenAIClient())

    def fake_request_embeddings(inputs):
        assert len(inputs) == 2
        for payload in inputs:
            content = payload["content"]
            assert content[0]["type"] == "image_url"
            assert content[0]["image_url"]["url"].startswith("data:image/")
        vec1 = np.ones(1536, dtype=np.float32)
        vec2 = np.ones(1536, dtype=np.float32) * 2.0
        vec1 /= np.linalg.norm(vec1)
        vec2 /= np.linalg.norm(vec2)
        return [vec1, vec2]

    monkeypatch.setattr(client, "_request_embeddings", fake_request_embeddings)
    batch = client.compute_embeddings_batch([image_a, image_b])
    assert batch.shape == (2, 1536)
    assert np.isclose(np.linalg.norm(batch[0]), 1.0)
    assert np.isclose(np.linalg.norm(batch[1]), 1.0)


def test_compute_embedding_single_image(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "single.png"
    _write_image(image_path, (0, 0, 255))
    client = EmbeddingClient(base_url="http://localhost:8001/v1", client=_DummyOpenAIClient())

    expected = np.ones(1536, dtype=np.float32)
    expected /= np.linalg.norm(expected)
    monkeypatch.setattr(client, "_request_embeddings", lambda inputs: [expected])

    vector = client.compute_embedding(image_path)
    assert vector.shape == (1536,)
    assert np.isclose(np.linalg.norm(vector), 1.0)
