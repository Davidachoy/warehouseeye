"""HTTP client wrapper for WarehouseEye FastAPI backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class BackendUnavailableError(RuntimeError):
    """Raised when API backend cannot be reached."""


@dataclass
class ApiClient:
    """Small sync API client tailored for Streamlit interactions."""

    base_url: str
    timeout_sec: float = 4.0

    def _request(self, method: str, path: str, json_data: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        try:
            response = httpx.request(
                method=method,
                url=url,
                json=json_data,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            raise RuntimeError(f"{exc.response.status_code} {exc.response.reason_phrase}: {detail}") from exc
        except httpx.HTTPError as exc:
            raise BackendUnavailableError(
                "WarehouseEye backend is unavailable. Start FastAPI and verify API_BASE_URL."
            ) from exc

    def get_health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def get_timeline(self, video_id: str) -> dict[str, Any]:
        return self._request("GET", f"/timeline/{video_id}")

    def get_track_timeline(self, video_id: str, track_id: int) -> dict[str, Any]:
        return self._request("GET", f"/timeline/{video_id}/track/{track_id}")

    def analyze(self, video_id: str, video_url: str, force: bool = False) -> dict[str, Any]:
        # API currently requires video_url in AnalyzeRequest schema.
        return self._request("POST", "/analyze", {"video_id": video_id, "video_url": video_url, "force": force})

    def query(self, video_id: str, question: str) -> dict[str, Any]:
        return self._request("POST", "/query", {"video_id": video_id, "question": question})

    def benchmark(self) -> dict[str, Any]:
        return self._request("GET", "/benchmark")
