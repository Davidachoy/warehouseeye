#!/usr/bin/env python3
"""Smoke test for OpenAI-compatible vLLM on the droplet (no extra deps beyond stdlib).

Usage (on droplet, server already running):
  python3 scripts/amd/test_vlm_inference.py
  BASE_URL=http://127.0.0.1:8000/v1 MODEL=Qwen/Qwen3-VL-8B-Instruct python3 scripts/amd/test_vlm_inference.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def main() -> None:
    base = os.environ.get("BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    model = os.environ.get("MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    image_url = os.environ.get(
        "TEST_IMAGE_URL",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
    )

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image in one short sentence."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": 128,
    }

    url = f"{base}/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        raise SystemExit(f"HTTP {exc.code} from {url}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Request failed: {exc}\nIs vLLM running on {base}?") from exc

    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        print(json.dumps(body, indent=2))
        raise SystemExit("Unexpected response shape") from exc

    print("OK — model reply:")
    print(text)


if __name__ == "__main__":
    main()
