"""Compare 8B vs 32B crop descriptions and render an HTML report."""

from __future__ import annotations

import argparse
import asyncio
import base64
import html
import json
import os
import random
from pathlib import Path
from typing import Any

from warehouseeye.gpu.vllm_client import VLLMClient


def _discover_crops(crops_root: Path) -> list[Path]:
    return sorted(
        path
        for path in crops_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def _load_prompt() -> str:
    prompt_path = Path(os.getenv("ACTIVITY_PROMPT_PATH", "prompts/activity_extraction.txt"))
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return "Describe the warehouse worker in this crop with concrete details."


def _image_data_url(path: Path) -> str:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    ext = path.suffix.lower().lstrip(".") or "jpeg"
    mime = "jpeg" if ext == "jpg" else ext
    return f"data:image/{mime};base64,{b64}"


def _pick_winner(desc_8b: str, desc_32b: str) -> str:
    if len(desc_32b) > len(desc_8b) + 40:
        return "32B likely better (more detail)."
    if len(desc_8b) > len(desc_32b) + 40:
        return "8B unexpectedly more detailed."
    return "Close call based on length alone."


async def _judge_response(client_32b: VLLMClient, desc_8b: str, desc_32b: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are judging two vision descriptions of the same crop. "
                "Respond in 2-3 concise sentences with which is better and why. "
                "Criteria: concrete visual detail, factual grounding, and operational usefulness."
            ),
        },
        {
            "role": "user",
            "content": (
                f"8B description:\n{desc_8b}\n\n"
                f"32B description:\n{desc_32b}\n\n"
                "Which one is better and why?"
            ),
        },
    ]
    try:
        return await client_32b.chat_completion_async(messages=messages, max_tokens=180)
    except Exception:
        return _pick_winner(desc_8b, desc_32b)


async def _run_compare(args: argparse.Namespace) -> dict[str, Any]:
    crops_root = Path(args.crops_root).resolve()
    if not crops_root.exists():
        raise FileNotFoundError(f"Crops root does not exist: {crops_root}")
    all_crops = _discover_crops(crops_root)
    if not all_crops:
        raise RuntimeError(f"No crops found under {crops_root}")

    rnd = random.Random(args.seed)
    sample_count = min(max(args.sample_count, 1), len(all_crops))
    selected = rnd.sample(all_crops, sample_count)

    url_8b = args.url_8b or os.getenv("AMD_URL_8B")
    url_32b = args.url_32b or os.getenv("AMD_URL_32B")
    model_8b = args.model_8b or os.getenv("AMD_MODEL_8B", "Qwen/Qwen3-VL-8B-Instruct")
    model_32b = args.model_32b or os.getenv("AMD_MODEL_32B", "Qwen/Qwen3-VL-32B-Instruct")
    if not url_8b or not url_32b:
        raise ValueError("Provide --url-8b and --url-32b, or set AMD_URL_8B / AMD_URL_32B.")

    prompt = _load_prompt()
    client_8b = VLLMClient(base_url=url_8b, model_name=model_8b)
    client_32b = VLLMClient(base_url=url_32b, model_name=model_32b)
    rows: list[dict[str, Any]] = []
    try:
        for idx, crop_path in enumerate(selected, start=1):
            print(f"[compare] {idx}/{sample_count}: {crop_path.name}")
            desc_8b, desc_32b = await asyncio.gather(
                client_8b.describe_image_async(crop_path, prompt=prompt, max_tokens=args.max_tokens),
                client_32b.describe_image_async(crop_path, prompt=prompt, max_tokens=args.max_tokens),
            )
            analysis = await _judge_response(client_32b=client_32b, desc_8b=desc_8b, desc_32b=desc_32b)
            rows.append(
                {
                    "crop_path": str(crop_path),
                    "crop_data_url": _image_data_url(crop_path),
                    "desc_8b": desc_8b,
                    "desc_32b": desc_32b,
                    "analysis": analysis,
                }
            )
    finally:
        await client_8b.aclose()
        await client_32b.aclose()

    return {
        "crops_root": str(crops_root),
        "sample_count": sample_count,
        "seed": args.seed,
        "model_8b": model_8b,
        "model_32b": model_32b,
        "url_8b": url_8b,
        "url_32b": url_32b,
        "rows": rows,
    }


def _render_html(payload: dict[str, Any]) -> str:
    cards: list[str] = []
    for row in payload["rows"]:
        cards.append(
            f"""
<div class="card">
  <div class="imgWrap"><img src="{row['crop_data_url']}" alt="crop" /></div>
  <div class="cols">
    <div><h3>8B</h3><pre>{html.escape(row['desc_8b'])}</pre></div>
    <div><h3>32B</h3><pre>{html.escape(row['desc_32b'])}</pre></div>
  </div>
  <div class="analysis"><strong>Analysis:</strong> {html.escape(row['analysis'])}</div>
  <div class="meta">{html.escape(row['crop_path'])}</div>
</div>
"""
        )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>8B vs 32B comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f7fb; }}
    h1 {{ margin-bottom: 8px; }}
    .metaTop {{ color: #444; margin-bottom: 16px; }}
    .card {{ background: white; border-radius: 10px; padding: 14px; margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
    .imgWrap img {{ max-width: 260px; border-radius: 8px; border: 1px solid #ddd; }}
    .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }}
    pre {{ white-space: pre-wrap; background: #f8f8f8; padding: 10px; border-radius: 8px; min-height: 80px; }}
    .analysis {{ margin-top: 10px; }}
    .meta {{ color: #666; margin-top: 6px; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Qwen3-VL 8B vs 32B</h1>
  <div class="metaTop">
    Sample count: {payload['sample_count']} | seed: {payload['seed']}<br/>
    8B: {html.escape(payload['model_8b'])} @ {html.escape(payload['url_8b'])}<br/>
    32B: {html.escape(payload['model_32b'])} @ {html.escape(payload['url_32b'])}
  </div>
  {''.join(cards)}
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an HTML report comparing 8B vs 32B crop outputs.")
    parser.add_argument("--crops-root", required=True, help="Path containing crop images.")
    parser.add_argument("--url-8b", default=None, help="OpenAI-compatible endpoint for 8B.")
    parser.add_argument("--url-32b", default=None, help="OpenAI-compatible endpoint for 32B.")
    parser.add_argument("--model-8b", default=None, help="Model name for 8B endpoint.")
    parser.add_argument("--model-32b", default=None, help="Model name for 32B endpoint.")
    parser.add_argument("--sample-count", type=int, default=8, help="Number of random crops to compare.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument("--max-tokens", type=int, default=280, help="max_tokens per model response.")
    parser.add_argument(
        "--output-html",
        default="comparing_8b_vs_32b.html",
        help="Output report path.",
    )
    parser.add_argument(
        "--output-json",
        default="comparing_8b_vs_32b.json",
        help="Raw comparison payload for reproducibility.",
    )
    args = parser.parse_args()

    payload = asyncio.run(_run_compare(args))
    html_report = _render_html(payload)

    output_html = Path(args.output_html).resolve()
    output_json = Path(args.output_json).resolve()
    output_html.write_text(html_report, encoding="utf-8")
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[compare] wrote HTML report: {output_html}")
    print(f"[compare] wrote JSON payload: {output_json}")


if __name__ == "__main__":
    main()
