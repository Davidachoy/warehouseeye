"""Tests for semantic analysis helpers."""

import json

from warehouseeye.gpu.vision_analyzer import VisionAnalyzer


def test_select_representative_rows_keeps_temporal_spread() -> None:
    rows = [(i,) for i in range(10)]
    sampled = VisionAnalyzer.select_representative_rows(rows, min_count=3, max_count=5)
    assert len(sampled) == 5
    assert sampled[0] == rows[0]
    assert sampled[-1] == rows[-1]


def test_extract_json_block_from_markdown_fence() -> None:
    text = """
    ```json
    {
      "activity": "packing product into box",
      "relative_location": "central packing table",
      "visible_tools": ["box cutter"],
      "object_interaction": "holding carton",
      "posture": "standing bent forward",
      "anomaly": false,
      "severity": null
    }
    ```
    """
    extracted = VisionAnalyzer._extract_json_block(text)
    parsed = json.loads(extracted)
    assert parsed["activity"] == "packing product into box"
    assert parsed["severity"] is None
