"""Tests for semantic analysis helpers."""

import json

from warehouseeye.gpu.vision_analyzer import VisionAnalyzer


def test_select_representative_rows_picks_largest_bboxes() -> None:
    rows = [
        (1, 1.0, 1, "a.jpg", "f1.jpg", 0.0, 0.0, 10.0, 10.0),
        (2, 2.0, 2, "b.jpg", "f2.jpg", 0.0, 0.0, 40.0, 40.0),
        (3, 3.0, 3, "c.jpg", "f3.jpg", 0.0, 0.0, 20.0, 20.0),
        (4, 4.0, 4, "d.jpg", "f4.jpg", 0.0, 0.0, 35.0, 35.0),
    ]
    sampled = VisionAnalyzer.select_representative_rows(rows, min_count=2, max_count=3)
    assert len(sampled) == 3
    sampled_ids = {int(row[0]) for row in sampled}
    assert sampled_ids == {2, 3, 4}


def test_extract_json_block_from_markdown_fence() -> None:
    text = """
    ```json
    {
      "activity": "handling_object",
      "activity_description": "Worker handling a box.",
      "objects_involved": ["box cutter"],
      "zone_inference": "packing area",
      "interaction_with_others": null,
      "anomaly_flag": false,
      "anomaly_reason": null,
      "supervisor_attention_recommended": false,
      "confidence": 0.82,
      "reasoning": "Hands and posture indicate object handling."
    }
    ```
    """
    extracted = VisionAnalyzer._extract_json_block(text)
    parsed = json.loads(extracted)
    assert parsed["activity"] == "handling_object"
    assert parsed["anomaly_reason"] is None
