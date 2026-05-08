"""Compatibility helpers for activity payload schema migration."""

from __future__ import annotations

from typing import Any

VALID_ACTIVITIES = {
    "walking",
    "standing",
    "handling_object",
    "lifting",
    "interacting",
    "inspecting",
    "idle",
    "other",
}


def _map_legacy_activity(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in VALID_ACTIVITIES:
        return text
    if any(token in text for token in ("pack", "carry", "load", "handle")):
        return "handling_object"
    if "lift" in text:
        return "lifting"
    if any(token in text for token in ("walk", "move")):
        return "walking"
    if any(token in text for token in ("inspect", "check", "scan")):
        return "inspecting"
    if any(token in text for token in ("interact", "talk", "handoff")):
        return "interacting"
    if any(token in text for token in ("stand", "wait", "idle")):
        return "standing"
    return "other"


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_activity_payload(payload: Any) -> dict[str, Any]:
    """Normalize legacy/new activity payloads into one backward-compatible shape."""
    if not isinstance(payload, dict):
        payload = {"raw": payload}

    if "relative_location" in payload and "zone_inference" not in payload:
        legacy_activity = _map_legacy_activity(payload.get("activity"))
        legacy_anomaly = _to_bool(payload.get("anomaly"))
        severity = payload.get("severity")
        payload = {
            "activity": legacy_activity or "other",
            "activity_description": str(payload.get("object_interaction") or "Legacy schema converted."),
            "objects_involved": [
                str(item)
                for item in payload.get("visible_tools", [])
                if isinstance(item, str) and item.strip()
            ],
            "zone_inference": str(payload.get("relative_location") or "unknown"),
            "interaction_with_others": None,
            "anomaly_flag": legacy_anomaly,
            "anomaly_reason": str(severity) if legacy_anomaly and severity else None,
            "supervisor_attention_recommended": legacy_anomaly,
            "confidence": 0.45,
            "reasoning": "Legacy payload normalized.",
            "_status": payload.get("_status", "legacy"),
        }
    else:
        candidate = dict(payload)
        candidate["activity"] = _map_legacy_activity(candidate.get("activity"))
        candidate.setdefault("activity_description", "No activity description provided.")
        objects = candidate.get("objects_involved")
        if not isinstance(objects, list):
            candidate["objects_involved"] = []
        else:
            candidate["objects_involved"] = [str(item) for item in objects if str(item).strip()]
        candidate.setdefault("zone_inference", "unknown")
        candidate["interaction_with_others"] = (
            str(candidate["interaction_with_others"])
            if candidate.get("interaction_with_others") not in (None, "")
            else None
        )
        candidate["anomaly_flag"] = _to_bool(candidate.get("anomaly_flag"))
        if candidate.get("anomaly_reason") in ("", "none", "null"):
            candidate["anomaly_reason"] = None
        candidate["supervisor_attention_recommended"] = _to_bool(
            candidate.get("supervisor_attention_recommended")
        )
        candidate["confidence"] = min(max(_to_float(candidate.get("confidence"), 0.5), 0.0), 1.0)
        candidate.setdefault("reasoning", "No explicit reasoning provided.")
        candidate.setdefault("_status", "ok")
        payload = candidate

    # Legacy aliases preserved for existing timeline/UI readers.
    payload["relative_location"] = payload.get("zone_inference", "unknown")
    payload["visible_tools"] = list(payload.get("objects_involved") or [])
    payload["object_interaction"] = payload.get("activity_description", "")
    payload["posture"] = payload.get("activity", "other")
    payload["anomaly"] = _to_bool(payload.get("anomaly_flag"))
    payload["severity"] = "medium" if payload["anomaly"] else None
    return payload
