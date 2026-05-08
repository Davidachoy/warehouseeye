"""Ask the Video tab rendering and interactions."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Callable

import streamlit as st

SUGGESTED_QUESTIONS = [
    "How many people worked this shift?",
    "What did the person in the orange vest with the bandana do?",
    "Who spent the most time in the central area?",
    "Are there any safety anomalies?",
]

NON_MEANINGFUL_REASONING = {
    "",
    "no explicit reasoning provided.",
    "no reasoning provided.",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
}

QueryFn = Callable[[str], dict[str, Any]]
TrackFn = Callable[[int], dict[str, Any]]


def _format_ts(timestamp: float) -> str:
    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    return f"{minutes:02d}:{seconds:02d}"


def _has_meaningful_reasoning(activity: dict[str, Any]) -> bool:
    raw_reasoning = activity.get("reasoning", "")
    if raw_reasoning is None:
        return False
    reasoning = str(raw_reasoning).strip().lower()
    return reasoning not in NON_MEANINGFUL_REASONING


def _resolve_crop_path(crop_path: str | None, workspace_root: Path) -> Path | None:
    if not crop_path:
        return None
    path = Path(crop_path)
    if path.exists():
        return path
    relative = workspace_root / crop_path
    if relative.exists():
        return relative
    return None


def _resolve_media_path(path_value: str | None, workspace_root: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.exists():
        return path
    candidate = workspace_root / path_value
    if candidate.exists():
        return candidate
    return None


def _render_timeline_expander(timeline_rows: list[dict[str, Any]], key_prefix: str) -> None:
    if not timeline_rows:
        return

    with st.expander("Timeline details", expanded=False):
        for index, entry in enumerate(timeline_rows[:40]):
            ts = float(entry.get("timestamp_sec", 0.0))
            ts_label = _format_ts(ts)
            track_id = entry.get("track_id", "n/a")
            activity = entry.get("activity", {})
            activity_label = activity.get("activity", "unknown") if isinstance(activity, dict) else "unknown"
            status_label = activity.get("_status", "ok") if isinstance(activity, dict) else "ok"
            status_suffix = f"  •  Status `{status_label}`" if status_label != "ok" else ""
            st.markdown(
                f"`{ts_label}`  •  Track `{track_id}`  •  Activity `{activity_label}`{status_suffix}",
            )
            st.button(
                f"Jump to {ts_label}",
                key=f"{key_prefix}-jump-{index}",
                help="Timestamp marker for live narration context.",
                type="tertiary",
                disabled=True,
            )


def _render_reasoning_details(
    timeline_rows: list[dict[str, Any]],
    workspace_root: Path,
    key_prefix: str,
) -> None:
    if not timeline_rows:
        return
    eligible_rows = [row for row in timeline_rows if isinstance(row.get("activity"), dict)]
    if not eligible_rows:
        return

    include_empty_reasoning = st.toggle(
        "Show entries without explicit reasoning",
        value=False,
        key=f"{key_prefix}-reasoning-toggle",
    )
    rows_with_reasoning = [
        row
        for row in eligible_rows
        if include_empty_reasoning or _has_meaningful_reasoning(row["activity"])
    ]
    if not rows_with_reasoning:
        if include_empty_reasoning:
            st.caption("No reasoning entries available for this response.")
        else:
            st.caption(
                "No explicit reasoning found in this response. "
                "Enable the toggle to inspect fallback entries."
            )
        return

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for row in rows_with_reasoning:
        key = (int(row.get("track_id", -1)), int(row.get("frame_idx", -1)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    with st.expander("VLM reasoning details", expanded=False):
        hidden_count = max(0, len(eligible_rows) - len(rows_with_reasoning))
        if not include_empty_reasoning and hidden_count > 0:
            st.caption(f"Hidden {hidden_count} fallback entries without explicit reasoning.")
        for index, row in enumerate(deduped[:8]):
            track_id = int(row.get("track_id", -1))
            frame_idx = int(row.get("frame_idx", -1))
            ts = _format_ts(float(row.get("timestamp_sec", 0.0)))
            activity = row.get("activity", {})
            status = str(activity.get("_status", "ok")) if isinstance(activity, dict) else "ok"
            reasoning = str(activity.get("reasoning", "")) if isinstance(activity, dict) else ""
            st.markdown(f"**Track `{track_id}`** · Frame `{frame_idx}` · Time `{ts}` · Status `{status}`")
            if status == "insufficient_resolution":
                st.warning("Marked as insufficient_resolution; no VLM call made for this keyframe.")
            if reasoning and _has_meaningful_reasoning(activity):
                st.info(reasoning)
            elif include_empty_reasoning:
                st.caption("No explicit reasoning provided.")
            st.json(activity)

            packet_paths = row.get("vlm_packet_paths")
            if not isinstance(packet_paths, list):
                packet_paths = activity.get("_vlm_packet_paths", []) if isinstance(activity, dict) else []
            resolved_paths = [
                resolved
                for resolved in (
                    _resolve_media_path(str(path_value), workspace_root)
                    for path_value in packet_paths[:3]
                )
                if resolved is not None
            ]
            if resolved_paths:
                cols = st.columns(min(3, len(resolved_paths)))
                for path_index, media_path in enumerate(resolved_paths):
                    with cols[path_index % len(cols)]:
                        st.image(str(media_path), width="stretch")
            st.divider()


def _render_crops(candidates: list[dict[str, Any]], workspace_root: Path, key_prefix: str) -> None:
    if not candidates:
        return

    st.markdown("**Highlighted crops**")
    cols = st.columns(min(4, len(candidates)))
    for index, candidate in enumerate(candidates[:8]):
        crop = _resolve_crop_path(candidate.get("crop_path"), workspace_root)
        caption = f"Track {candidate.get('track_id', '?')}"
        with cols[index % len(cols)]:
            if crop is not None:
                st.image(str(crop), caption=caption, width="stretch")
            else:
                st.caption(f"{caption}: crop unavailable")
            st.button(
                f"Select {caption}",
                key=f"{key_prefix}-select-{index}",
                type="secondary",
                disabled=True,
            )


def _handle_ambiguous_response(
    response: dict[str, Any],
    resolve_track: TrackFn,
    workspace_root: Path,
    key_prefix: str,
) -> dict[str, Any] | None:
    alternatives = response.get("alternatives", [])
    if not response.get("ambiguous") or not alternatives:
        return None

    st.warning("Multiple candidates found. Click one to refine.")
    cols = st.columns(min(4, len(alternatives)))
    for index, item in enumerate(alternatives):
        label = f"Track {item.get('track_id')}"
        color_tag = item.get("color_tag") or "unknown"
        crop = _resolve_crop_path(item.get("crop_path"), workspace_root)
        with cols[index % len(cols)]:
            st.markdown(f"**{label}**")
            st.caption(f"Color tag: `{color_tag}`")
            if crop is not None:
                st.image(str(crop), width="stretch")
            if st.button(
                f"Refine to {label}",
                key=f"{key_prefix}-refine-{index}",
                type="primary",
            ):
                payload = resolve_track(int(item["track_id"]))
                payload["narrative"] = (
                    f"Refined to track {item['track_id']} based on your selection."
                )
                return payload
    return None


def _next_message_id(role: str) -> str:
    counter = int(st.session_state.message_counter)
    st.session_state.message_counter = counter + 1
    return f"{role}-{counter}"


def _render_suggestion_chips() -> None:
    st.markdown("#### Suggested Questions")
    quick_questions = SUGGESTED_QUESTIONS[:2]
    quick_cols = st.columns(2)
    for idx, prompt in enumerate(quick_questions):
        if quick_cols[idx].button(prompt, key=f"suggest-quick-{idx}", width="stretch"):
            st.session_state.pending_prompt = prompt

    with st.expander("More suggestions", expanded=False):
        extra_questions = SUGGESTED_QUESTIONS[2:]
        if not extra_questions:
            st.caption("No more suggestions available.")
            return
        extra_cols = st.columns(2)
        for idx, prompt in enumerate(extra_questions):
            if extra_cols[idx % 2].button(prompt, key=f"suggest-extra-{idx}", width="stretch"):
                st.session_state.pending_prompt = prompt


def _render_messages_only(history: list[dict[str, Any]]) -> None:
    if not history:
        st.caption("Start the conversation by asking about people, zones, or safety.")
        return
    rows: list[str] = []
    for message in history:
        role = str(message.get("role", "assistant")).lower()
        safe_text = html.escape(str(message.get("content", ""))).replace("\n", "<br>")
        if role == "user":
            rows.append(
                f'<div class="we-chat-row user">'
                f'<div class="we-chat-bubble user">{safe_text}</div>'
                f'<div class="we-chat-avatar user">🙂</div>'
                f"</div>"
            )
        else:
            rows.append(
                f'<div class="we-chat-row assistant">'
                f'<div class="we-chat-avatar assistant">🤖</div>'
                f'<div class="we-chat-bubble assistant">{safe_text}</div>'
                f"</div>"
            )
    st.markdown(f'<div class="conversation-thread">{"".join(rows)}</div>', unsafe_allow_html=True)


def _conversation_container_height(history: list[dict[str, Any]]) -> int:
    """Estimate a dynamic chat viewport height from message volume."""
    if not history:
        return 280
    total_chars = sum(len(str(message.get("content", ""))) for message in history)
    estimated_lines = max(1, total_chars // 95)
    estimated = 220 + (len(history) * 36) + (estimated_lines * 8)
    return max(280, min(680, estimated))


def _render_response_details(history: list[dict[str, Any]], workspace_root: Path) -> None:
    assistant_messages = [msg for msg in history if msg.get("role") == "assistant"]
    if not assistant_messages:
        return

    st.markdown("#### Response Details")
    for index, message in enumerate(reversed(assistant_messages), start=1):
        response_label = f"Response {len(assistant_messages) - index + 1}"
        expanded = message.get("id") == st.session_state.get("active_response_id")
        with st.expander(response_label, expanded=expanded):
            timeline_rows = message.get("timeline", [])
            alternatives = message.get("alternatives", [])
            _render_timeline_expander(timeline_rows, key_prefix=f"details-timeline-{message['id']}")
            _render_reasoning_details(
                timeline_rows=timeline_rows,
                workspace_root=workspace_root,
                key_prefix=f"details-reasoning-{message['id']}",
            )
            _render_crops(
                alternatives,
                workspace_root,
                key_prefix=f"details-crops-{message['id']}",
            )


def _render_latest_photo_result(history: list[dict[str, Any]], workspace_root: Path) -> None:
    photo_messages = [
        message
        for message in history
        if message.get("role") == "assistant"
        and message.get("intent") == "keyframe_lookup"
        and message.get("alternatives")
    ]
    if not photo_messages:
        return
    latest = photo_messages[-1]
    st.markdown("#### Photo result")
    _render_crops(
        latest.get("alternatives", []),
        workspace_root,
        key_prefix=f"photo-result-{latest['id']}",
    )


def render_query_tab(
    *,
    video_id: str,
    run_query: QueryFn,
    resolve_track: TrackFn,
    workspace_root: Path,
) -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "query_cache" not in st.session_state:
        st.session_state.query_cache = {}
    if "message_counter" not in st.session_state:
        st.session_state.message_counter = 0
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "query_in_flight" not in st.session_state:
        st.session_state.query_in_flight = False
    if "active_response_id" not in st.session_state:
        st.session_state.active_response_id = None
    if "scroll_anchor" not in st.session_state:
        st.session_state.scroll_anchor = None

    _render_suggestion_chips()
    st.markdown("#### Conversation")
    with st.container(height=_conversation_container_height(st.session_state.chat_history), border=False):
        _render_messages_only(st.session_state.chat_history)

    user_prompt = st.chat_input("Ask a question about this shift...")
    if st.session_state.get("pending_prompt"):
        user_prompt = st.session_state.pop("pending_prompt")
    _render_latest_photo_result(st.session_state.chat_history, workspace_root)
    _render_response_details(st.session_state.chat_history, workspace_root)

    if not user_prompt:
        return

    user_message_id = _next_message_id("user")
    st.session_state.chat_history.append({"role": "user", "content": user_prompt, "id": user_message_id})

    cache_key = (video_id, user_prompt.strip().lower())
    narrative = "No matching track was found for that query."
    timeline: list[dict[str, Any]] = []
    alternatives: list[dict[str, Any]] = []
    intent: str | None = None
    assistant_message_id = _next_message_id("assistant")
    st.session_state.query_in_flight = True
    try:
        with st.spinner("Querying timeline intelligence..."):
            if cache_key in st.session_state.query_cache:
                response = st.session_state.query_cache[cache_key]
            else:
                response = run_query(user_prompt)
                st.session_state.query_cache[cache_key] = response

        narrative = response.get("narrative") or "No matching track was found for that query."
        timeline = response.get("timeline", [])
        alternatives = response.get("alternatives", [])
        intent = response.get("intent")

        if not timeline and not alternatives and "no" in narrative.lower():
            narrative = (
                f"{narrative}\n\nNo strong match found yet. "
                "Try color, position, or activity details."
            )
    finally:
        st.session_state.query_in_flight = False

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": narrative,
            "timeline": timeline,
            "alternatives": alternatives,
            "intent": intent,
            "id": assistant_message_id,
        }
    )
    st.session_state.active_response_id = assistant_message_id
    st.session_state.scroll_anchor = assistant_message_id
    st.rerun()
