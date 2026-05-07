"""Ask the Video tab rendering and interactions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import streamlit as st

SUGGESTED_QUESTIONS = [
    "How many people worked this shift?",
    "What did the person in the orange vest with the bandana do?",
    "Who spent the most time in the central area?",
    "Are there any safety anomalies?",
]

QueryFn = Callable[[str], dict[str, Any]]
TrackFn = Callable[[int], dict[str, Any]]


def _format_ts(timestamp: float) -> str:
    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    return f"{minutes:02d}:{seconds:02d}"


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
            st.markdown(
                f"`{ts_label}`  •  Track `{track_id}`  •  Activity `{activity_label}`",
            )
            st.button(
                f"Jump to {ts_label}",
                key=f"{key_prefix}-jump-{index}",
                help="Timestamp marker for live narration context.",
                type="tertiary",
            )


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

    st.markdown("#### Suggested Questions")
    suggestion_cols = st.columns(2)
    for idx, prompt in enumerate(SUGGESTED_QUESTIONS):
        if suggestion_cols[idx % 2].button(prompt, key=f"suggest-{idx}", width="stretch"):
            st.session_state.pending_prompt = prompt

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("timeline"):
                _render_timeline_expander(message["timeline"], key_prefix=message["id"])
            if message.get("alternatives"):
                _render_crops(message["alternatives"], workspace_root, key_prefix=f"crops-{message['id']}")

    user_prompt = st.chat_input("Ask a question about this shift...")
    if st.session_state.get("pending_prompt"):
        user_prompt = st.session_state.pop("pending_prompt")

    if not user_prompt:
        return

    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt, "id": f"user-{len(st.session_state.chat_history)}"}
    )
    with st.chat_message("user"):
        st.markdown(user_prompt)

    cache_key = (video_id, user_prompt.strip().lower())
    with st.chat_message("assistant"):
        with st.spinner("Querying timeline intelligence..."):
            if cache_key in st.session_state.query_cache:
                response = st.session_state.query_cache[cache_key]
            else:
                response = run_query(user_prompt)
                st.session_state.query_cache[cache_key] = response

        refined = _handle_ambiguous_response(
            response=response,
            resolve_track=resolve_track,
            workspace_root=workspace_root,
            key_prefix=f"ambig-{len(st.session_state.chat_history)}",
        )
        final_response = refined or response
        narrative = final_response.get("narrative") or "No matching track was found for that query."
        timeline = final_response.get("timeline", [])
        alternatives = final_response.get("alternatives", [])

        if not timeline and not alternatives and "no" in narrative.lower():
            st.info("No strong match found yet. Try color, position, or activity details.")

        st.markdown(narrative)
        _render_timeline_expander(timeline, key_prefix=f"timeline-{len(st.session_state.chat_history)}")
        _render_crops(alternatives, workspace_root, key_prefix=f"alts-{len(st.session_state.chat_history)}")

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": narrative,
            "timeline": timeline,
            "alternatives": alternatives,
            "id": f"assistant-{len(st.session_state.chat_history)}",
        }
    )
