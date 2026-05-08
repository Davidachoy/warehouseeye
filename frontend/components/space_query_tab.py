"""Ask-the-video tab for prerendered HF Space demo."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import streamlit as st

TimelineRows = list[dict[str, Any]]


@dataclass(frozen=True)
class CannedAnswer:
    narrative: str
    timeline: TimelineRows
    crops: TimelineRows


@dataclass(frozen=True)
class CannedQuery:
    prompt: str
    aliases: list[str]
    responder: Callable[[TimelineRows], CannedAnswer]


def _normalized(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _format_ts(timestamp: float) -> str:
    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)
    return f"{minutes:02d}:{seconds:02d}"


def _track_rows(timeline_rows: TimelineRows) -> dict[int, TimelineRows]:
    grouped: dict[int, TimelineRows] = {}
    for row in timeline_rows:
        track_id = int(row.get("track_id", -1))
        if track_id < 0:
            continue
        grouped.setdefault(track_id, []).append(row)
    return grouped


def _crops_from_rows(rows: TimelineRows, limit: int = 6) -> TimelineRows:
    seen: set[tuple[int, int]] = set()
    selected: TimelineRows = []
    for row in rows:
        key = (int(row.get("track_id", -1)), int(row.get("frame_idx", -1)))
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def _answer_people_count(timeline_rows: TimelineRows) -> CannedAnswer:
    tracks = sorted(_track_rows(timeline_rows).keys())
    narrative = f"Detected {len(tracks)} tracked worker(s) in this pre-rendered shift: {', '.join(f'#{t}' for t in tracks)}."
    return CannedAnswer(narrative=narrative, timeline=timeline_rows[:15], crops=_crops_from_rows(timeline_rows))


def _answer_longest_presence(timeline_rows: TimelineRows) -> CannedAnswer:
    grouped = _track_rows(timeline_rows)
    if not grouped:
        return CannedAnswer("No tracks available for this video.", [], [])

    best_track = -1
    best_duration = -1.0
    for track_id, rows in grouped.items():
        timestamps = [float(row.get("timestamp_sec", 0.0)) for row in rows]
        duration = max(timestamps) - min(timestamps)
        if duration > best_duration:
            best_track = track_id
            best_duration = duration

    winner_rows = grouped[best_track]
    start_ts = _format_ts(float(winner_rows[0].get("timestamp_sec", 0.0)))
    end_ts = _format_ts(float(winner_rows[-1].get("timestamp_sec", 0.0)))
    narrative = (
        f"Track {best_track} has the longest observed presence, roughly {best_duration:.1f}s "
        f"(from {start_ts} to {end_ts})."
    )
    return CannedAnswer(narrative=narrative, timeline=winner_rows[:20], crops=_crops_from_rows(winner_rows))


def _answer_track_one(timeline_rows: TimelineRows) -> CannedAnswer:
    grouped = _track_rows(timeline_rows)
    rows = grouped.get(1) or []
    if not rows:
        return CannedAnswer("Track 1 is not present in this video.", [], [])
    first = rows[0]
    summary = first.get("narrative_summary") or "No narrative summary available."
    narrative = f"Track 1 summary: {summary}"
    return CannedAnswer(narrative=narrative, timeline=rows[:20], crops=_crops_from_rows(rows))


def _answer_yellow_worker(timeline_rows: TimelineRows) -> CannedAnswer:
    rows = [row for row in timeline_rows if row.get("color_tag") == "yellow_top"]
    if not rows:
        return CannedAnswer("No worker tagged `yellow_top` was found in this sample.", [], [])
    grouped = _track_rows(rows)
    track_id = sorted(grouped.keys())[0]
    summary = grouped[track_id][0].get("narrative_summary") or "No narrative summary available."
    narrative = f"Yellow-top worker maps to track {track_id}. {summary}"
    return CannedAnswer(narrative=narrative, timeline=grouped[track_id][:20], crops=_crops_from_rows(grouped[track_id]))


def _answer_anomalies(timeline_rows: TimelineRows) -> CannedAnswer:
    anomalous = [
        row
        for row in timeline_rows
        if isinstance(row.get("activity"), dict) and bool(row["activity"].get("anomaly"))
    ]
    if not anomalous:
        return CannedAnswer(
            "No explicit safety anomaly flags were found in this pre-rendered timeline.",
            timeline_rows[:12],
            _crops_from_rows(timeline_rows),
        )
    narrative = f"Found {len(anomalous)} anomaly-tagged event(s)."
    return CannedAnswer(narrative=narrative, timeline=anomalous[:30], crops=_crops_from_rows(anomalous))


def _answer_activities(timeline_rows: TimelineRows) -> CannedAnswer:
    counts: dict[str, int] = {}
    for row in timeline_rows:
        activity = row.get("activity")
        if isinstance(activity, dict):
            label = str(activity.get("activity") or "unknown")
        else:
            label = "unknown"
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return CannedAnswer("No activity labels available.", [], [])
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    narrative = "Observed activities: " + ", ".join(f"{name} ({count})" for name, count in ordered[:8]) + "."
    return CannedAnswer(narrative=narrative, timeline=timeline_rows[:20], crops=_crops_from_rows(timeline_rows))


def _answer_busiest_moment(timeline_rows: TimelineRows) -> CannedAnswer:
    moment_counts: dict[float, set[int]] = {}
    for row in timeline_rows:
        ts = float(row.get("timestamp_sec", 0.0))
        track_id = int(row.get("track_id", -1))
        moment_counts.setdefault(ts, set()).add(track_id)
    if not moment_counts:
        return CannedAnswer("No timestamp data available.", [], [])
    peak_ts, tracks = max(moment_counts.items(), key=lambda item: len(item[1]))
    peak_rows = [row for row in timeline_rows if float(row.get("timestamp_sec", 0.0)) == peak_ts]
    narrative = (
        f"Busiest timestamp is {_format_ts(peak_ts)} with {len(tracks)} concurrent tracked worker(s): "
        f"{', '.join(f'#{track}' for track in sorted(tracks))}."
    )
    return CannedAnswer(narrative=narrative, timeline=peak_rows, crops=_crops_from_rows(peak_rows))


def _answer_all_tracks(timeline_rows: TimelineRows) -> CannedAnswer:
    grouped = _track_rows(timeline_rows)
    if not grouped:
        return CannedAnswer("No tracks available in this timeline.", [], [])
    descriptions: list[str] = []
    selected: TimelineRows = []
    for track_id in sorted(grouped.keys()):
        rows = grouped[track_id]
        first_ts = float(rows[0].get("timestamp_sec", 0.0))
        last_ts = float(rows[-1].get("timestamp_sec", 0.0))
        descriptions.append(f"#{track_id} ({_format_ts(first_ts)}-{_format_ts(last_ts)})")
        selected.extend(rows[:1])
    narrative = f"Tracked workers in this video: {', '.join(descriptions)}."
    return CannedAnswer(narrative=narrative, timeline=selected, crops=_crops_from_rows(timeline_rows, limit=8))


CANNED_QUERIES: list[CannedQuery] = [
    CannedQuery(
        prompt="How many people worked this shift?",
        aliases=["how many people", "people worked", "worker count", "number of workers"],
        responder=_answer_people_count,
    ),
    CannedQuery(
        prompt="Who stayed the longest in view?",
        aliases=["stayed the longest", "longest activity", "longest time"],
        responder=_answer_longest_presence,
    ),
    CannedQuery(
        prompt="Show me track 1",
        aliases=["track 1", "show track one"],
        responder=_answer_track_one,
    ),
    CannedQuery(
        prompt="What did the yellow-top worker do?",
        aliases=["yellow-top", "yellow top", "orange vest"],
        responder=_answer_yellow_worker,
    ),
    CannedQuery(
        prompt="Are there any safety anomalies?",
        aliases=["safety anomalies", "anomaly", "unsafe"],
        responder=_answer_anomalies,
    ),
    CannedQuery(
        prompt="What activities happened?",
        aliases=["activities happened", "what activities", "activity breakdown"],
        responder=_answer_activities,
    ),
    CannedQuery(
        prompt="When was the warehouse busiest?",
        aliases=["warehouse busiest", "busiest", "most people at once"],
        responder=_answer_busiest_moment,
    ),
    CannedQuery(
        prompt="Show me all tracks",
        aliases=["all tracks", "list tracks", "show tracks"],
        responder=_answer_all_tracks,
    ),
]


def _match_query(prompt: str) -> CannedQuery | None:
    normalized = _normalized(prompt)
    for item in CANNED_QUERIES:
        for alias in item.aliases:
            if _normalized(alias) in normalized:
                return item
    return None


def _resolve_crop_path(crop_path: str | None, workspace_root: Path) -> Path | None:
    if not crop_path:
        return None
    path = Path(crop_path)
    if path.exists():
        return path
    fallback = workspace_root / crop_path
    if fallback.exists():
        return fallback
    return None


def _render_timeline_expander(timeline_rows: TimelineRows, key_prefix: str) -> None:
    if not timeline_rows:
        return
    with st.expander("Timeline details", expanded=False):
        for index, row in enumerate(timeline_rows[:40]):
            ts = _format_ts(float(row.get("timestamp_sec", 0.0)))
            track_id = row.get("track_id", "n/a")
            activity = row.get("activity")
            label = activity.get("activity", "unknown") if isinstance(activity, dict) else "unknown"
            st.markdown(f"`{ts}`  •  Track `{track_id}`  •  Activity `{label}`")
            st.button(
                f"Jump to {ts}",
                key=f"{key_prefix}-jump-{index}",
                type="tertiary",
                disabled=True,
            )


def _render_crops(candidates: TimelineRows, workspace_root: Path, key_prefix: str) -> None:
    if not candidates:
        return
    st.markdown("**Highlighted crops**")
    cols = st.columns(min(4, len(candidates)))
    for index, row in enumerate(candidates[:8]):
        crop = _resolve_crop_path(row.get("crop_path"), workspace_root)
        caption = f"Track {row.get('track_id', '?')}"
        with cols[index % len(cols)]:
            if crop is not None:
                st.image(str(crop), caption=caption, use_container_width=True)
            else:
                st.caption(f"{caption}: crop unavailable")
            st.button(
                f"Select {caption}",
                key=f"{key_prefix}-select-{index}",
                type="secondary",
                disabled=True,
            )


def _next_message_id(role: str) -> str:
    counter = int(st.session_state.space_message_counter)
    st.session_state.space_message_counter = counter + 1
    return f"{role}-{counter}"


def _render_suggestion_chips() -> None:
    st.markdown("#### Suggested Questions")
    quick = CANNED_QUERIES[:2]
    quick_cols = st.columns(2)
    for idx, query in enumerate(quick):
        if quick_cols[idx].button(query.prompt, key=f"space-suggest-quick-{idx}", width="stretch"):
            st.session_state.space_pending_prompt = query.prompt

    with st.expander("More suggestions", expanded=False):
        extra = CANNED_QUERIES[2:]
        if not extra:
            st.caption("No more suggestions available.")
            return
        cols = st.columns(2)
        for idx, query in enumerate(extra):
            if cols[idx % 2].button(query.prompt, key=f"space-suggest-extra-{idx}", width="stretch"):
                st.session_state.space_pending_prompt = query.prompt


def _render_messages_only(history: list[dict[str, Any]]) -> None:
    if not history:
        st.caption("Start the conversation with one of the suggested prompts.")
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
        expanded = message.get("id") == st.session_state.get("space_active_response_id")
        with st.expander(response_label, expanded=expanded):
            _render_timeline_expander(
                message.get("timeline", []),
                key_prefix=f"space-details-timeline-{message['id']}",
            )
            _render_crops(
                message.get("crops", []),
                workspace_root,
                key_prefix=f"space-details-crops-{message['id']}",
            )


def _fallback_message(github_url: str) -> str:
    query_list = "; ".join(f'"{item.prompt}"' for item in CANNED_QUERIES)
    return (
        "This is a pre-rendered demo. Available queries are: "
        f"{query_list}. "
        f"For free queries, deploy the full system from GitHub: {github_url}"
    )


def render_space_query_tab(*, timeline_rows: TimelineRows, workspace_root: Path, github_url: str) -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "query_cache" not in st.session_state:
        st.session_state.query_cache = {}
    if "space_message_counter" not in st.session_state:
        st.session_state.space_message_counter = 0
    if "space_pending_prompt" not in st.session_state:
        st.session_state.space_pending_prompt = None
    if "space_query_in_flight" not in st.session_state:
        st.session_state.space_query_in_flight = False
    if "space_active_response_id" not in st.session_state:
        st.session_state.space_active_response_id = None
    if "space_scroll_anchor" not in st.session_state:
        st.session_state.space_scroll_anchor = None

    _render_suggestion_chips()
    st.markdown("#### Conversation")
    with st.container(
        height=_conversation_container_height(st.session_state.chat_history),
        border=False,
    ):
        _render_messages_only(st.session_state.chat_history)

    user_prompt = st.chat_input("Ask a question about this pre-rendered shift...")
    if st.session_state.get("space_pending_prompt"):
        user_prompt = st.session_state.pop("space_pending_prompt")
    _render_response_details(st.session_state.chat_history, workspace_root)
    if not user_prompt:
        return

    user_message_id = _next_message_id("user")
    st.session_state.chat_history.append({"role": "user", "content": user_prompt, "id": user_message_id})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    cache_key = _normalized(user_prompt)
    assistant_message_id = _next_message_id("assistant")
    answer = CannedAnswer("No answer available.", [], [])
    st.session_state.space_query_in_flight = True
    try:
        with st.spinner("Searching pre-rendered responses..."):
            if cache_key in st.session_state.query_cache:
                answer = st.session_state.query_cache[cache_key]
            else:
                matched = _match_query(user_prompt)
                if matched is None:
                    answer = CannedAnswer(_fallback_message(github_url), [], [])
                else:
                    answer = matched.responder(timeline_rows)
                st.session_state.query_cache[cache_key] = answer
    finally:
        st.session_state.space_query_in_flight = False

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer.narrative,
            "timeline": answer.timeline,
            "crops": answer.crops,
            "id": assistant_message_id,
        }
    )
    st.session_state.space_active_response_id = assistant_message_id
    st.session_state.space_scroll_anchor = assistant_message_id
    st.rerun()
