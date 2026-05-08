"""Shared layout primitives and theme styling."""

from __future__ import annotations

import streamlit as st

ACCENT_ORANGE = "#FF6B35"
ACCENT_CYAN = "#00B8D9"


def apply_theme() -> None:
    st.set_page_config(
        page_title="WarehouseEye Live Demo",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        f"""
        <style>
        :root {{
            --accent-orange: {ACCENT_ORANGE};
            --accent-cyan: {ACCENT_CYAN};
        }}
        .stApp {{
            background: radial-gradient(circle at top, #1b2329 0%, #10151a 55%, #0b0f13 100%);
            color: #E8EEF2;
        }}
        .main .block-container {{
            max-width: 1120px;
            padding-top: 1.25rem;
            padding-bottom: 1.25rem;
        }}
        h1, h2, h3 {{
            color: #F7FAFC;
        }}
        .logo-box {{
            border: 1px solid rgba(255, 107, 53, 0.45);
            border-radius: 10px;
            padding: 8px 14px;
            display: inline-block;
            background: rgba(0, 184, 217, 0.08);
            font-family: "JetBrains Mono", "Fira Code", monospace;
            font-size: 12px;
            letter-spacing: 1.8px;
            color: var(--accent-cyan);
            margin-bottom: 8px;
        }}
        .industrial-header {{
            border-bottom: 2px solid rgba(255, 107, 53, 0.4);
            padding-bottom: 10px;
            margin-bottom: 14px;
        }}
        .data-mono, .stMetricValue, .stMetricLabel, .stDataFrame, .stTable {{
            font-family: "JetBrains Mono", "Fira Code", monospace !important;
        }}
        .status-pill {{
            display: inline-block;
            border: 1px solid rgba(0, 184, 217, 0.45);
            border-radius: 999px;
            padding: 4px 10px;
            margin-right: 8px;
            color: #B9F2FF;
            font-family: "JetBrains Mono", "Fira Code", monospace;
            font-size: 12px;
        }}
        [data-testid="stChatMessage"] {{
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 14px;
            background: rgba(8, 15, 23, 0.55);
            margin-bottom: 0.6rem;
            padding: 0.2rem 0.4rem;
            width: 100%;
        }}
        [data-testid="stChatMessageContent"] {{
            min-width: 0;
            width: 100%;
            overflow: hidden;
        }}
        [data-testid="stChatMessageContent"] * {{
            max-width: 100%;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
            word-break: break-word !important;
        }}
        [data-testid="stChatMessageContent"] p,
        [data-testid="stChatMessageContent"] div,
        [data-testid="stChatMessageContent"] span,
        [data-testid="stChatMessageContent"] li {{
            margin-bottom: 0;
            line-height: 1.45;
        }}
        [data-testid="stChatInput"] {{
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            padding-top: 0.45rem;
            background: rgba(6, 10, 18, 0.62);
            backdrop-filter: blur(4px);
        }}
        [data-testid="stChatInput"] textarea {{
            border-radius: 12px !important;
        }}
        .conversation-thread {{
            display: flex;
            flex-direction: column;
            gap: 0.55rem;
            width: 100%;
        }}
        .we-chat-row {{
            display: flex;
            align-items: flex-end;
            gap: 0.55rem;
            width: 100%;
        }}
        .we-chat-row.assistant {{
            justify-content: flex-start;
        }}
        .we-chat-row.user {{
            justify-content: flex-end;
        }}
        .we-chat-avatar {{
            width: 28px;
            height: 28px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            flex: 0 0 28px;
            border: 1px solid rgba(255, 255, 255, 0.22);
            background: rgba(255, 255, 255, 0.06);
        }}
        .we-chat-avatar.user {{
            background: rgba(255, 75, 75, 0.22);
            border-color: rgba(255, 75, 75, 0.55);
        }}
        .we-chat-avatar.assistant {{
            background: rgba(255, 159, 10, 0.22);
            border-color: rgba(255, 159, 10, 0.55);
        }}
        .we-chat-bubble {{
            max-width: min(78%, 860px);
            min-width: 120px;
            border-radius: 14px;
            padding: 10px 12px;
            line-height: 1.45;
            font-size: 0.95rem;
            white-space: normal;
            overflow-wrap: anywhere;
            word-break: break-word;
            border: 1px solid rgba(255, 255, 255, 0.13);
        }}
        .we-chat-bubble.assistant {{
            background: rgba(9, 20, 30, 0.72);
        }}
        .we-chat-bubble.user {{
            background: rgba(25, 30, 39, 0.75);
            border-color: rgba(255, 255, 255, 0.2);
        }}
        footer {{
            visibility: hidden;
        }}
        .app-footer {{
            margin-top: 20px;
            padding-top: 12px;
            border-top: 1px solid rgba(255, 107, 53, 0.3);
            color: #B9C2CC;
            font-size: 13px;
            font-family: "JetBrains Mono", "Fira Code", monospace;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown('<div class="logo-box">WAREHOUSEEYE / CCTV OPS</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="industrial-header">
            <h1>WarehouseEye Control Room</h1>
            <p>Real-time multimodal tracking and natural-language video intelligence.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="app-footer">
            Built on FieldMind. AMD Developer Hackathon 2026. MIT License. github.com/[user]/warehouseeye
        </div>
        """,
        unsafe_allow_html=True,
    )
