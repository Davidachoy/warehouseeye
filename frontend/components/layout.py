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
