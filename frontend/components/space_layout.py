"""Layout primitives specific to HF Space presentation."""

from __future__ import annotations

import streamlit as st

ACCENT_ORANGE = "#FF6B35"
ACCENT_CYAN = "#00B8D9"


def apply_space_theme() -> None:
    st.set_page_config(
        page_title="WarehouseEye",
        page_icon="📦",
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
        .hero {{
            border: 1px solid rgba(255, 107, 53, 0.35);
            border-radius: 14px;
            background: linear-gradient(120deg, rgba(255, 107, 53, 0.10), rgba(0, 184, 217, 0.12));
            padding: 18px 20px 8px;
            margin-bottom: 16px;
        }}
        .hero-title {{
            font-size: 2.25rem;
            line-height: 1.15;
            margin: 0;
        }}
        .hero-subtitle {{
            color: #C9D6E1;
            margin-top: 8px;
            margin-bottom: 12px;
        }}
        .badge {{
            display: inline-block;
            border: 1px solid rgba(0, 184, 217, 0.45);
            border-radius: 999px;
            padding: 4px 10px;
            margin-right: 8px;
            margin-bottom: 8px;
            color: #B9F2FF;
            font-family: "JetBrains Mono", "Fira Code", monospace;
            font-size: 12px;
        }}
        .how-it-works p {{
            color: #D4DEE7;
        }}
        .like-cta {{
            border: 1px solid rgba(255, 107, 53, 0.45);
            border-radius: 10px;
            background: rgba(255, 107, 53, 0.12);
            padding: 10px 12px;
            margin-top: 10px;
            font-size: 0.95rem;
            color: #FFE2D6;
        }}
        .app-footer {{
            margin-top: 22px;
            padding-top: 12px;
            border-top: 1px solid rgba(255, 107, 53, 0.3);
            color: #B9C2CC;
            font-size: 13px;
            font-family: "JetBrains Mono", "Fira Code", monospace;
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
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(github_url: str, hf_space_url: str) -> None:
    st.markdown(
        """
        <div class="hero">
          <h1 class="hero-title">WarehouseEye</h1>
          <p class="hero-subtitle">Operational intelligence for warehouse CCTV — on AMD MI300X</p>
          <div>
            <span class="badge">Built on AMD MI300X</span>
            <span class="badge">Powered by Qwen3-VL</span>
            <span class="badge">MIT License</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    action_col_1, action_col_2 = st.columns(2)
    action_col_1.link_button("View on GitHub", github_url, use_container_width=True)
    action_col_2.link_button("⭐ Like this Space", hf_space_url, use_container_width=True)


def render_how_it_works() -> None:
    with st.expander("How it works", expanded=False):
        st.markdown(
            """
            <div class="how-it-works">
              <p><strong>The problem:</strong> warehouse operators often review long CCTV footage manually, which is slow, expensive, and easy to miss critical safety or process events.</p>
              <p><strong>The solution:</strong> WarehouseEye is an open-source pipeline that combines tracking with a vision-language model to convert long videos into searchable operational timelines.</p>
              <p><strong>Why AMD MI300X:</strong> its 192 GB HBM memory enables high-capacity multimodal inference on a single node, avoiding multi-GPU coordination complexity for this workload.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_space_footer(github_url: str, paper_url: str, twitter_url: str) -> None:
    links: list[str] = [
        f'<a href="{github_url}" target="_blank" rel="noopener noreferrer">GitHub repo</a>',
        f'<a href="{twitter_url}" target="_blank" rel="noopener noreferrer">Author X/Twitter</a>',
    ]
    if paper_url:
        links.insert(
            1,
            f'<a href="{paper_url}" target="_blank" rel="noopener noreferrer">Technical paper</a>',
        )

    st.markdown(
        f"""
        <div class="app-footer">
          <div>AMD · Hugging Face · lablab.ai · Qwen</div>
          <div>{" | ".join(links)}</div>
          <div>Demo running on CPU with pre-rendered results. Real-time inference requires AMD MI300X.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
