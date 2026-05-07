"""Performance dashboard tab."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st


def _fmt_currency(value: float) -> str:
    return f"${value:,.2f}"


def _build_comparison_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"System": "WarehouseEye MI300X", "Cost/min video": "$0.04", "Privacy": "Local", "Open Source": "Yes"},
            {"System": "GPT-4V API est.", "Cost/min video": "$1.20", "Privacy": "Cloud", "Open Source": "No"},
            {"System": "BriefCam", "Cost/min video": "~$50K/year", "Privacy": "Either", "Open Source": "No"},
        ]
    )


def render_performance_dashboard(
    benchmark_payload: dict[str, Any],
    throughput_rows: list[dict[str, float]],
) -> None:
    wall_time_sec = float(benchmark_payload.get("wall_time_sec", 0.0))
    total_cost_usd = float(benchmark_payload.get("total_cost_usd", 0.0))
    cost_per_min = (total_cost_usd / wall_time_sec * 60.0) if wall_time_sec > 0 else 0.0

    row1 = st.columns(5)
    row1[0].metric("Frames analyzed", f"{int(benchmark_payload.get('frames_analyzed', 0))}")
    row1[1].metric("Tokens/sec sustained", f"{float(benchmark_payload.get('tokens_per_second_avg', 0.0)):.2f}")
    row1[2].metric("Latency per crop (ms)", f"{float(benchmark_payload.get('latency_per_crop_ms', 0.0)):.2f}")
    row1[3].metric("Total cost USD", _fmt_currency(total_cost_usd))
    row1[4].metric("Cost per minute video", _fmt_currency(cost_per_min))

    st.markdown("#### WarehouseEye on AMD MI300X vs alternatives")
    st.dataframe(_build_comparison_table(), width="stretch", hide_index=True)

    st.markdown("#### Throughput over processing time")
    if throughput_rows:
        df = pd.DataFrame(throughput_rows)
        fig = px.line(df, x="elapsed_sec", y="tokens_per_sec", markers=True)
        fig.update_layout(
            paper_bgcolor="#0F141A",
            plot_bgcolor="#0F141A",
            font={"family": "JetBrains Mono, Fira Code, monospace", "color": "#E8EEF2"},
            xaxis_title="Time (sec)",
            yaxis_title="Tokens/sec",
            margin={"l": 10, "r": 10, "t": 20, "b": 10},
        )
        st.plotly_chart(fig, width="stretch", theme=None)
    else:
        st.info("Throughput data will appear after analyze/query actions.")
