"""
Plotly charts for scenario comparison and 3-year forecasts.
"""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from optimizer.engine import OptimizationResult
from models.revenue_model import YearlyRevenue


def _scenario_color(name: str) -> str:
    colors = {
        "Current Pricing (Manual)": "#6c757d",
        "Margin Optimized": "#28a745",
        "Revenue Optimized": "#007bff",
        "Best Strategy": "#fd7e14",
    }
    return colors.get(name, "#17a2b8")


def render_comparison_chart(
    scenarios: dict[str, dict],
) -> None:
    """
    Bar chart comparing Total Revenue, Total Margin, and Win Prob
    across all scenarios.

    scenarios: {label: {"yearly": dict[int,YearlyRevenue], "win_prob": float}}
    """
    labels = list(scenarios.keys())
    revenues = []
    margins = []
    win_probs = []

    for label, data in scenarios.items():
        yearly: dict[int, YearlyRevenue] = data["yearly"]
        revenues.append(sum(yr.total_revenue for yr in yearly.values()))
        margins.append(sum(yr.margin for yr in yearly.values()))
        win_probs.append(data["win_prob"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="3-Year Revenue",
        x=labels,
        y=revenues,
        marker_color=[_scenario_color(l) for l in labels],
        opacity=0.7,
    ))
    fig.add_trace(go.Bar(
        name="3-Year Margin",
        x=labels,
        y=margins,
        marker_color=[_scenario_color(l) for l in labels],
        opacity=1.0,
    ))

    fig.update_layout(
        title="Scenario Comparison: Revenue vs Margin",
        barmode="group",
        yaxis_title="Dollars ($)",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=labels,
        y=[wp * 100 for wp in win_probs],
        marker_color=[_scenario_color(l) for l in labels],
        text=[f"{wp:.0%}" for wp in win_probs],
        textposition="outside",
    ))
    fig2.update_layout(
        title="Win Probability by Scenario",
        yaxis_title="Win Probability (%)",
        yaxis_range=[0, 100],
        template="plotly_white",
        height=350,
    )
    st.plotly_chart(fig2, use_container_width=True)


def render_yearly_trend_chart(
    scenarios: dict[str, dict],
) -> None:
    """Line chart showing revenue and margin by year for each scenario."""
    fig_rev = go.Figure()
    fig_margin = go.Figure()

    years = [1, 2, 3]

    for label, data in scenarios.items():
        yearly: dict[int, YearlyRevenue] = data["yearly"]
        color = _scenario_color(label)

        rev_vals = [yearly[y].total_revenue for y in years]
        margin_vals = [yearly[y].margin for y in years]

        fig_rev.add_trace(go.Scatter(
            x=[f"Year {y}" for y in years],
            y=rev_vals,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=3),
        ))

        fig_margin.add_trace(go.Scatter(
            x=[f"Year {y}" for y in years],
            y=margin_vals,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=3),
        ))

    fig_rev.update_layout(
        title="Revenue Trend by Year",
        yaxis_title="Revenue ($)",
        template="plotly_white",
        height=400,
    )
    fig_margin.update_layout(
        title="Margin Trend by Year",
        yaxis_title="Margin ($)",
        template="plotly_white",
        height=400,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_rev, use_container_width=True)
    with c2:
        st.plotly_chart(fig_margin, use_container_width=True)
