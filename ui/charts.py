"""
Plotly charts for scenario comparison and 3-year forecasts.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from optimizer.engine import OptimizationResult
from models.revenue_model import YearlyRevenue, PricingScenario


def _scenario_color(name: str) -> str:
    colors = {
        "Pre Discount Pricing Today": "#dc3545",
        "Standard Pricing Today (Manual)": "#6c757d",
        "Margin % Optimized": "#1B6AC9",
        "Take Rate Optimized": "#28a745",
        "LTV Optimized": "#fd7e14",
    }
    return colors.get(name, "#17a2b8")


TODAY_COLOR = "#9333ea"


# ── Executive Comparison Table ────────────────────────────────────────

def render_executive_table(
    scenarios: dict[str, dict],
    today: dict,
) -> None:
    """Color-coded comparison table: all scenarios + Today's baseline."""
    st.subheader("Executive Summary")

    rows = []
    for label, data in scenarios.items():
        yearly = data["yearly"]
        total_rev = sum(yr.total_revenue for yr in yearly.values())
        total_margin = sum(yr.margin for yr in yearly.values())
        total_vol = sum(
            yr.total_revenue / yr.take_rate
            for yr in yearly.values() if yr.take_rate > 0
        )
        rows.append({
            "Scenario": label,
            "Win Rate": data["win_prob"],
            "Take Rate": total_rev / total_vol if total_vol > 0 else 0,
            "Margin %": total_margin / total_rev if total_rev > 0 else 0,
            "3-Year Revenue": total_rev,
            "3-Year Margin $": total_margin,
        })

    df = pd.DataFrame(rows)

    today_wr = today["win_rate"]
    today_tr = today["take_rate"]
    today_mp = today["margin_pct"]

    def _delta_style(val: float, ref: float, fmt: str = "pct") -> str:
        diff = val - ref
        if abs(diff) < 0.001:
            arrow, clr = "—", "#666"
        elif diff > 0:
            arrow, clr = "▲", "#16a34a"
        else:
            arrow, clr = "▼", "#dc2626"
        if fmt == "pct":
            return f'<span style="color:{clr}">{arrow} {abs(diff):.1%}</span>'
        return f'<span style="color:{clr}">{arrow} ${abs(diff):,.0f}</span>'

    header = (
        "<table style='width:100%; border-collapse:collapse; font-size:14px;'>"
        "<tr style='border-bottom:2px solid #ddd; background:#f8f9fa;'>"
        "<th style='text-align:left;padding:8px;'>Scenario</th>"
        "<th style='text-align:center;padding:8px;'>Win Rate</th>"
        "<th style='text-align:center;padding:8px;'>Take Rate</th>"
        "<th style='text-align:center;padding:8px;'>Margin %</th>"
        "<th style='text-align:center;padding:8px;'>3-Year Revenue</th>"
        "<th style='text-align:center;padding:8px;'>3-Year Margin</th>"
        "</tr>"
    )

    today_row = (
        f"<tr style='border-bottom:1px solid #ddd; background:#f3e8ff;'>"
        f"<td style='padding:8px;font-weight:bold;color:{TODAY_COLOR};'>Today (Actual)</td>"
        f"<td style='text-align:center;padding:8px;font-weight:bold;'>{today_wr:.0%}</td>"
        f"<td style='text-align:center;padding:8px;font-weight:bold;'>{today_tr:.2%}</td>"
        f"<td style='text-align:center;padding:8px;font-weight:bold;'>{today_mp:.1%}</td>"
        f"<td style='text-align:center;padding:8px;'>—</td>"
        f"<td style='text-align:center;padding:8px;'>—</td>"
        f"</tr>"
    )

    body_rows = ""
    for _, row in df.iterrows():
        color = _scenario_color(row["Scenario"])
        body_rows += (
            f"<tr style='border-bottom:1px solid #eee;'>"
            f"<td style='padding:8px;font-weight:bold;color:{color};'>{row['Scenario']}</td>"
            f"<td style='text-align:center;padding:8px;'>{row['Win Rate']:.0%} "
            f"{_delta_style(row['Win Rate'], today_wr)}</td>"
            f"<td style='text-align:center;padding:8px;'>{row['Take Rate']:.2%} "
            f"{_delta_style(row['Take Rate'], today_tr)}</td>"
            f"<td style='text-align:center;padding:8px;'>{row['Margin %']:.1%} "
            f"{_delta_style(row['Margin %'], today_mp)}</td>"
            f"<td style='text-align:center;padding:8px;'>${row['3-Year Revenue']:,.0f}</td>"
            f"<td style='text-align:center;padding:8px;'>${row['3-Year Margin $']:,.0f}</td>"
            f"</tr>"
        )

    html = header + today_row + body_rows + "</table>"
    st.markdown(html, unsafe_allow_html=True)


# ── Trade-off Scatter Plot ────────────────────────────────────────────

def render_tradeoff_scatter(
    scenarios: dict[str, dict],
    today: dict,
) -> None:
    """Scatter: Win Rate (x) vs Margin % (y), bubble size = 3yr revenue."""
    fig = go.Figure()

    for label, data in scenarios.items():
        yearly = data["yearly"]
        total_rev = sum(yr.total_revenue for yr in yearly.values())
        total_margin = sum(yr.margin for yr in yearly.values())
        margin_pct = total_margin / total_rev if total_rev > 0 else 0

        fig.add_trace(go.Scatter(
            x=[data["win_prob"] * 100],
            y=[margin_pct * 100],
            mode="markers+text",
            name=label,
            text=[label.replace(" Optimized", "").replace(" Today", "")],
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(
                size=max(15, min(50, total_rev / 5000)),
                color=_scenario_color(label),
                opacity=0.8,
                line=dict(width=1, color="#333"),
            ),
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"Win Rate: %{{x:.0f}}%<br>"
                f"Margin: %{{y:.1f}}%<br>"
                f"Revenue: ${total_rev:,.0f}<extra></extra>"
            ),
        ))

    fig.add_trace(go.Scatter(
        x=[today["win_rate"] * 100],
        y=[today["margin_pct"] * 100],
        mode="markers+text",
        name="Today (Actual)",
        text=["Today"],
        textposition="top center",
        textfont=dict(size=11, color=TODAY_COLOR),
        marker=dict(
            size=20,
            color=TODAY_COLOR,
            symbol="diamond",
            line=dict(width=2, color="#333"),
        ),
    ))

    fig.update_layout(
        title="Trade-off: Win Rate vs Margin %",
        xaxis_title="Win Rate (%)",
        yaxis_title="Margin (%)",
        template="plotly_white",
        height=450,
        showlegend=False,
        xaxis=dict(range=[0, 105]),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Revenue Composition ──────────────────────────────────────────────

def render_revenue_composition(
    scenarios: dict[str, dict],
) -> None:
    """Stacked bar: revenue breakdown by source per scenario."""
    labels = list(scenarios.keys())
    saas_vals, cc_vals, ach_vals, float_vals, impl_vals = [], [], [], [], []

    for data in scenarios.values():
        yearly = data["yearly"]
        saas_vals.append(sum(yr.saas_revenue for yr in yearly.values()))
        cc_vals.append(sum(yr.cc_revenue for yr in yearly.values()))
        ach_vals.append(sum(yr.ach_revenue + yr.bank_network_revenue for yr in yearly.values()))
        float_vals.append(sum(yr.float_income for yr in yearly.values()))
        impl_vals.append(sum(yr.impl_fee_revenue for yr in yearly.values()))

    fig = go.Figure()
    for vals, name, color in [
        (saas_vals, "SaaS", "#1B6AC9"),
        (cc_vals, "Credit Card", "#dc3545"),
        (ach_vals, "ACH / Bank", "#28a745"),
        (float_vals, "Float", "#fd7e14"),
        (impl_vals, "Impl Fee", "#6c757d"),
    ]:
        fig.add_trace(go.Bar(
            name=name, x=labels, y=vals,
            marker_color=color,
            hovertemplate=f"<b>{name}</b>: $%{{y:,.0f}}<extra></extra>",
        ))

    fig.update_layout(
        title="3-Year Revenue Composition by Scenario",
        barmode="stack",
        yaxis_title="Revenue ($)",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Pricing Decisions Comparison ─────────────────────────────────────

def render_pricing_comparison(
    results: dict[str, OptimizationResult],
    manual_pricing: PricingScenario | None = None,
) -> None:
    """Side-by-side table of pricing lever choices per scenario."""
    st.subheader("Pricing Decisions Comparison")

    scenarios = {}
    for key, result in results.items():
        scenarios[result.name] = result.pricing
    if manual_pricing:
        scenarios["Standard Pricing Today (Manual)"] = manual_pricing

    lever_labels = [
        ("SaaS Discount", lambda p: f"{p.saas_arr_discount_pct:.0%}"),
        ("Impl Discount", lambda p: f"{p.impl_fee_discount_pct:.0%}"),
        ("CC Base Rate", lambda p: f"{p.cc_base_rate:.2%}"),
        ("AMEX Rate", lambda p: f"{p.cc_amex_rate:.2%}"),
        ("ACH Mode", lambda p: p.ach_mode.replace("_", " ").title()),
        ("ACH Rate/Fee", lambda p: (
            f"${p.ach_fixed_fee:.2f}/txn" if p.ach_mode == "fixed_fee"
            else f"{p.ach_pct_rate:.2%}" + (f" (cap ${p.ach_cap:.2f})" if p.ach_mode == "capped" else "")
        )),
        ("Hold (CC/Bank/ACH)", lambda p: f"{p.hold_days_cc}/{p.hold_days_bank}/{p.hold_days_ach}"),
    ]

    header = (
        "<table style='width:100%; border-collapse:collapse; font-size:14px;'>"
        "<tr style='border-bottom:2px solid #ddd; background:#f8f9fa;'>"
        "<th style='text-align:left;padding:8px;'>Lever</th>"
    )
    for name in scenarios:
        color = _scenario_color(name)
        header += f"<th style='text-align:center;padding:8px;color:{color};'>{name}</th>"
    header += "</tr>"

    body = ""
    for lever_name, fmt_fn in lever_labels:
        body += f"<tr style='border-bottom:1px solid #eee;'><td style='padding:8px;font-weight:bold;'>{lever_name}</td>"
        for name, pricing in scenarios.items():
            body += f"<td style='text-align:center;padding:8px;'>{fmt_fn(pricing)}</td>"
        body += "</tr>"

    html = header + body + "</table>"
    st.markdown(html, unsafe_allow_html=True)


# ── Existing Charts (with Today reference lines) ─────────────────────

def render_comparison_chart(
    scenarios: dict[str, dict],
    today: dict | None = None,
) -> None:
    """Bar chart comparing Revenue, Margin, and Win Prob with Today reference."""
    labels = list(scenarios.keys())
    revenues, margins, win_probs = [], [], []
    margin_pcts, take_rates = [], []

    for label, data in scenarios.items():
        yearly = data["yearly"]
        total_rev = sum(yr.total_revenue for yr in yearly.values())
        total_margin = sum(yr.margin for yr in yearly.values())
        total_vol = sum(
            yr.total_revenue / yr.take_rate
            for yr in yearly.values() if yr.take_rate > 0
        )
        revenues.append(total_rev)
        margins.append(total_margin)
        win_probs.append(data["win_prob"])
        margin_pcts.append(total_margin / total_rev * 100 if total_rev > 0 else 0)
        take_rates.append(total_rev / total_vol * 100 if total_vol > 0 else 0)

    fig_rev = go.Figure()
    fig_rev.add_trace(go.Bar(
        name="3-Year Revenue",
        x=labels, y=revenues,
        marker_color=[_scenario_color(l) for l in labels],
        opacity=0.7,
    ))
    fig_rev.add_trace(go.Bar(
        name="3-Year Margin",
        x=labels, y=margins,
        marker_color=[_scenario_color(l) for l in labels],
        opacity=1.0,
    ))
    fig_rev.update_layout(
        title="Scenario Comparison: Revenue vs Margin",
        barmode="group", yaxis_title="Dollars ($)",
        template="plotly_white", height=400,
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        fig_wp = go.Figure()
        fig_wp.add_trace(go.Bar(
            x=labels, y=[wp * 100 for wp in win_probs],
            marker_color=[_scenario_color(l) for l in labels],
            text=[f"{wp:.0%}" for wp in win_probs],
            textposition="auto",
            textfont=dict(color="white", size=12),
        ))
        if today:
            fig_wp.add_hline(
                y=today["win_rate"] * 100,
                line_dash="dash", line_color=TODAY_COLOR, line_width=2,
                annotation_text=f"Today: {today['win_rate']:.0%}",
                annotation_position="top left",
                annotation_font_color=TODAY_COLOR,
            )
        fig_wp.update_layout(
            title="Win Rate", yaxis_title="%",
            yaxis=dict(rangemode="tozero"), template="plotly_white",
            height=350, margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_wp, use_container_width=True)

    with c2:
        fig_mp = go.Figure()
        fig_mp.add_trace(go.Bar(
            x=labels, y=margin_pcts,
            marker_color=[_scenario_color(l) for l in labels],
            text=[f"{m:.1f}%" for m in margin_pcts],
            textposition="auto",
            textfont=dict(color="white", size=12),
        ))
        if today:
            fig_mp.add_hline(
                y=today["margin_pct"] * 100,
                line_dash="dash", line_color=TODAY_COLOR, line_width=2,
                annotation_text=f"Today: {today['margin_pct']:.0%}",
                annotation_position="top left",
                annotation_font_color=TODAY_COLOR,
            )
        fig_mp.update_layout(
            title="Margin %", yaxis_title="%",
            yaxis=dict(rangemode="tozero"), template="plotly_white",
            height=350, margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_mp, use_container_width=True)

    with c3:
        fig_tr = go.Figure()
        fig_tr.add_trace(go.Bar(
            x=labels, y=take_rates,
            marker_color=[_scenario_color(l) for l in labels],
            text=[f"{t:.2f}%" for t in take_rates],
            textposition="auto",
            textfont=dict(color="white", size=12),
        ))
        if today:
            fig_tr.add_hline(
                y=today["take_rate"] * 100,
                line_dash="dash", line_color=TODAY_COLOR, line_width=2,
                annotation_text=f"Today: {today['take_rate']:.2%}",
                annotation_position="top left",
                annotation_font_color=TODAY_COLOR,
            )
        fig_tr.update_layout(
            title="Take Rate", yaxis_title="%",
            yaxis=dict(rangemode="tozero"), template="plotly_white",
            height=350, margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_tr, use_container_width=True)


def render_yearly_trend_chart(
    scenarios: dict[str, dict],
) -> None:
    """Line chart showing revenue and margin by year for each scenario."""
    fig_rev = go.Figure()
    fig_margin = go.Figure()

    years = [1, 2, 3]

    for label, data in scenarios.items():
        yearly = data["yearly"]
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
        template="plotly_white", height=400,
    )
    fig_margin.update_layout(
        title="Margin Trend by Year",
        yaxis_title="Margin ($)",
        template="plotly_white", height=400,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig_rev, use_container_width=True)
    with c2:
        st.plotly_chart(fig_margin, use_container_width=True)
