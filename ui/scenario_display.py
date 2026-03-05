"""
Scenario comparison cards and 3-year forecast tables.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from models.revenue_model import PricingScenario, YearlyRevenue
from optimizer.engine import OptimizationResult


def _pricing_summary(p: PricingScenario) -> dict:
    """Flatten pricing into display-friendly dict."""
    ach_desc = {
        "percentage": f"{p.ach_pct_rate:.2%}",
        "capped": f"{p.ach_pct_rate:.2%} (cap ${p.ach_cap:.2f})",
        "fixed_fee": f"${p.ach_fixed_fee:.2f} / txn",
    }
    return {
        "SaaS ARR Discount": f"{p.saas_arr_discount_pct:.0%}",
        "Effective ARR": f"${p.effective_saas_arr:,.0f}",
        "Impl Fee Discount": f"{p.impl_fee_discount_pct:.0%}",
        "Effective Impl Fee": f"${p.effective_impl_fee:,.0f}",
        "CC Base Rate": f"{p.cc_base_rate:.2%}",
        "AMEX Rate": f"{p.cc_amex_rate:.2%}",
        "ACH Mode": p.ach_mode.replace("_", " ").title(),
        "ACH Rate": ach_desc.get(p.ach_mode, ""),
        "Hold Days (CC/ACH/Bank)": f"{p.hold_days_cc} / {p.hold_days_ach} / {p.hold_days_bank}",
    }


def _yearly_to_df(yearly: dict[int, YearlyRevenue]) -> pd.DataFrame:
    rows = []
    for y in [1, 2, 3]:
        yr = yearly[y]
        rows.append({
            "Year": y,
            "SaaS Rev": yr.saas_revenue,
            "Impl Fee": yr.impl_fee_revenue,
            "CC Rev": yr.cc_revenue,
            "ACH Rev": yr.ach_revenue,
            "Bank Net Rev": yr.bank_network_revenue,
            "Float": yr.float_income,
            "Total Revenue": yr.total_revenue,
            "Total Cost": yr.total_cost,
            "Margin": yr.margin,
            "Take Rate": yr.take_rate,
        })
    totals = {
        "Year": "Total",
        "SaaS Rev": sum(r["SaaS Rev"] for r in rows),
        "Impl Fee": sum(r["Impl Fee"] for r in rows),
        "CC Rev": sum(r["CC Rev"] for r in rows),
        "ACH Rev": sum(r["ACH Rev"] for r in rows),
        "Bank Net Rev": sum(r["Bank Net Rev"] for r in rows),
        "Float": sum(r["Float"] for r in rows),
        "Total Revenue": sum(r["Total Revenue"] for r in rows),
        "Total Cost": sum(r["Total Cost"] for r in rows),
        "Margin": sum(r["Margin"] for r in rows),
        "Take Rate": sum(r["Total Revenue"] for r in rows)
        / max(sum(r["Total Revenue"] for r in rows), 1),
    }
    rows.append(totals)
    return pd.DataFrame(rows)


def _format_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format the dataframe for display."""
    formatted = df.copy()
    dollar_cols = [
        "SaaS Rev", "Impl Fee", "CC Rev", "ACH Rev",
        "Bank Net Rev", "Float", "Total Revenue", "Total Cost", "Margin",
    ]
    for col in dollar_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(
                lambda v: f"${v:,.0f}" if isinstance(v, (int, float)) else v
            )
    if "Take Rate" in formatted.columns:
        formatted["Take Rate"] = formatted["Take Rate"].apply(
            lambda v: f"{v:.2%}" if isinstance(v, (int, float)) else v
        )
    return formatted


def render_scenario_card(result: OptimizationResult) -> None:
    """Render a single scenario card."""
    st.subheader(result.name)

    m1, m2, m3 = st.columns(3)
    total_rev = sum(yr.total_revenue for yr in result.yearly.values())
    total_margin = sum(yr.margin for yr in result.yearly.values())
    m1.metric("Win Probability", f"{result.win_prob:.0%}")
    m2.metric("3-Year Revenue", f"${total_rev:,.0f}")
    m3.metric("3-Year Margin", f"${total_margin:,.0f}")

    with st.expander("Pricing Details"):
        summary = _pricing_summary(result.pricing)
        cols = st.columns(3)
        items = list(summary.items())
        for i, (k, v) in enumerate(items):
            cols[i % 3].markdown(f"**{k}:** {v}")

    df = _yearly_to_df(result.yearly)
    st.dataframe(_format_df(df), use_container_width=True, hide_index=True)

    if result.explanation:
        st.info(result.explanation)


def render_manual_scenario_card(
    yearly: dict[int, YearlyRevenue],
    pricing: PricingScenario,
    win_prob: float,
) -> None:
    """Render the manual / current pricing scenario."""
    st.subheader("Current Pricing (Manual)")

    m1, m2, m3 = st.columns(3)
    total_rev = sum(yr.total_revenue for yr in yearly.values())
    total_margin = sum(yr.margin for yr in yearly.values())
    m1.metric("Win Probability", f"{win_prob:.0%}")
    m2.metric("3-Year Revenue", f"${total_rev:,.0f}")
    m3.metric("3-Year Margin", f"${total_margin:,.0f}")

    with st.expander("Pricing Details"):
        summary = _pricing_summary(pricing)
        cols = st.columns(3)
        items = list(summary.items())
        for i, (k, v) in enumerate(items):
            cols[i % 3].markdown(f"**{k}:** {v}")

    df = _yearly_to_df(yearly)
    st.dataframe(_format_df(df), use_container_width=True, hide_index=True)
