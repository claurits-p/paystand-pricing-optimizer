"""
Scenario comparison cards and 3-year forecast tables.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config as cfg
from models.revenue_model import PricingScenario, YearlyRevenue, _saas_arr_for_year
from optimizer.engine import OptimizationResult


def _pricing_summary(p: PricingScenario) -> dict:
    """Flatten pricing into display-friendly dict."""
    ach_desc = {
        "percentage": f"{p.ach_pct_rate:.2%}",
        "capped": f"{p.ach_pct_rate:.2%} (cap ${p.ach_cap:.2f})",
        "fixed_fee": f"${p.ach_fixed_fee:.2f} / txn",
    }
    saas_note = "(all years)" if p.saas_discount_persists else "(Y1 only)"
    cc_note = "(Y1 only)"
    return {
        "SaaS ARR Discount": f"{p.saas_arr_discount_pct:.0%} {saas_note}",
        "Effective Y1 ARR": f"${p.effective_saas_arr:,.0f}",
        "Renewal ARR (Y2)": f"${_saas_arr_for_year(p, 2):,.0f}",
        "Impl Fee Discount": f"{p.impl_fee_discount_pct:.0%}",
        "CC Base Rate": f"{p.cc_base_rate:.2%} {cc_note}",
        "AMEX Rate": f"{p.cc_amex_rate:.2%} {cc_note}",
        "ACH Mode": p.ach_mode.replace("_", " ").title(),
        "ACH Rate": ach_desc.get(p.ach_mode, ""),
        "Hold Days (CC/Bank/ACH)": f"{p.hold_days_cc} / {p.hold_days_bank} / {p.hold_days_ach}",
    }


def _yearly_to_df(yearly: dict[int, YearlyRevenue]) -> pd.DataFrame:
    rows = []
    for y in [1, 2, 3]:
        yr = yearly[y]
        margin_pct = yr.margin / yr.total_revenue if yr.total_revenue > 0 else 0
        rows.append({
            "Year": str(y),
            "SaaS Rev": yr.saas_revenue,
            "Impl Fee": yr.impl_fee_revenue,
            "CC Rev": yr.cc_revenue,
            "ACH Rev": yr.ach_revenue,
            "Bank Net Rev": yr.bank_network_revenue,
            "Float": yr.float_income,
            "Total Revenue": yr.total_revenue,
            "Total Cost": yr.total_cost,
            "Margin $": yr.margin,
            "Margin %": margin_pct,
            "Take Rate": yr.take_rate,
        })
    total_rev = sum(r["Total Revenue"] for r in rows)
    total_margin = sum(r["Margin $"] for r in rows)
    total_volume = sum(
        r["Total Revenue"] / r["Take Rate"]
        for r in rows if r["Take Rate"] > 0
    )
    totals = {
        "Year": "Total",
        "SaaS Rev": sum(r["SaaS Rev"] for r in rows),
        "Impl Fee": sum(r["Impl Fee"] for r in rows),
        "CC Rev": sum(r["CC Rev"] for r in rows),
        "ACH Rev": sum(r["ACH Rev"] for r in rows),
        "Bank Net Rev": sum(r["Bank Net Rev"] for r in rows),
        "Float": sum(r["Float"] for r in rows),
        "Total Revenue": total_rev,
        "Total Cost": sum(r["Total Cost"] for r in rows),
        "Margin $": total_margin,
        "Margin %": total_margin / total_rev if total_rev > 0 else 0,
        "Take Rate": total_rev / total_volume if total_volume > 0 else 0,
    }
    rows.append(totals)
    return pd.DataFrame(rows)


def _format_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format the dataframe for display."""
    formatted = df.copy()
    dollar_cols = [
        "SaaS Rev", "Impl Fee", "CC Rev", "ACH Rev",
        "Bank Net Rev", "Float", "Total Revenue", "Total Cost", "Margin $",
    ]
    for col in dollar_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].apply(
                lambda v: f"${v:,.0f}" if isinstance(v, (int, float)) else v
            )
    for pct_col in ["Margin %", "Take Rate"]:
        if pct_col in formatted.columns:
            formatted[pct_col] = formatted[pct_col].apply(
                lambda v: f"{v:.1%}" if isinstance(v, (int, float)) else v
            )
    return formatted


def _revenue_mix(yearly: dict[int, YearlyRevenue]) -> dict:
    """Calculate 3-year revenue mix by source."""
    total_saas = sum(yr.saas_revenue for yr in yearly.values())
    total_cc = sum(yr.cc_revenue for yr in yearly.values())
    total_ach = sum(yr.ach_revenue + yr.bank_network_revenue for yr in yearly.values())
    total_other = sum(yr.impl_fee_revenue + yr.float_income for yr in yearly.values())
    total = total_saas + total_cc + total_ach + total_other
    if total == 0:
        return {}
    return {
        "SaaS": total_saas / total,
        "CC": total_cc / total,
        "ACH/Bank": total_ach / total,
        "Other": total_other / total,
    }


def _render_metrics_row(
    yearly: dict[int, YearlyRevenue],
    pricing: PricingScenario,
    win_prob: float,
) -> None:
    """Render the top metrics row for any scenario."""
    total_rev = sum(yr.total_revenue for yr in yearly.values())
    total_margin = sum(yr.margin for yr in yearly.values())
    margin_pct = total_margin / total_rev if total_rev > 0 else 0
    renewal_arr = _saas_arr_for_year(pricing, 2)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Win Probability", f"{win_prob:.0%}")
    m2.metric("3-Year Revenue", f"${total_rev:,.0f}")
    m3.metric("3-Year Margin", f"${total_margin:,.0f}")
    m4.metric("Margin %", f"{margin_pct:.1%}")
    m5.metric("Renewal ARR (Y2)", f"${renewal_arr:,.0f}")

    mix = _revenue_mix(yearly)
    if mix:
        mix_str = " · ".join(f"{k}: {v:.0%}" for k, v in mix.items() if v > 0.005)
        st.caption(f"Revenue Mix: {mix_str}")


def render_scenario_card(result: OptimizationResult) -> None:
    """Render a single scenario card."""
    st.subheader(result.name)

    _render_metrics_row(result.yearly, result.pricing, result.win_prob)

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

    _render_metrics_row(yearly, pricing, win_prob)

    with st.expander("Pricing Details"):
        summary = _pricing_summary(pricing)
        cols = st.columns(3)
        items = list(summary.items())
        for i, (k, v) in enumerate(items):
            cols[i % 3].markdown(f"**{k}:** {v}")

    df = _yearly_to_df(yearly)
    st.dataframe(_format_df(df), use_container_width=True, hide_index=True)
