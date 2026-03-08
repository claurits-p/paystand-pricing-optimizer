"""
Scenario comparison cards and 3-year forecast tables.
"""
from __future__ import annotations
import copy

import pandas as pd
import streamlit as st

import config as cfg
from models.revenue_model import (
    PricingScenario, YearlyRevenue,
    _saas_arr_for_year, compute_three_year_financials,
)
from models.win_probability import (
    solve_saas_for_target_win_rate,
    solve_multi_lever_for_target_win_rate,
    win_probability,
    win_probability_uncapped,
)
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


def _format_lever_changes(changes: dict) -> str:
    """Human-readable summary of which levers were adjusted."""
    parts = []
    labels = {
        "saas_arr_discount_pct": ("SaaS Discount", lambda o, n: f"{o:.0%} → {n:.0%}"),
        "cc_base_rate": ("CC Base Rate", lambda o, n: f"{o:.2%} → {n:.2%}"),
        "ach_pct_rate": ("ACH Rate", lambda o, n: f"{o:.2%} → {n:.2%}"),
        "ach_fixed_fee": ("ACH Fixed Fee", lambda o, n: f"${o:.2f} → ${n:.2f}"),
    }
    for key, (old, new) in changes.items():
        label, fmt = labels.get(key, (key, lambda o, n: f"{o} → {n}"))
        parts.append(f"**{label}:** {fmt(old, new)}")
    return " · ".join(parts)


def render_boost_analysis(
    pricing: PricingScenario,
    yearly: dict[int, YearlyRevenue],
    win_prob: float,
    boost_pct: float,
    volumes: dict,
    wp_params: dict,
) -> None:
    """Show the win rate boost analysis for a scenario."""
    if boost_pct <= 0:
        return

    target_wp = min(win_prob + boost_pct, wp_params.get("ceiling", 0.80))
    actual_boost = target_wp - win_prob

    if actual_boost <= 0.005:
        st.caption("Already at or near ceiling — no boost needed.")
        return

    result = solve_multi_lever_for_target_win_rate(pricing, target_wp, wp_params)

    if result is None:
        maxed = copy.copy(pricing)
        changes = {}
        lb = cfg.LEVER_BOUNDS
        if maxed.saas_arr_discount_pct < lb["saas_arr_discount_pct"]["max"]:
            changes["saas_arr_discount_pct"] = (pricing.saas_arr_discount_pct, lb["saas_arr_discount_pct"]["max"])
            maxed.saas_arr_discount_pct = lb["saas_arr_discount_pct"]["max"]
        if maxed.cc_base_rate > lb["cc_base_rate"]["min"]:
            changes["cc_base_rate"] = (pricing.cc_base_rate, lb["cc_base_rate"]["min"])
            maxed.cc_base_rate = lb["cc_base_rate"]["min"]
        if maxed.cc_amex_rate > lb["cc_amex_rate"]["min"]:
            changes["cc_amex_rate"] = (pricing.cc_amex_rate, lb["cc_amex_rate"]["min"])
            maxed.cc_amex_rate = lb["cc_amex_rate"]["min"]
        if maxed.ach_mode == "percentage" and maxed.ach_pct_rate > lb["ach_pct_rate"]["min"]:
            changes["ach_pct_rate"] = (pricing.ach_pct_rate, lb["ach_pct_rate"]["min"])
            maxed.ach_pct_rate = lb["ach_pct_rate"]["min"]
        elif maxed.ach_mode == "capped":
            if maxed.ach_pct_rate > lb["ach_pct_rate"]["min"]:
                changes["ach_pct_rate"] = (pricing.ach_pct_rate, lb["ach_pct_rate"]["min"])
                maxed.ach_pct_rate = lb["ach_pct_rate"]["min"]
            if maxed.ach_cap > lb["ach_cap"]["min"]:
                changes["ach_cap"] = (pricing.ach_cap, lb["ach_cap"]["min"])
                maxed.ach_cap = lb["ach_cap"]["min"]
        elif maxed.ach_mode == "fixed_fee" and maxed.ach_fixed_fee > lb["ach_fixed_fee"]["min"]:
            changes["ach_fixed_fee"] = (pricing.ach_fixed_fee, lb["ach_fixed_fee"]["min"])
            maxed.ach_fixed_fee = lb["ach_fixed_fee"]["min"]

        max_wp = win_probability_uncapped(maxed, **wp_params)
        max_boost = max_wp - win_prob
        if max_boost < 0.005 or not changes:
            st.caption("All levers already at maximum — no further boost possible.")
            return

        actual_boost = max_boost
        st.caption(f"Max achievable boost: **+{max_boost:.1%}** (target +{boost_pct:.0%} not fully reachable)")
        result = {"pricing": maxed, "changes": changes}

    boosted = result["pricing"]
    changes = result["changes"]
    boosted_yearly = compute_three_year_financials(volumes, boosted)
    boosted_wp = win_prob + actual_boost

    orig_rev = sum(yr.total_revenue for yr in yearly.values())
    orig_margin = sum(yr.margin for yr in yearly.values())
    orig_margin_pct = orig_margin / orig_rev if orig_rev > 0 else 0
    orig_vol = sum(yr.total_revenue / yr.take_rate for yr in yearly.values() if yr.take_rate > 0)
    orig_take_rate = orig_rev / orig_vol if orig_vol > 0 else 0

    boost_rev = sum(yr.total_revenue for yr in boosted_yearly.values())
    boost_margin = sum(yr.margin for yr in boosted_yearly.values())
    boost_margin_pct = boost_margin / boost_rev if boost_rev > 0 else 0
    boost_vol = sum(yr.total_revenue / yr.take_rate for yr in boosted_yearly.values() if yr.take_rate > 0)
    boost_take_rate = boost_rev / boost_vol if boost_vol > 0 else 0

    with st.expander(f"Win Rate Boost Analysis (+{actual_boost:.0%})", expanded=False):
        st.markdown(f"Levers adjusted: {_format_lever_changes(changes)}")

        saas_disc = boosted.saas_arr_discount_pct
        b1, b2, b3, b4 = st.columns(4)
        b1.metric(
            "SaaS Discount",
            f"{saas_disc:.0%}",
            delta=f"{saas_disc - pricing.saas_arr_discount_pct:+.0%} from {pricing.saas_arr_discount_pct:.0%}",
        )
        b2.metric(
            "Win Rate",
            f"{boosted_wp:.0%}",
            delta=f"+{actual_boost:.0%}",
        )
        b3.metric(
            "Margin %",
            f"{boost_margin_pct:.1%}",
            delta=f"{(boost_margin_pct - orig_margin_pct) * 100:+.1f}pp",
        )
        b4.metric(
            "Take Rate",
            f"{boost_take_rate:.2%}",
            delta=f"{(boost_take_rate - orig_take_rate) * 100:+.2f}pp",
        )

        st.markdown("**Year-by-Year Impact** (Original → Boosted)")
        rows = []
        for y in [1, 2, 3]:
            orig_y = yearly[y]
            boost_y = boosted_yearly[y]
            orig_mp = orig_y.margin / orig_y.total_revenue if orig_y.total_revenue > 0 else 0
            boost_mp = boost_y.margin / boost_y.total_revenue if boost_y.total_revenue > 0 else 0
            rows.append({
                "Year": str(y),
                "SaaS (Orig)": f"${orig_y.saas_revenue:,.0f}",
                "SaaS (Boosted)": f"${boost_y.saas_revenue:,.0f}",
                "SaaS Delta": f"${boost_y.saas_revenue - orig_y.saas_revenue:+,.0f}",
                "Revenue (Orig)": f"${orig_y.total_revenue:,.0f}",
                "Revenue (Boosted)": f"${boost_y.total_revenue:,.0f}",
                "Rev Delta": f"${boost_y.total_revenue - orig_y.total_revenue:+,.0f}",
                "Margin % (Orig)": f"{orig_mp:.1%}",
                "Margin % (Boosted)": f"{boost_mp:.1%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        y1_delta = boosted_yearly[1].total_revenue - yearly[1].total_revenue
        y2_delta = boosted_yearly[2].total_revenue - yearly[2].total_revenue
        y3_delta = boosted_yearly[3].total_revenue - yearly[3].total_revenue
        total_delta = boost_rev - orig_rev

        if y1_delta < 0 and (y2_delta + y3_delta) > 0:
            st.info(
                f"Year 1 trade-off: **\\${y1_delta:+,.0f}** in revenue (discount hit). "
                f"Years 2-3 recovery: **\\${y2_delta + y3_delta:+,.0f}** as discounts revert. "
                f"Net 3-year impact: **\\${total_delta:+,.0f}**."
            )
        else:
            st.info(f"Net 3-year revenue impact: **\\${total_delta:+,.0f}**.")

        orig_ev = win_prob * orig_rev
        boost_ev = boosted_wp * boost_rev
        st.markdown(
            f"**Expected Value:** "
            f"Original: {win_prob:.0%} x \\${orig_rev:,.0f} = **\\${orig_ev:,.0f}** · "
            f"Boosted: {boosted_wp:.0%} x \\${boost_rev:,.0f} = **\\${boost_ev:,.0f}** · "
            f"EV Gain: **\\${boost_ev - orig_ev:+,.0f}**"
        )


def render_manual_scenario_card(
    yearly: dict[int, YearlyRevenue],
    pricing: PricingScenario,
    win_prob: float,
) -> None:
    """Render the manual / current pricing scenario."""
    st.subheader("Standard Pricing Today (Manual)")

    _render_metrics_row(yearly, pricing, win_prob)

    with st.expander("Pricing Details"):
        summary = _pricing_summary(pricing)
        cols = st.columns(3)
        items = list(summary.items())
        for i, (k, v) in enumerate(items):
            cols[i % 3].markdown(f"**{k}:** {v}")

    df = _yearly_to_df(yearly)
    st.dataframe(_format_df(df), use_container_width=True, hide_index=True)
