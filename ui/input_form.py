"""
Streamlit input form for deal parameters.
"""
from __future__ import annotations
import streamlit as st

import config as cfg


def render_deal_inputs() -> dict:
    """Render the deal input form and return collected values."""

    st.header("Deal Inputs")

    col1, col2 = st.columns(2)

    with col1:
        company_name = st.text_input("Company Name", placeholder="Input Company Name Here")
        processing_tier_volume = st.number_input(
            "Processing Tier Total Volume ($/yr)",
            min_value=0.0,
            value=10_000_000.0,
            step=100_000.0,
            format="%.0f",
        )
        expected_cc_volume = st.number_input(
            "Expected CC Volume ($/yr)",
            min_value=0.0,
            value=4_000_000.0,
            step=100_000.0,
            format="%.0f",
        )

    with col2:
        conv_fee_today = st.selectbox(
            "Convenience Fees Today?",
            options=[("No", 0), ("Yes", 1)],
            format_func=lambda x: x[0],
            index=0,
        )[1]
        conv_fee_with_paystand = st.selectbox(
            "Convenience Fees with Paystand?",
            options=[("No", 0), ("Yes", 1)],
            format_func=lambda x: x[0],
            index=1,
        )[1]

    st.subheader("SaaS Pricing")
    s1, s2 = st.columns(2)
    with s1:
        saas_arr_list = st.number_input(
            "SaaS ARR List Price ($/yr)",
            min_value=0.0,
            value=float(cfg.SAAS_ARR_DEFAULT),
            step=1000.0,
            format="%.0f",
        )
    with s2:
        impl_fee_list = st.number_input(
            "Implementation Fee List Price ($)",
            min_value=0.0,
            value=float(cfg.SAAS_IMPL_FEE_DEFAULT),
            step=500.0,
            format="%.0f",
        )

    st.subheader("Current Performance (Today)")
    t1, t2, t3 = st.columns(3)
    with t1:
        today_win_rate = st.number_input(
            "Pricing → Win Rate (%)", min_value=0.0, max_value=100.0,
            value=52.0, step=1.0, format="%.1f",
        ) / 100
    with t2:
        today_take_rate = st.number_input(
            "Take Rate Today (%)", min_value=0.0, max_value=10.0,
            value=1.62, step=0.1, format="%.2f",
        ) / 100
    with t3:
        today_margin_pct = st.number_input(
            "Margin % Today", min_value=0.0, max_value=100.0,
            value=33.0, step=1.0, format="%.1f",
        ) / 100

    return {
        "company_name": company_name,
        "processing_tier_volume": processing_tier_volume,
        "expected_cc_volume": expected_cc_volume,
        "conv_fee_today": conv_fee_today,
        "conv_fee_with_paystand": conv_fee_with_paystand,
        "saas_arr_list": saas_arr_list,
        "impl_fee_list": impl_fee_list,
        "today_win_rate": today_win_rate,
        "today_take_rate": today_take_rate,
        "today_margin_pct": today_margin_pct,
    }


def render_manual_scenario() -> dict | None:
    """
    Render inputs for the manual 'Current Pricing' scenario.
    Returns pricing dict or None if user skips.
    """
    with st.expander("Current Pricing (Manual Baseline)", expanded=False):
        include = st.checkbox("Include current pricing scenario", value=True)
        if not include:
            return None

        c1, c2, c3 = st.columns(3)
        with c1:
            saas_disc = st.slider(
                "SaaS ARR Discount %", 0, 70, 20, key="manual_saas_disc"
            ) / 100
            impl_disc = st.slider(
                "Impl Fee Discount %", 0, 100, 0, key="manual_impl_disc"
            ) / 100
        with c2:
            cc_rate = st.number_input(
                "CC Base Rate %", min_value=1.99, max_value=2.39,
                value=1.99, step=0.05, key="manual_cc",
            ) / 100
            amex_rate = st.number_input(
                "AMEX Rate %", min_value=2.50, max_value=4.0,
                value=3.25, step=0.05, key="manual_amex",
            ) / 100
        with c3:
            ach_mode = st.selectbox(
                "ACH Mode", cfg.ACH_MODES,
                format_func=lambda m: {
                    "percentage": "Percentage (e.g. 0.49%)",
                    "capped": "Capped (% with $ cap per txn)",
                    "fixed_fee": "Fixed Fee ($ per txn)",
                }.get(m, m),
                index=2, key="manual_ach_mode",
            )
            ach_pct = 0.49
            ach_cap = 10.0
            ach_fixed = 2.00
            if ach_mode in ("percentage", "capped"):
                ach_pct = st.number_input(
                    "ACH % Rate", min_value=0.10, max_value=1.0,
                    value=0.49, step=0.05, key="manual_ach_pct",
                )
            if ach_mode == "capped":
                ach_cap = st.number_input(
                    "ACH Cap ($)", min_value=1.0, max_value=25.0,
                    value=10.0, step=0.50, key="manual_ach_cap",
                )
            if ach_mode == "fixed_fee":
                ach_fixed = st.number_input(
                    "ACH Fixed Fee ($)", min_value=0.50, max_value=10.0,
                    value=2.00, step=0.25, key="manual_ach_fixed",
                )

        st.markdown("**Hold Days by Payment Type**")
        h1, h2, h3 = st.columns(3)
        with h1:
            hold_cc = st.slider("CC Hold Days", 1, 2, 2, key="manual_hold_cc")
        with h2:
            hold_bank = st.slider("Bank Hold Days", 1, 5, 2, key="manual_hold_bank")
        with h3:
            hold_ach = st.slider("ACH Hold Days", 1, 7, 3, key="manual_hold_ach")

        return {
            "saas_arr_discount_pct": saas_disc,
            "impl_fee_discount_pct": impl_disc,
            "cc_base_rate": cc_rate,
            "cc_amex_rate": amex_rate,
            "ach_mode": ach_mode,
            "ach_pct_rate": ach_pct / 100,
            "ach_cap": ach_cap,
            "ach_fixed_fee": ach_fixed,
            "hold_days_cc": hold_cc,
            "hold_days_ach": hold_ach,
            "hold_days_bank": hold_bank,
        }


def render_model_config() -> dict:
    """Render win probability / model tuning in the sidebar."""

    st.sidebar.header("Model Configuration")

    st.sidebar.subheader("Win Probability Curve")
    floor = st.sidebar.slider(
        "Floor (min win %)", 0, 50, int(cfg.WIN_PROB_DEFAULTS["floor"] * 100),
    ) / 100
    ceiling = st.sidebar.slider(
        "Ceiling (max win %)", 50, 100, int(cfg.WIN_PROB_DEFAULTS["ceiling"] * 100),
    ) / 100
    steepness = st.sidebar.slider(
        "Steepness", 1.0, 20.0, cfg.WIN_PROB_DEFAULTS["steepness"], step=0.5,
    )

    st.sidebar.subheader("Lever Weights")
    w = cfg.WIN_PROB_DEFAULTS["weights"]
    w_cc = st.sidebar.slider("CC Rate Weight", 0.0, 1.0, w["cc_rate"], 0.05)
    w_ach = st.sidebar.slider("ACH Pricing Weight", 0.0, 1.0, w["ach_rate"], 0.05)
    w_saas = st.sidebar.slider("SaaS Discount Weight", 0.0, 1.0, w["saas_discount"], 0.05)
    w_impl = st.sidebar.slider("Impl Discount Weight", 0.0, 1.0, w["impl_discount"], 0.05)
    w_hold = st.sidebar.slider("Hold Time Weight", 0.0, 1.0, w["hold_time"], 0.05)

    st.sidebar.subheader("Benchmarks")
    bm = cfg.MARKET_BENCHMARKS
    bm_cc = st.sidebar.number_input(
        "Benchmark Blended CC Rate %", value=bm["cc_rate"] * 100, step=0.10,
    ) / 100
    bm_ach = st.sidebar.number_input(
        "Benchmark ACH Eff. Rate %", value=bm["ach_effective_rate"] * 100, step=0.05,
    ) / 100

    st.sidebar.subheader("Benchmark Hold Days")
    bm_hold_cc = st.sidebar.number_input(
        "Benchmark CC Hold Days", value=bm["hold_days_cc"], step=1, min_value=1,
    )
    bm_hold_bank = st.sidebar.number_input(
        "Benchmark Bank Hold Days", value=bm["hold_days_bank"], step=1, min_value=1,
    )
    bm_hold_ach = st.sidebar.number_input(
        "Benchmark ACH Hold Days", value=bm["hold_days_ach"], step=1, min_value=1,
    )

    return {
        "floor": floor,
        "ceiling": ceiling,
        "steepness": steepness,
        "weights": {
            "cc_rate": w_cc,
            "ach_rate": w_ach,
            "saas_discount": w_saas,
            "impl_discount": w_impl,
            "hold_time": w_hold,
        },
        "benchmarks": {
            **bm,
            "cc_rate": bm_cc,
            "ach_effective_rate": bm_ach,
            "hold_days_cc": bm_hold_cc,
            "hold_days_ach": bm_hold_ach,
            "hold_days_bank": bm_hold_bank,
        },
    }
