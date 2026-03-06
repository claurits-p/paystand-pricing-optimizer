"""
Paystand Pricing Optimizer — Streamlit App

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

st.set_page_config(
    page_title="Pricing Optimizer",
    page_icon="💰",
    layout="wide",
)

def _check_password():
    """Simple password gate using Streamlit secrets."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    pwd = st.text_input("Enter password to access the app", type="password")
    if pwd:
        if pwd == st.secrets.get("password", ""):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False


if not _check_password():
    st.stop()

from models.volume_forecast import forecast_volume_y1_y3
from models.revenue_model import (
    PricingScenario,
    compute_three_year_financials,
)
from models.win_probability import win_probability
from optimizer.engine import run_all_optimizations
from ui.input_form import render_deal_inputs, render_manual_scenario, render_model_config
from ui.scenario_display import render_scenario_card, render_manual_scenario_card
from ui.charts import (
    render_executive_table,
    render_tradeoff_scatter,
    render_revenue_composition,
    render_pricing_comparison,
    render_comparison_chart,
    render_yearly_trend_chart,
)


def main():
    logo_col, title_col = st.columns([0.06, 0.94], gap="small")
    with logo_col:
        st.image("paystand_logo.png", width=55)
    with title_col:
        st.markdown(
            '<h1 style="color: #1B2A4A; margin-top: -5px;">Paystand Payment Processing Pricing Optimizer</h1>',
            unsafe_allow_html=True,
        )
    st.markdown(
        "Input deal parameters, then run optimization to see "
        "**Margin % Optimized**, **Take Rate Optimized**, and **LTV Optimized** "
        "pricing strategies with 3-year forecasts."
    )

    model_cfg = render_model_config()

    deal = render_deal_inputs()

    today = {
        "win_rate": deal["today_win_rate"],
        "take_rate": deal["today_take_rate"],
        "margin_pct": deal["today_margin_pct"],
    }

    st.divider()
    manual_pricing_dict = render_manual_scenario()

    st.divider()

    if st.button("Run Optimization", type="primary", use_container_width=True):
        volumes = forecast_volume_y1_y3(
            processing_tier_volume=deal["processing_tier_volume"],
            expected_cc_volume=deal["expected_cc_volume"],
            conv_fee_with_paystand=deal["conv_fee_with_paystand"],
            conv_fee_today=deal["conv_fee_today"],
        )

        st.subheader("Volume Forecast")
        vol_cols = st.columns(3)
        for i, year in enumerate([1, 2, 3]):
            v = volumes[year]
            with vol_cols[i]:
                st.metric(f"Year {year} Total", f"${v.total:,.0f}")
                st.markdown(
                    f"CC: \\${v.cc:,.0f} · ACH: \\${v.ach:,.0f} · "
                    f"Bank: \\${v.bank_network:,.0f}"
                )

        st.divider()

        wp_params = {
            "floor": model_cfg["floor"],
            "ceiling": model_cfg["ceiling"],
            "steepness": model_cfg["steepness"],
            "weights": model_cfg["weights"],
            "benchmarks": model_cfg["benchmarks"],
        }

        manual_pricing = None
        manual_yearly = None
        manual_wp = None

        if manual_pricing_dict:
            manual_pricing = PricingScenario(
                saas_arr_list=deal["saas_arr_list"],
                impl_fee_list=deal["impl_fee_list"],
                **manual_pricing_dict,
            )
            manual_yearly = compute_three_year_financials(volumes, manual_pricing)
            manual_wp = win_probability(manual_pricing, **wp_params)

        with st.spinner("Optimizing pricing scenarios..."):
            results = run_all_optimizations(
                volumes=volumes,
                saas_arr_list=deal["saas_arr_list"],
                impl_fee_list=deal["impl_fee_list"],
                wp_params=wp_params,
            )

        all_scenarios = {}

        if manual_pricing:
            all_scenarios["Standard Pricing Today (Manual)"] = {
                "yearly": manual_yearly,
                "win_prob": manual_wp,
            }

        scenario_order = ["msrp", "ltv", "saas_passive", "margin_pct", "take_rate"]
        for key in scenario_order:
            result = results[key]
            all_scenarios[result.name] = {
                "yearly": result.yearly,
                "win_prob": result.win_prob,
            }

        st.header(f"Results — {deal['company_name']}")

        render_executive_table(all_scenarios, today)

        st.divider()

        render_tradeoff_scatter(all_scenarios, today)

        st.divider()

        render_comparison_chart(all_scenarios, today)
        render_yearly_trend_chart(all_scenarios)

        st.divider()

        render_revenue_composition(all_scenarios)

        st.divider()

        render_pricing_comparison(results, manual_pricing)

        st.divider()

        render_scenario_card(results["msrp"])
        st.divider()

        render_scenario_card(results["ltv"])
        st.divider()

        render_scenario_card(results["saas_passive"])
        st.divider()

        if manual_pricing:
            render_manual_scenario_card(manual_yearly, manual_pricing, manual_wp)
            st.divider()

        for key in ["margin_pct", "take_rate"]:
            render_scenario_card(results[key])
            st.divider()

        st.subheader("Export")
        import pandas as pd
        export_rows = []
        for label, data in all_scenarios.items():
            for year in [1, 2, 3]:
                yr = data["yearly"][year]
                margin_pct = yr.margin / yr.total_revenue if yr.total_revenue > 0 else 0
                export_rows.append({
                    "Scenario": label,
                    "Year": year,
                    "Win Prob": data["win_prob"],
                    "Total Revenue": yr.total_revenue,
                    "Total Cost": yr.total_cost,
                    "Margin $": yr.margin,
                    "Margin %": margin_pct,
                    "Take Rate": yr.take_rate,
                    "SaaS Rev": yr.saas_revenue,
                    "CC Rev": yr.cc_revenue,
                    "ACH Rev": yr.ach_revenue,
                })
        export_df = pd.DataFrame(export_rows)
        csv = export_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            data=csv,
            file_name=f"{deal['company_name']}_pricing_scenarios.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
