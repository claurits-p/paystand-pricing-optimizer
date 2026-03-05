"""
Optimization engine.

For each ACH pricing mode (percentage, capped, fixed_fee), runs a
continuous optimization over the remaining levers and picks the best.

Three objectives:
  1. Margin-optimized:  maximize  sum(margin_y1..y3) * P(win)
  2. Revenue-optimized: maximize  sum(revenue_y1..y3) * P(win)
  3. Best strategy:     maximize  sum(margin_y1..y3 * retention_y) * P(win)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution

import config as cfg
from models.volume_forecast import VolumeForecastYear
from models.revenue_model import (
    PricingScenario,
    YearlyRevenue,
    compute_three_year_financials,
)
from models.win_probability import win_probability


@dataclass
class OptimizationResult:
    name: str
    pricing: PricingScenario
    win_prob: float
    yearly: dict[int, YearlyRevenue]
    objective_value: float
    explanation: str = ""


RETENTION_CURVE = {1: 1.0, 2: 0.90, 3: 0.82}


def _build_pricing_from_vector(
    x: np.ndarray,
    ach_mode: str,
    saas_arr_list: float,
    impl_fee_list: float,
) -> PricingScenario:
    """
    Map continuous optimization vector → PricingScenario.

    x layout:
      [0] saas_arr_discount_pct
      [1] impl_fee_discount_pct
      [2] cc_base_rate
      [3] cc_amex_rate
      [4] ach_pct_rate       (used in percentage / capped modes)
      [5] ach_cap             (used in capped mode)
      [6] ach_fixed_fee       (used in fixed_fee mode)
      [7] hold_days_cc
      [8] hold_days_ach
      [9] hold_days_bank
    """
    return PricingScenario(
        saas_arr_discount_pct=float(x[0]),
        impl_fee_discount_pct=float(x[1]),
        cc_base_rate=float(x[2]),
        cc_amex_rate=float(x[3]),
        ach_mode=ach_mode,
        ach_pct_rate=float(x[4]),
        ach_cap=float(x[5]),
        ach_fixed_fee=float(x[6]),
        hold_days_cc=int(round(x[7])),
        hold_days_ach=int(round(x[8])),
        hold_days_bank=int(round(x[9])),
        saas_arr_list=saas_arr_list,
        impl_fee_list=impl_fee_list,
    )


def _get_bounds() -> list[tuple[float, float]]:
    """Bounds for the 10-element optimization vector."""
    lb = cfg.LEVER_BOUNDS
    return [
        (lb["saas_arr_discount_pct"]["min"], lb["saas_arr_discount_pct"]["max"]),
        (lb["impl_fee_discount_pct"]["min"], lb["impl_fee_discount_pct"]["max"]),
        (lb["cc_base_rate"]["min"], lb["cc_base_rate"]["max"]),
        (lb["cc_amex_rate"]["min"], lb["cc_amex_rate"]["max"]),
        (lb["ach_pct_rate"]["min"], lb["ach_pct_rate"]["max"]),
        (lb["ach_cap"]["min"], lb["ach_cap"]["max"]),
        (lb["ach_fixed_fee"]["min"], lb["ach_fixed_fee"]["max"]),
        (lb["hold_days_cc"]["min"], lb["hold_days_cc"]["max"]),
        (lb["hold_days_ach"]["min"], lb["hold_days_ach"]["max"]),
        (lb["hold_days_bank"]["min"], lb["hold_days_bank"]["max"]),
    ]


def _objective_margin(
    x: np.ndarray,
    ach_mode: str,
    volumes: dict[int, VolumeForecastYear],
    saas_arr_list: float,
    impl_fee_list: float,
    wp_params: dict,
) -> float:
    """Negative expected 3-year margin: margin * P(win)."""
    pricing = _build_pricing_from_vector(x, ach_mode, saas_arr_list, impl_fee_list)
    yearly = compute_three_year_financials(volumes, pricing)
    wp = win_probability(pricing, **wp_params)
    total_margin = sum(yr.margin for yr in yearly.values())
    return -(total_margin * wp)


def _objective_revenue(
    x: np.ndarray,
    ach_mode: str,
    volumes: dict[int, VolumeForecastYear],
    saas_arr_list: float,
    impl_fee_list: float,
    wp_params: dict,
) -> float:
    """Negative expected 3-year revenue: revenue * P(win)."""
    pricing = _build_pricing_from_vector(x, ach_mode, saas_arr_list, impl_fee_list)
    yearly = compute_three_year_financials(volumes, pricing)
    wp = win_probability(pricing, **wp_params)
    total_rev = sum(yr.total_revenue for yr in yearly.values())
    return -(total_rev * wp)


def _objective_best_strategy(
    x: np.ndarray,
    ach_mode: str,
    volumes: dict[int, VolumeForecastYear],
    saas_arr_list: float,
    impl_fee_list: float,
    wp_params: dict,
) -> float:
    """
    Negative expected lifetime value accounting for retention.
    ELV = P(win) * Σ(margin_y * retention_y)
    """
    pricing = _build_pricing_from_vector(x, ach_mode, saas_arr_list, impl_fee_list)
    yearly = compute_three_year_financials(volumes, pricing)
    wp = win_probability(pricing, **wp_params)

    elv = sum(
        yr.margin * RETENTION_CURVE.get(yr.year, 0.80)
        for yr in yearly.values()
    )
    return -(elv * wp)


def _run_single_optimization(
    objective_fn: Callable,
    ach_mode: str,
    volumes: dict[int, VolumeForecastYear],
    saas_arr_list: float,
    impl_fee_list: float,
    wp_params: dict,
) -> tuple[float, np.ndarray]:
    """Run differential_evolution for one ACH mode, return (best_obj, best_x)."""
    bounds = _get_bounds()
    result = differential_evolution(
        objective_fn,
        bounds=bounds,
        args=(ach_mode, volumes, saas_arr_list, impl_fee_list, wp_params),
        seed=42,
        maxiter=400,
        tol=1e-8,
        popsize=20,
        mutation=(0.5, 1.5),
        recombination=0.8,
    )
    return result.fun, result.x


def optimize_scenario(
    name: str,
    objective_fn: Callable,
    volumes: dict[int, VolumeForecastYear],
    saas_arr_list: float,
    impl_fee_list: float,
    wp_params: dict,
    explanation_fn: Callable | None = None,
) -> OptimizationResult:
    """
    Run optimization across all ACH modes and return the best result.
    """
    best_obj = float("inf")
    best_x = None
    best_mode = "percentage"

    for mode in cfg.ACH_MODES:
        obj, x = _run_single_optimization(
            objective_fn, mode, volumes,
            saas_arr_list, impl_fee_list, wp_params,
        )
        if obj < best_obj:
            best_obj = obj
            best_x = x
            best_mode = mode

    pricing = _build_pricing_from_vector(
        best_x, best_mode, saas_arr_list, impl_fee_list
    )
    yearly = compute_three_year_financials(volumes, pricing)
    wp = win_probability(pricing, **wp_params)

    explanation = ""
    if explanation_fn:
        explanation = explanation_fn(pricing, yearly, wp)

    return OptimizationResult(
        name=name,
        pricing=pricing,
        win_prob=wp,
        yearly=yearly,
        objective_value=-best_obj,
        explanation=explanation,
    )


def _best_strategy_explanation(
    pricing: PricingScenario,
    yearly: dict[int, YearlyRevenue],
    wp: float,
) -> str:
    total_margin = sum(yr.margin for yr in yearly.values())
    total_rev = sum(yr.total_revenue for yr in yearly.values())
    avg_take = sum(yr.take_rate for yr in yearly.values()) / 3

    parts = [f"Win probability: {wp:.0%}"]

    if pricing.saas_arr_discount_pct > 0.20:
        parts.append(
            f"Recommends a significant SaaS discount ({pricing.saas_arr_discount_pct:.0%}) "
            "to improve competitiveness and secure the deal."
        )

    if pricing.cc_base_rate < 0.021:
        parts.append(
            "Aggressive CC rate to maximize win probability, accepting lower CC margin."
        )
    elif pricing.cc_base_rate >= 0.023:
        parts.append(
            "Maintains strong CC rate near list price — volume supports it."
        )

    mode_desc = {
        "percentage": f"ACH at {pricing.ach_pct_rate:.2%}",
        "capped": f"ACH at {pricing.ach_pct_rate:.2%} capped at ${pricing.ach_cap:.2f}",
        "fixed_fee": f"ACH fixed at ${pricing.ach_fixed_fee:.2f}/txn",
    }
    parts.append(f"ACH structure: {mode_desc.get(pricing.ach_mode, pricing.ach_mode)}")

    parts.append(
        f"Projects ${total_margin:,.0f} total margin and "
        f"${total_rev:,.0f} total revenue over 3 years at {avg_take:.2%} avg take rate."
    )

    return " | ".join(parts)


def run_all_optimizations(
    volumes: dict[int, VolumeForecastYear],
    saas_arr_list: float = cfg.SAAS_ARR_DEFAULT,
    impl_fee_list: float = cfg.SAAS_IMPL_FEE_DEFAULT,
    wp_params: dict | None = None,
) -> dict[str, OptimizationResult]:
    """
    Run all three optimization scenarios.
    Returns {"margin": ..., "revenue": ..., "best_strategy": ...}.
    """
    wp = wp_params or {}

    margin_result = optimize_scenario(
        name="Margin Optimized",
        objective_fn=_objective_margin,
        volumes=volumes,
        saas_arr_list=saas_arr_list,
        impl_fee_list=impl_fee_list,
        wp_params=wp,
    )

    revenue_result = optimize_scenario(
        name="Revenue Optimized",
        objective_fn=_objective_revenue,
        volumes=volumes,
        saas_arr_list=saas_arr_list,
        impl_fee_list=impl_fee_list,
        wp_params=wp,
    )

    best_result = optimize_scenario(
        name="Best Strategy",
        objective_fn=_objective_best_strategy,
        volumes=volumes,
        saas_arr_list=saas_arr_list,
        impl_fee_list=impl_fee_list,
        wp_params=wp,
        explanation_fn=_best_strategy_explanation,
    )

    return {
        "margin": margin_result,
        "revenue": revenue_result,
        "best_strategy": best_result,
    }
