"""
Optimization engine.

Three strategy-based scenarios with different constraint profiles:
  1. Balanced:      SaaS-led strategy, CC near standard, maximize LTV * P(win)
  2. Aggressive:    maximize win probability with a minimum margin floor
  3. SaaS Passive:  deep persistent SaaS discount (50-70%), optimize processing
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
    _saas_arr_for_year,
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
    saas_discount_persists: bool = False,
) -> PricingScenario:
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
        saas_discount_persists=saas_discount_persists,
    )


def _get_bounds(strategy: str = "balanced") -> list[tuple[float, float]]:
    """Bounds for the 10-element vector, constrained by strategy."""
    lb = cfg.LEVER_BOUNDS

    if strategy == "aggressive":
        return [
            (0.30, lb["saas_arr_discount_pct"]["max"]),                   # saas discount: at least 30%
            (0.0, lb["impl_fee_discount_pct"]["max"]),
            (lb["cc_base_rate"]["min"], lb["cc_base_rate"]["max"]),
            (lb["cc_amex_rate"]["min"], lb["cc_amex_rate"]["max"]),
            (lb["ach_pct_rate"]["min"], lb["ach_pct_rate"]["max"]),
            (lb["ach_cap"]["min"], lb["ach_cap"]["max"]),
            (lb["ach_fixed_fee"]["min"], lb["ach_fixed_fee"]["max"]),
            (lb["hold_days_cc"]["min"], lb["hold_days_cc"]["max"]),
            (lb["hold_days_ach"]["min"], lb["hold_days_ach"]["max"]),
            (lb["hold_days_bank"]["min"], lb["hold_days_bank"]["max"]),
        ]
    elif strategy == "saas_passive":
        return [
            (0.50, 0.70),                                                     # saas discount: 50-70%, persists all years
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
    else:  # balanced: SaaS-led strategy, CC stays near standard
        return [
            (lb["saas_arr_discount_pct"]["min"], lb["saas_arr_discount_pct"]["max"]),
            (lb["impl_fee_discount_pct"]["min"], lb["impl_fee_discount_pct"]["max"]),
            (0.0219, 0.0239),                                                 # cc base: 2.19%-2.39% (near standard)
            (0.0330, 0.035),                                                  # amex: 3.30%-3.50% (near standard)
            (lb["ach_pct_rate"]["min"], lb["ach_pct_rate"]["max"]),
            (lb["ach_cap"]["min"], lb["ach_cap"]["max"]),
            (lb["ach_fixed_fee"]["min"], lb["ach_fixed_fee"]["max"]),
            (lb["hold_days_cc"]["min"], lb["hold_days_cc"]["max"]),
            (lb["hold_days_ach"]["min"], lb["hold_days_ach"]["max"]),
            (lb["hold_days_bank"]["min"], lb["hold_days_bank"]["max"]),
        ]


def _objective_margin(
    x, ach_mode, volumes, saas_arr_list, impl_fee_list, wp_params,
    saas_discount_persists=False,
) -> float:
    """Maximize 3-year margin * P(win)."""
    pricing = _build_pricing_from_vector(x, ach_mode, saas_arr_list, impl_fee_list, saas_discount_persists)
    yearly = compute_three_year_financials(volumes, pricing)
    wp = win_probability(pricing, **wp_params)
    total_margin = sum(yr.margin for yr in yearly.values())
    return -(total_margin * wp)


def _objective_win_prob(
    x, ach_mode, volumes, saas_arr_list, impl_fee_list, wp_params,
    saas_discount_persists=False,
) -> float:
    """Maximize win probability while keeping margin above a floor."""
    pricing = _build_pricing_from_vector(x, ach_mode, saas_arr_list, impl_fee_list, saas_discount_persists)
    yearly = compute_three_year_financials(volumes, pricing)
    wp = win_probability(pricing, **wp_params)
    total_rev = sum(yr.total_revenue for yr in yearly.values())
    total_margin = sum(yr.margin for yr in yearly.values())
    margin_pct = total_margin / total_rev if total_rev > 0 else 0
    if margin_pct < 0.35:
        return 0.0
    return -(wp * total_margin)


def _objective_ltv(
    x, ach_mode, volumes, saas_arr_list, impl_fee_list, wp_params,
    saas_discount_persists=False,
) -> float:
    """Maximize expected lifetime value: P(win) * sum(margin * retention)."""
    pricing = _build_pricing_from_vector(x, ach_mode, saas_arr_list, impl_fee_list, saas_discount_persists)
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
    strategy: str = "balanced",
    saas_discount_persists: bool = False,
) -> tuple[float, np.ndarray]:
    bounds = _get_bounds(strategy)
    result = differential_evolution(
        objective_fn,
        bounds=bounds,
        args=(ach_mode, volumes, saas_arr_list, impl_fee_list, wp_params, saas_discount_persists),
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
    strategy: str = "balanced",
    explanation_fn: Callable | None = None,
    saas_discount_persists: bool = False,
) -> OptimizationResult:
    best_obj = float("inf")
    best_x = None
    best_mode = "percentage"

    for mode in cfg.ACH_MODES:
        obj, x = _run_single_optimization(
            objective_fn, mode, volumes,
            saas_arr_list, impl_fee_list, wp_params,
            strategy=strategy,
            saas_discount_persists=saas_discount_persists,
        )
        if obj < best_obj:
            best_obj = obj
            best_x = x
            best_mode = mode

    pricing = _build_pricing_from_vector(
        best_x, best_mode, saas_arr_list, impl_fee_list, saas_discount_persists
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


def build_msrp_scenario(
    volumes: dict[int, VolumeForecastYear],
    saas_arr_list: float = cfg.SAAS_ARR_DEFAULT,
    impl_fee_list: float = cfg.SAAS_IMPL_FEE_DEFAULT,
    wp_params: dict | None = None,
) -> OptimizationResult:
    """MSRP / sticker price scenario -- no discounts, standard everything."""
    pricing = PricingScenario(
        saas_arr_discount_pct=0.0,
        impl_fee_discount_pct=0.0,
        cc_base_rate=cfg.CC_STANDARD_BASE_RATE,
        cc_amex_rate=cfg.CC_STANDARD_AMEX_RATE,
        ach_mode="percentage",
        ach_pct_rate=cfg.ACH_STANDARD_RATE,
        ach_cap=10.0,
        ach_fixed_fee=2.50,
        hold_days_cc=cfg.HOLD_DAYS_CC_DEFAULT,
        hold_days_ach=cfg.HOLD_DAYS_ACH_DEFAULT,
        hold_days_bank=cfg.HOLD_DAYS_BANK_DEFAULT,
        saas_arr_list=saas_arr_list,
        impl_fee_list=impl_fee_list,
    )
    yearly = compute_three_year_financials(volumes, pricing)
    wp = win_probability(pricing, **(wp_params or {}))

    return OptimizationResult(
        name="MSRP (Sticker Price)",
        pricing=pricing,
        win_prob=wp,
        yearly=yearly,
        objective_value=0.0,
        explanation="Standard pricing with no discounts or concessions.",
    )


def _scenario_explanation(
    pricing: PricingScenario,
    yearly: dict[int, YearlyRevenue],
    wp: float,
) -> str:
    total_margin = sum(yr.margin for yr in yearly.values())
    total_rev = sum(yr.total_revenue for yr in yearly.values())
    margin_pct = total_margin / total_rev * 100 if total_rev > 0 else 0
    renewal = _saas_arr_for_year(pricing, 2)

    processing_rev = sum(
        yr.cc_revenue + yr.ach_revenue + yr.bank_network_revenue
        for yr in yearly.values()
    )
    processing_pct = processing_rev / total_rev * 100 if total_rev > 0 else 0

    parts = []
    if pricing.saas_discount_persists:
        parts.append(
            f"Persistent {pricing.saas_arr_discount_pct:.0%} SaaS discount, "
            f"Y2 ARR at ${renewal:,.0f} (+7% escalator)"
        )
    elif pricing.saas_arr_discount_pct > 0.25:
        parts.append(
            f"Y1 SaaS discount of {pricing.saas_arr_discount_pct:.0%} drives win rate, "
            f"then renews at ${renewal:,.0f}/yr"
        )
    if pricing.cc_base_rate < cfg.CC_STANDARD_BASE_RATE:
        parts.append(f"CC promo at {pricing.cc_base_rate:.2%} reverts to {cfg.CC_STANDARD_BASE_RATE:.2%} in Y2")

    parts.append(f"{margin_pct:.0f}% blended margin over 3 years")
    parts.append(f"{processing_pct:.0f}% of revenue from processing")

    return " | ".join(parts)


def run_all_optimizations(
    volumes: dict[int, VolumeForecastYear],
    saas_arr_list: float = cfg.SAAS_ARR_DEFAULT,
    impl_fee_list: float = cfg.SAAS_IMPL_FEE_DEFAULT,
    wp_params: dict | None = None,
) -> dict[str, OptimizationResult]:
    """
    Run all optimization scenarios.
    Returns {"msrp": ..., "balanced": ..., "aggressive": ..., "saas_passive": ...}.
    """
    wp = wp_params or {}

    msrp = build_msrp_scenario(volumes, saas_arr_list, impl_fee_list, wp)

    balanced = optimize_scenario(
        name="Balanced",
        objective_fn=_objective_ltv,
        volumes=volumes,
        saas_arr_list=saas_arr_list,
        impl_fee_list=impl_fee_list,
        wp_params=wp,
        strategy="balanced",
        explanation_fn=_scenario_explanation,
    )

    aggressive = optimize_scenario(
        name="Aggressive (Land & Expand)",
        objective_fn=_objective_win_prob,
        volumes=volumes,
        saas_arr_list=saas_arr_list,
        impl_fee_list=impl_fee_list,
        wp_params=wp,
        strategy="aggressive",
        explanation_fn=_scenario_explanation,
    )

    saas_passive = optimize_scenario(
        name="SaaS Passive",
        objective_fn=_objective_ltv,
        volumes=volumes,
        saas_arr_list=saas_arr_list,
        impl_fee_list=impl_fee_list,
        wp_params=wp,
        strategy="saas_passive",
        explanation_fn=_scenario_explanation,
        saas_discount_persists=True,
    )

    return {
        "msrp": msrp,
        "balanced": balanced,
        "aggressive": aggressive,
        "saas_passive": saas_passive,
    }
