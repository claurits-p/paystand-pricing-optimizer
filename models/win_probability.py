"""
Win probability model.

Maps a set of pricing levers to a probability of winning the deal
using a composite competitiveness score fed through a tunable sigmoid.

Uses range-based normalization so each lever's full practical range
maps to [-1, +1], making weights directly control relative influence.
"""
from __future__ import annotations
import math

import config as cfg
from models.revenue_model import PricingScenario


def _blended_cc(base_rate: float, amex_rate: float) -> float:
    return (
        cfg.CC_FIXED_COMPONENT
        + base_rate * cfg.CC_BASE_VOLUME_SHARE
        + amex_rate * cfg.CC_AMEX_VOLUME_SHARE
    )


CC_BLENDED_BEST = _blended_cc(
    cfg.LEVER_BOUNDS["cc_base_rate"]["min"],
    cfg.LEVER_BOUNDS["cc_amex_rate"]["min"],
)
CC_BLENDED_WORST = _blended_cc(
    cfg.LEVER_BOUNDS["cc_base_rate"]["max"],
    cfg.LEVER_BOUNDS["cc_amex_rate"]["max"],
)

ACH_EFF_BEST = cfg.LEVER_BOUNDS["ach_pct_rate"]["min"]
ACH_EFF_WORST = cfg.LEVER_BOUNDS["ach_pct_rate"]["max"]

SAAS_DISC_BEST = cfg.LEVER_BOUNDS["saas_arr_discount_pct"]["max"]
SAAS_DISC_WORST = cfg.LEVER_BOUNDS["saas_arr_discount_pct"]["min"]

IMPL_DISC_BEST = cfg.LEVER_BOUNDS["impl_fee_discount_pct"]["max"]
IMPL_DISC_WORST = cfg.LEVER_BOUNDS["impl_fee_discount_pct"]["min"]


def _range_norm_lower(value: float, benchmark: float,
                      best: float, worst: float) -> float:
    """
    Range-based normalization where lower is better.
    Returns +1 at best (lowest), 0 at benchmark, -1 at worst (highest).
    Clamps output to [-1, +1].
    """
    if value <= benchmark:
        denom = benchmark - best
        if denom == 0:
            return 0.0
        return min(1.0, (benchmark - value) / denom)
    else:
        denom = worst - benchmark
        if denom == 0:
            return 0.0
        return max(-1.0, -(value - benchmark) / denom)


def _range_norm_higher(value: float, benchmark: float,
                       best: float, worst: float) -> float:
    """
    Range-based normalization where higher is better.
    Returns +1 at best (highest), 0 at benchmark, -1 at worst (lowest).
    Clamps output to [-1, +1].
    """
    if value >= benchmark:
        denom = best - benchmark
        if denom == 0:
            return 0.0
        return min(1.0, (value - benchmark) / denom)
    else:
        denom = benchmark - worst
        if denom == 0:
            return 0.0
        return max(-1.0, -(benchmark - value) / denom)


def effective_ach_rate(pricing: PricingScenario) -> float:
    """
    Convert any ACH pricing mode to an effective rate for comparison.
    Uses the average transaction size to normalize.
    """
    avg = cfg.ACH_AVG_TXN_SIZE
    if pricing.ach_mode == "percentage":
        return pricing.ach_pct_rate

    elif pricing.ach_mode == "capped":
        uncapped = avg * pricing.ach_pct_rate
        capped = min(uncapped, pricing.ach_cap)
        return capped / avg if avg > 0 else pricing.ach_pct_rate

    elif pricing.ach_mode == "fixed_fee":
        return pricing.ach_fixed_fee / avg if avg > 0 else 0.0

    return pricing.ach_pct_rate


MAX_HOLD_IMPACT = 0.017


def composite_score(
    pricing: PricingScenario,
    benchmarks: dict | None = None,
    weights: dict | None = None,
) -> float:
    """
    Compute a single competitiveness score from pricing levers.
    Positive = more competitive than market, negative = more expensive.

    Each lever is range-normalized to [-1, +1] before weighting,
    so weights directly control each lever's share of influence.
    """
    bm = benchmarks or cfg.MARKET_BENCHMARKS
    w = weights or cfg.WIN_PROB_DEFAULTS["weights"]

    blended_cc = _blended_cc(pricing.cc_base_rate, pricing.cc_amex_rate)

    hold_raw = (
        _range_norm_lower(pricing.hold_days_cc, bm["hold_days_cc"],
                          cfg.LEVER_BOUNDS["hold_days_cc"]["min"],
                          cfg.LEVER_BOUNDS["hold_days_cc"]["max"]) * 0.30
        + _range_norm_lower(pricing.hold_days_ach, bm["hold_days_ach"],
                            cfg.LEVER_BOUNDS["hold_days_ach"]["min"],
                            cfg.LEVER_BOUNDS["hold_days_ach"]["max"]) * 0.50
        + _range_norm_lower(pricing.hold_days_bank, bm["hold_days_bank"],
                            cfg.LEVER_BOUNDS["hold_days_bank"]["min"],
                            cfg.LEVER_BOUNDS["hold_days_bank"]["max"]) * 0.20
    )
    hold_weighted = w.get("hold_time", 0) * hold_raw
    hold_clamped = max(-MAX_HOLD_IMPACT, min(MAX_HOLD_IMPACT, hold_weighted))

    scores = {
        "cc_rate": _range_norm_lower(
            blended_cc, bm["cc_rate"], CC_BLENDED_BEST, CC_BLENDED_WORST,
        ),
        "ach_rate": _range_norm_lower(
            effective_ach_rate(pricing), bm["ach_effective_rate"],
            ACH_EFF_BEST, ACH_EFF_WORST,
        ),
        "saas_discount": _range_norm_higher(
            pricing.saas_arr_discount_pct, bm["saas_discount_pct"],
            SAAS_DISC_BEST, SAAS_DISC_WORST,
        ),
        "impl_discount": _range_norm_higher(
            pricing.impl_fee_discount_pct, bm["impl_discount_pct"],
            IMPL_DISC_BEST, IMPL_DISC_WORST,
        ),
    }

    other_score = sum(w.get(k, 0) * v for k, v in scores.items())
    return other_score + hold_clamped


def win_probability(
    pricing: PricingScenario,
    floor: float | None = None,
    ceiling: float | None = None,
    steepness: float | None = None,
    benchmarks: dict | None = None,
    weights: dict | None = None,
) -> float:
    """
    P(win) = floor + (ceiling - floor) * sigmoid(steepness * composite_score)

    Returns a value in [floor, ceiling].
    """
    f = floor if floor is not None else cfg.WIN_PROB_DEFAULTS["floor"]
    c = ceiling if ceiling is not None else cfg.WIN_PROB_DEFAULTS["ceiling"]
    s = steepness if steepness is not None else cfg.WIN_PROB_DEFAULTS["steepness"]

    score = composite_score(pricing, benchmarks, weights)
    sigmoid = 1.0 / (1.0 + math.exp(-s * score))

    return f + (c - f) * sigmoid
