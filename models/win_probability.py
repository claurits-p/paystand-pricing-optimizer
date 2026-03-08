"""
Win probability model.

Maps a set of pricing levers to a probability of winning the deal
using a composite competitiveness score fed through a tunable sigmoid.

Uses range-based normalization so each lever's full practical range
maps to [-1, +1], making weights directly control relative influence.
"""
from __future__ import annotations
import copy
import math

from scipy.optimize import brentq

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
    lever_cap: float | None = None,
    saas_cap: float | None = None,
) -> float:
    """
    Compute a single competitiveness score from pricing levers.
    Positive = more competitive than market, negative = more expensive.

    Each lever is range-normalized to [-1, +1] before weighting,
    then each lever's contribution is clamped to ±lever_cap so no
    single concession can dominate the win probability.
    SaaS gets its own (optionally higher) cap via saas_cap.
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
    if lever_cap is not None:
        hold_clamped = max(-lever_cap, min(lever_cap, hold_clamped))

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

    total = 0.0
    for k, v in scores.items():
        contrib = w.get(k, 0) * v
        cap = saas_cap if (k == "saas_discount" and saas_cap is not None) else lever_cap
        if cap is not None:
            contrib = max(-cap, min(cap, contrib))
        total += contrib

    return total + hold_clamped


def win_probability(
    pricing: PricingScenario,
    floor: float | None = None,
    ceiling: float | None = None,
    steepness: float | None = None,
    benchmarks: dict | None = None,
    weights: dict | None = None,
    max_lever_impact: float | None = None,
) -> float:
    """
    P(win) = floor + (ceiling - floor) * sigmoid(steepness * composite_score)

    Each lever's composite contribution is capped so that no single
    concession can shift P(win) by more than `max_lever_impact` (default 10%).
    Returns a value in [floor, ceiling].
    """
    f = floor if floor is not None else cfg.WIN_PROB_DEFAULTS["floor"]
    c = ceiling if ceiling is not None else cfg.WIN_PROB_DEFAULTS["ceiling"]
    s = steepness if steepness is not None else cfg.WIN_PROB_DEFAULTS["steepness"]
    mli = max_lever_impact if max_lever_impact is not None else cfg.WIN_PROB_DEFAULTS.get("max_lever_impact", 0.10)
    msi = cfg.WIN_PROB_DEFAULTS.get("max_saas_impact", mli)

    pwin_range = c - f
    lever_cap = None
    saas_cap = None
    if pwin_range > 0 and s > 0:
        target_sigmoid = min(0.5 + mli / pwin_range, 0.999)
        lever_cap = math.log(target_sigmoid / (1 - target_sigmoid)) / s

        saas_sigmoid = min(0.5 + msi / pwin_range, 0.999)
        saas_cap = math.log(saas_sigmoid / (1 - saas_sigmoid)) / s

    score = composite_score(pricing, benchmarks, weights, lever_cap=lever_cap, saas_cap=saas_cap)
    sigmoid = 1.0 / (1.0 + math.exp(-s * score))

    return f + (c - f) * sigmoid


def win_probability_uncapped(
    pricing: PricingScenario,
    floor: float | None = None,
    ceiling: float | None = None,
    steepness: float | None = None,
    benchmarks: dict | None = None,
    weights: dict | None = None,
    **kwargs,
) -> float:
    """P(win) without per-lever caps — used for boost what-if analysis."""
    f = floor if floor is not None else cfg.WIN_PROB_DEFAULTS["floor"]
    c = ceiling if ceiling is not None else cfg.WIN_PROB_DEFAULTS["ceiling"]
    s = steepness if steepness is not None else cfg.WIN_PROB_DEFAULTS["steepness"]

    score = composite_score(pricing, benchmarks, weights, lever_cap=None, saas_cap=None)
    sigmoid = 1.0 / (1.0 + math.exp(-s * score))
    return f + (c - f) * sigmoid


def solve_saas_for_target_win_rate(
    pricing: PricingScenario,
    target_wp: float,
    wp_params: dict,
) -> float | None:
    """Find the SaaS discount that achieves *target_wp*, all other levers fixed.

    Uses the uncapped win probability model so the boost analysis
    isn't blocked by per-lever caps.

    Returns the required saas_arr_discount_pct, or None if the target
    cannot be reached within the allowed range [0, 0.70].
    """
    lo = 0.0
    hi = cfg.LEVER_BOUNDS["saas_arr_discount_pct"]["max"]

    def _wp_at_discount(d: float) -> float:
        p = copy.copy(pricing)
        p.saas_arr_discount_pct = d
        return win_probability_uncapped(p, **wp_params) - target_wp

    wp_lo = _wp_at_discount(lo)
    wp_hi = _wp_at_discount(hi)

    if wp_lo >= 0:
        return lo
    if wp_hi <= 0:
        return None

    try:
        return brentq(_wp_at_discount, lo, hi, xtol=1e-4)
    except ValueError:
        return None


def solve_multi_lever_for_target_win_rate(
    pricing: PricingScenario,
    target_wp: float,
    wp_params: dict,
) -> dict | None:
    """Find lever adjustments to achieve *target_wp* (uncapped model),
    trying SaaS first, then CC rate, then ACH rate in priority order.

    Returns a dict with the adjusted pricing and which levers changed,
    or None if the target is unreachable.
    """
    adjusted = copy.copy(pricing)
    changes = {}

    wp_fn = win_probability_uncapped

    current_wp = wp_fn(adjusted, **wp_params)
    if current_wp >= target_wp:
        return {"pricing": adjusted, "changes": changes}

    # 1) Try SaaS discount first
    saas_lo = adjusted.saas_arr_discount_pct
    saas_hi = cfg.LEVER_BOUNDS["saas_arr_discount_pct"]["max"]

    if saas_hi > saas_lo:
        def _wp_saas(d):
            p = copy.copy(adjusted)
            p.saas_arr_discount_pct = d
            return wp_fn(p, **wp_params) - target_wp

        if _wp_saas(saas_hi) >= 0:
            try:
                result = brentq(_wp_saas, saas_lo, saas_hi, xtol=1e-4)
                changes["saas_arr_discount_pct"] = (pricing.saas_arr_discount_pct, result)
                adjusted.saas_arr_discount_pct = result
                return {"pricing": adjusted, "changes": changes}
            except ValueError:
                pass

        adjusted.saas_arr_discount_pct = saas_hi
        if saas_hi > pricing.saas_arr_discount_pct:
            changes["saas_arr_discount_pct"] = (pricing.saas_arr_discount_pct, saas_hi)

    # 2) Try CC base rate + AMEX together (lower = more competitive)
    cc_hi = adjusted.cc_base_rate
    cc_lo = cfg.LEVER_BOUNDS["cc_base_rate"]["min"]
    amex_hi = adjusted.cc_amex_rate
    amex_lo = cfg.LEVER_BOUNDS["cc_amex_rate"]["min"]

    if cc_hi > cc_lo or amex_hi > amex_lo:
        def _wp_cc(t):
            p = copy.copy(adjusted)
            frac = max(0.0, min(1.0, t))
            if cc_hi > cc_lo:
                p.cc_base_rate = cc_hi - frac * (cc_hi - cc_lo)
            if amex_hi > amex_lo:
                p.cc_amex_rate = amex_hi - frac * (amex_hi - amex_lo)
            return wp_fn(p, **wp_params) - target_wp

        if _wp_cc(1.0) >= 0:
            try:
                result = brentq(_wp_cc, 0.0, 1.0, xtol=1e-5)
                new_base = cc_hi - result * (cc_hi - cc_lo) if cc_hi > cc_lo else cc_hi
                new_amex = amex_hi - result * (amex_hi - amex_lo) if amex_hi > amex_lo else amex_hi
                if abs(new_base - pricing.cc_base_rate) > 1e-5:
                    changes["cc_base_rate"] = (pricing.cc_base_rate, new_base)
                if abs(new_amex - pricing.cc_amex_rate) > 1e-5:
                    changes["cc_amex_rate"] = (pricing.cc_amex_rate, new_amex)
                adjusted.cc_base_rate = new_base
                adjusted.cc_amex_rate = new_amex
                return {"pricing": adjusted, "changes": changes}
            except ValueError:
                pass

        adjusted.cc_base_rate = cc_lo if cc_hi > cc_lo else cc_hi
        adjusted.cc_amex_rate = amex_lo if amex_hi > amex_lo else amex_hi
        if cc_lo < pricing.cc_base_rate:
            changes["cc_base_rate"] = (pricing.cc_base_rate, cc_lo)
        if amex_lo < pricing.cc_amex_rate:
            changes["cc_amex_rate"] = (pricing.cc_amex_rate, amex_lo)

    # 3) Try ACH rate (lower = more competitive)
    if adjusted.ach_mode == "percentage":
        ach_hi = adjusted.ach_pct_rate
        ach_lo = cfg.LEVER_BOUNDS["ach_pct_rate"]["min"]

        if ach_hi > ach_lo:
            def _wp_ach(r):
                p = copy.copy(adjusted)
                p.ach_pct_rate = r
                return wp_fn(p, **wp_params) - target_wp

            if _wp_ach(ach_lo) >= 0:
                try:
                    result = brentq(_wp_ach, ach_lo, ach_hi, xtol=1e-5)
                    changes["ach_pct_rate"] = (pricing.ach_pct_rate, result)
                    adjusted.ach_pct_rate = result
                    return {"pricing": adjusted, "changes": changes}
                except ValueError:
                    pass

            adjusted.ach_pct_rate = ach_lo
            if ach_lo < pricing.ach_pct_rate:
                changes["ach_pct_rate"] = (pricing.ach_pct_rate, ach_lo)

    elif adjusted.ach_mode == "capped":
        ach_hi = adjusted.ach_pct_rate
        ach_lo = cfg.LEVER_BOUNDS["ach_pct_rate"]["min"]
        cap_hi = adjusted.ach_cap
        cap_lo = cfg.LEVER_BOUNDS["ach_cap"]["min"]

        if ach_hi > ach_lo or cap_hi > cap_lo:
            def _wp_ach_capped(t):
                p = copy.copy(adjusted)
                frac = max(0.0, min(1.0, t))
                p.ach_pct_rate = ach_hi - frac * (ach_hi - ach_lo)
                p.ach_cap = cap_hi - frac * (cap_hi - cap_lo)
                return wp_fn(p, **wp_params) - target_wp

            if _wp_ach_capped(1.0) >= 0:
                try:
                    result = brentq(_wp_ach_capped, 0.0, 1.0, xtol=1e-4)
                    new_rate = ach_hi - result * (ach_hi - ach_lo)
                    new_cap = cap_hi - result * (cap_hi - cap_lo)
                    changes["ach_pct_rate"] = (pricing.ach_pct_rate, new_rate)
                    changes["ach_cap"] = (pricing.ach_cap, new_cap)
                    adjusted.ach_pct_rate = new_rate
                    adjusted.ach_cap = new_cap
                    return {"pricing": adjusted, "changes": changes}
                except ValueError:
                    pass

            adjusted.ach_pct_rate = ach_lo
            adjusted.ach_cap = cap_lo
            if ach_lo < pricing.ach_pct_rate:
                changes["ach_pct_rate"] = (pricing.ach_pct_rate, ach_lo)
            if cap_lo < pricing.ach_cap:
                changes["ach_cap"] = (pricing.ach_cap, cap_lo)

    elif adjusted.ach_mode == "fixed_fee":
        ach_hi = adjusted.ach_fixed_fee
        ach_lo = cfg.LEVER_BOUNDS["ach_fixed_fee"]["min"]

        if ach_hi > ach_lo:
            def _wp_ach_fixed(r):
                p = copy.copy(adjusted)
                p.ach_fixed_fee = r
                return wp_fn(p, **wp_params) - target_wp

            if _wp_ach_fixed(ach_lo) >= 0:
                try:
                    result = brentq(_wp_ach_fixed, ach_lo, ach_hi, xtol=1e-4)
                    changes["ach_fixed_fee"] = (pricing.ach_fixed_fee, result)
                    adjusted.ach_fixed_fee = result
                    return {"pricing": adjusted, "changes": changes}
                except ValueError:
                    pass

            adjusted.ach_fixed_fee = ach_lo
            if ach_lo < pricing.ach_fixed_fee:
                changes["ach_fixed_fee"] = (pricing.ach_fixed_fee, ach_lo)

    final_wp = wp_fn(adjusted, **wp_params)
    if final_wp >= target_wp - 0.005:
        return {"pricing": adjusted, "changes": changes}

    return None
