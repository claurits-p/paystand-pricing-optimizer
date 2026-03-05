"""
Revenue model for payment processing.
Calculates revenue from pricing levers applied to volume forecast.
"""
from __future__ import annotations
from dataclasses import dataclass

import config as cfg
from models.volume_forecast import VolumeForecastYear
from models.cost_model import YearlyCosts, compute_yearly_costs


@dataclass
class PricingScenario:
    """A complete set of pricing levers."""
    saas_arr_discount_pct: float   # 0.0 – 0.70
    impl_fee_discount_pct: float   # 0.0 – 1.0  (1.0 = fully waived)
    cc_base_rate: float            # e.g. 0.0259
    cc_amex_rate: float            # e.g. 0.035
    ach_mode: str                  # "percentage" | "capped" | "fixed_fee"
    ach_pct_rate: float            # used in percentage and capped modes
    ach_cap: float                 # used in capped mode only ($ per txn)
    ach_fixed_fee: float           # used in fixed_fee mode only ($ per txn)
    hold_days_cc: int
    hold_days_ach: int
    hold_days_bank: int

    # Derived fields populated at deal level
    saas_arr_list: float = cfg.SAAS_ARR_DEFAULT
    impl_fee_list: float = cfg.SAAS_IMPL_FEE_DEFAULT

    @property
    def effective_saas_arr(self) -> float:
        return self.saas_arr_list * (1 - self.saas_arr_discount_pct)

    @property
    def effective_impl_fee(self) -> float:
        return self.impl_fee_list * (1 - self.impl_fee_discount_pct)


@dataclass
class YearlyRevenue:
    year: int
    saas_revenue: float
    impl_fee_revenue: float
    cc_revenue: float
    ach_revenue: float
    bank_network_revenue: float
    float_income: float
    total_revenue: float
    total_cost: float
    margin: float
    take_rate: float               # total_revenue / total_volume


def _ach_revenue_for_volume(
    volume: float,
    txn_count: int,
    pricing: PricingScenario,
) -> float:
    """Calculate ACH revenue given volume, txn count, and pricing mode."""
    if pricing.ach_mode == "percentage":
        return volume * pricing.ach_pct_rate

    elif pricing.ach_mode == "capped":
        per_txn_pct = (cfg.ACH_AVG_TXN_SIZE * pricing.ach_pct_rate)
        per_txn = min(per_txn_pct, pricing.ach_cap)
        return txn_count * per_txn

    elif pricing.ach_mode == "fixed_fee":
        return txn_count * pricing.ach_fixed_fee

    return volume * pricing.ach_pct_rate


def compute_yearly_revenue(
    vol: VolumeForecastYear,
    pricing: PricingScenario,
    costs: YearlyCosts,
) -> YearlyRevenue:
    """Compute revenue for a single year."""
    saas_rev = pricing.effective_saas_arr
    impl_rev = pricing.effective_impl_fee if vol.year == 1 else 0.0

    blended_cc_rate = (
        cfg.CC_FIXED_COMPONENT
        + pricing.cc_base_rate * cfg.CC_BASE_VOLUME_SHARE
        + pricing.cc_amex_rate * cfg.CC_AMEX_VOLUME_SHARE
    )
    cc_rev = vol.cc * blended_cc_rate

    ach_rev = _ach_revenue_for_volume(vol.ach, vol.ach_txn_count, pricing)
    bank_rev = _ach_revenue_for_volume(
        vol.bank_network, vol.bank_network_txn_count, pricing
    )

    cc_float = (vol.cc / 365) * pricing.hold_days_cc * cfg.MONEY_MARKET_RATE / 365
    ach_float = (vol.ach / 365) * pricing.hold_days_ach * cfg.MONEY_MARKET_RATE / 365
    bank_float = (vol.bank_network / 365) * pricing.hold_days_bank * cfg.MONEY_MARKET_RATE / 365
    float_income = cc_float + ach_float + bank_float

    total_rev = saas_rev + impl_rev + cc_rev + ach_rev + bank_rev + float_income
    margin = total_rev - costs.total
    take_rate = total_rev / vol.total if vol.total > 0 else 0.0

    return YearlyRevenue(
        year=vol.year,
        saas_revenue=saas_rev,
        impl_fee_revenue=impl_rev,
        cc_revenue=cc_rev,
        ach_revenue=ach_rev,
        bank_network_revenue=bank_rev,
        float_income=float_income,
        total_revenue=total_rev,
        total_cost=costs.total,
        margin=margin,
        take_rate=take_rate,
    )


def compute_three_year_financials(
    volumes: dict[int, VolumeForecastYear],
    pricing: PricingScenario,
) -> dict[int, YearlyRevenue]:
    """Full 3-year financial projection for a pricing scenario."""
    results: dict[int, YearlyRevenue] = {}
    for year in [1, 2, 3]:
        vol = volumes[year]
        costs = compute_yearly_costs(vol, pricing.effective_saas_arr)
        results[year] = compute_yearly_revenue(vol, pricing, costs)
    return results
