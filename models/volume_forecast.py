"""
3-year volume forecast ported from existing Colab model.
Predicts CC, ACH, and bank network volumes based on convenience fee
scenarios and processing tier adoption curves.
"""
from __future__ import annotations
from dataclasses import dataclass

import config as cfg


@dataclass
class VolumeForecastYear:
    year: int
    total: float
    cc: float
    ach: float
    bank_network: float

    @property
    def ach_txn_count(self) -> int:
        """Estimated number of ACH transactions based on avg txn size."""
        if cfg.ACH_AVG_TXN_SIZE <= 0:
            return 0
        return int(self.ach / cfg.ACH_AVG_TXN_SIZE)

    @property
    def bank_network_txn_count(self) -> int:
        if cfg.ACH_AVG_TXN_SIZE <= 0:
            return 0
        return int(self.bank_network / cfg.ACH_AVG_TXN_SIZE)


def forecast_volume_y1_y3(
    processing_tier_volume: float,
    expected_cc_volume: float,
    conv_fee_with_paystand: float,
    conv_fee_today: float,
) -> dict[int, VolumeForecastYear]:
    """
    Returns {1: VolumeForecastYear, 2: ..., 3: ...}.

    CC drop logic:
      - Using CF today AND with Paystand → 20% CC drop
      - Using CF only with Paystand (not today) → 60% CC drop
      - Dropped CC splits 80% ACH / 20% bank network

    Delta adoption (volume beyond current CC):
      - Year 1: 30%, Year 2: 50%, Year 3: 60%
      - Split: 30% CC / 60% ACH / 10% bank network
      - Year 1 CC gets a 2/3 ramp factor
    """
    uses_cf_today = conv_fee_today > 0
    uses_cf_paystand = conv_fee_with_paystand > 0

    baseline_cc = expected_cc_volume
    baseline_ach = 0.0
    baseline_bank = 0.0

    if uses_cf_paystand and uses_cf_today:
        cc_drop_pct = 0.20
    elif uses_cf_paystand and not uses_cf_today:
        cc_drop_pct = 0.60
    else:
        cc_drop_pct = 0.0

    if cc_drop_pct > 0:
        dropped_cc = baseline_cc * cc_drop_pct
        baseline_cc -= dropped_cc
        baseline_ach += dropped_cc * 0.80
        baseline_bank += dropped_cc * 0.20

    delta = max(processing_tier_volume - expected_cc_volume, 0)

    adoption = {1: 0.30, 2: 0.50, 3: 0.60}
    split = {"cc": 0.30, "ach": 0.60, "bank": 0.10}

    forecast: dict[int, VolumeForecastYear] = {}

    for year in [1, 2, 3]:
        adopted = delta * adoption[year]

        cc = baseline_cc + adopted * split["cc"]
        ach = baseline_ach + adopted * split["ach"]
        bank = baseline_bank + adopted * split["bank"]

        if year == 1:
            cc = cc * (2 / 3)

        total = cc + ach + bank

        forecast[year] = VolumeForecastYear(
            year=year,
            total=total,
            cc=cc,
            ach=ach,
            bank_network=bank,
        )

    return forecast
