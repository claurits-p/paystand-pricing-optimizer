"""
Default assumptions, market benchmarks, and pricing lever bounds.
All values are configurable via the Streamlit UI.
"""

# ── SaaS Defaults ──────────────────────────────────────────────
SAAS_ARR_DEFAULT = 25_000          # $/year (standard list price)
SAAS_IMPL_FEE_DEFAULT = 3_000      # $ one-time implementation fee
SAAS_ARR_MARGIN = 0.85             # 85% margin on ARR

# ── CC Defaults ────────────────────────────────────────────────
CC_STANDARD_BASE_RATE = 0.0239     # 2.39% standard base CC rate
CC_STANDARD_AMEX_RATE = 0.035      # 3.50% AMEX standard
CC_STANDARD_BLENDED = 0.032        # 3.20% blended at standard (base 2.39% + AMEX 3.50%)
CC_FIXED_COMPONENT = 0.0053        # 0.53% fixed component (mid-tier cards, assessments, etc.)
CC_BASE_VOLUME_SHARE = 0.75        # 75% of CC volume at base rate
CC_AMEX_VOLUME_SHARE = 0.25        # 25% of CC volume at AMEX rate
CC_COST_RATE = 0.024               # 2.40% blended cost (interchange + assessments + markup)

# ── ACH / Bank Network Defaults ───────────────────────────────
ACH_STANDARD_RATE = 0.0049         # 0.49% standard revenue rate
ACH_COST_PER_TXN = 0.13           # $0.13 per transaction cost
ACH_AVG_TXN_SIZE = 1_700          # $1,700 average transaction size

# ── Hold Time (per payment type) ──────────────────────────────
HOLD_DAYS_CC_DEFAULT = 2
HOLD_DAYS_ACH_DEFAULT = 6
HOLD_DAYS_BANK_DEFAULT = 4
FLOAT_ANNUAL_RATE = 0.04           # 4% return on float balances
FLOAT_CALENDAR_FACTOR = 7 / 5     # convert business hold days to calendar days
SAAS_ANNUAL_ESCALATOR = 0.07       # 7% annual increase on standard ARR

# ── Pricing Lever Bounds ──────────────────────────────────────
LEVER_BOUNDS = {
    "saas_arr_discount_pct": {"min": 0.0, "max": 0.70, "default": 0.0, "step": 0.05},
    "impl_fee_discount_pct":  {"min": 0.0, "max": 1.0,  "default": 0.0, "step": 0.10},
    "cc_base_rate":           {"min": 0.0199, "max": 0.0239, "default": 0.0239, "step": 0.001},
    "cc_amex_rate":           {"min": 0.0315, "max": 0.035, "default": 0.035, "step": 0.005},
    "ach_pct_rate":           {"min": 0.0019, "max": 0.0049, "default": 0.0049, "step": 0.0005},
    "ach_cap":                {"min": 2.50, "max": 10.0, "default": 5.0, "step": 0.50},
    "ach_fixed_fee":          {"min": 1.00, "max": 5.00, "default": 2.50, "step": 0.25},
    "hold_days_cc":           {"min": 1, "max": 2, "default": 2, "step": 1},
    "hold_days_ach":          {"min": 1, "max": 7, "default": 6, "step": 1},
    "hold_days_bank":         {"min": 1, "max": 5, "default": 4, "step": 1},
}

# ── ACH Pricing Modes ─────────────────────────────────────────
ACH_MODES = ["percentage", "capped", "fixed_fee"]

# ── Win Probability Defaults ──────────────────────────────────
WIN_PROB_DEFAULTS = {
    "floor": 0.05,        # 5% min win rate
    "ceiling": 1.0,       # 100% max win rate
    "steepness": 9.0,     # sigmoid steepness
    "weights": {
        "cc_rate": 0.35,
        "saas_discount": 0.50,
        "ach_rate": 0.20,
        "impl_discount": 0.05,
        "hold_time": 0.05,
    },
}

# ── Market Benchmarks (median competitive rates) ──────────────
MARKET_BENCHMARKS = {
    "cc_rate": 0.032,              # 3.2% standard blended CC rate (benchmark)
    "ach_effective_rate": 0.003,   # 0.3% typical competitive ACH effective rate
    "saas_discount_pct": 0.10,    # typical discount given
    "impl_discount_pct": 0.25,    # typical impl fee reduction
    "hold_days_cc": 2,            # typical CC hold
    "hold_days_ach": 6,           # typical ACH hold
    "hold_days_bank": 4,          # typical bank network hold
}
