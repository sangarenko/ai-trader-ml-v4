#!/usr/bin/env python3
"""Slippage model — square-root market impact.

Estimates slippage (in basis points) for a market order on MOEX equities.
The model follows the classical square-root impact law:

    slippage_bps = base_bps + impact_bps * sqrt(qty / adv)

Liquidity is bucketed by ticker ADV (Average Daily Volume) into 3 classes:

    HIGH    : SBER, GAZP, LKOH        — base=2 bps,  impact=10 bps
    MEDIUM  : GMKN, VTBR, ROSN, TATN   — base=3 bps,  impact=15 bps
    LOW     : MGNT, MTSS, NVTK, PLZL    — base=5 bps,  impact=25 bps

Buy orders push price up → positive slippage (you pay more).
Sell orders push price down → negative slippage (you receive less).

Examples:
    >>> estimate_slippage(100, 270, 5_000_000, "SBER", "buy")
    2.0447...    # 2 bps base + 10*sqrt(100/5e6) = 2 + 0.0447 bps
    >>> estimate_slippage(100_000, 270, 5_000_000, "SBER", "buy")
    3.4142...    # 2 + 10*sqrt(0.02) = 2 + 1.414 bps
    >>> estimate_slippage(100_000, 270, 5_000_000, "SBER", "sell")
    -3.4142...
"""
from typing import Union

import numpy as np


# --------------------------------------------------------------------------- #
# Liquidity classes (by ADV / ticker)
# --------------------------------------------------------------------------- #
_LIQUIDITY_CLASS = {
    # HIGH — most liquid RU equities (ADV > 5M shares/day typically)
    "SBER": "high",
    "GAZP": "high",
    "LKOH": "high",
    # MEDIUM — liquid blue chips (ADV 1-5M shares/day)
    "GMKN": "medium",
    "VTBR": "medium",
    "ROSN": "medium",
    "TATN": "medium",
    # LOW — less liquid blue chips (ADV < 1M shares/day)
    "MGNT": "low",
    "MTSS": "low",
    "NVTK": "low",
    "PLZL": "low",
}

_CLASS_PARAMS = {
    "high":   {"base_bps": 2.0, "impact_bps": 10.0},
    "medium": {"base_bps": 3.0, "impact_bps": 15.0},
    "low":    {"base_bps": 5.0, "impact_bps": 25.0},
}


def get_liquidity_class(ticker: str) -> str:
    """Return 'high' | 'medium' | 'low' for a ticker (case-insensitive).

    Unknown tickers default to 'low' (conservative).
    """
    if not ticker:
        return "low"
    return _LIQUIDITY_CLASS.get(str(ticker).strip().upper(), "low")


def estimate_slippage(
    qty: Union[int, float],
    price: float,
    adv: float,
    ticker: str,
    side: str,
) -> float:
    """Estimate slippage in basis points.

    Args:
        qty: order quantity (shares)
        price: current price (per share) — unused in the model itself but
            kept in the signature because callers (and a future price-aware
            extension of the model) need it.
        adv: average daily volume (shares traded per day, ~20-day mean)
        ticker: ticker symbol for liquidity class lookup (case-insensitive)
        side: 'buy' or 'sell'

    Returns:
        slippage_bps: basis points (1 bps = 0.01%). Sign convention:
            buy  → positive (price pushed up, you pay more)
            sell → negative (price pushed down, you receive less)

    Model: square-root impact
        slippage_bps = base_bps + impact_bps * sqrt(qty / adv)

    If any input is invalid (qty<=0, price<=0, adv<=0, side not buy/sell),
    returns 0.0 (no slippage estimated → treat as no-trade).
    """
    # ---- input validation ----
    if side not in ("buy", "sell"):
        return 0.0
    try:
        qty_f = float(qty)
        price_f = float(price)
        adv_f = float(adv)
    except (TypeError, ValueError):
        return 0.0
    if qty_f <= 0 or price_f <= 0 or adv_f <= 0:
        return 0.0

    # ---- liquidity class ----
    cls = get_liquidity_class(ticker)
    base_bps = _CLASS_PARAMS[cls]["base_bps"]
    impact_bps = _CLASS_PARAMS[cls]["impact_bps"]

    # ---- square-root impact ----
    participation = qty_f / adv_f  # fraction of daily volume consumed
    # Cap participation at 1.0 to avoid absurd slippage for outsized orders
    participation = min(participation, 1.0)
    impact = impact_bps * float(np.sqrt(participation))
    slippage_bps = base_bps + impact

    # ---- sign ----
    sign = 1.0 if side == "buy" else -1.0
    return sign * slippage_bps


def effective_price(
    qty: Union[int, float],
    price: float,
    adv: float,
    ticker: str,
    side: str,
) -> float:
    """Compute the effective execution price including slippage.

    effective_price = price * (1 + slippage_bps / 10000)

    For a BUY:  effective_price > price  (you pay more)
    For a SELL: effective_price < price  (you receive less)

    Examples:
        >>> effective_price(100_000, 270, 5_000_000, "SBER", "buy")
        270.0922...   # 270 * (1 + 3.4142/10000)
        >>> effective_price(100_000, 270, 5_000_000, "SBER", "sell")
        269.9077...
    """
    slip_bps = estimate_slippage(qty, price, adv, ticker, side)
    return price * (1.0 + slip_bps / 10000.0)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=== slippage_model self-test ===\n")

    # Tiny order: just base slippage
    s = estimate_slippage(100, 270, 5_000_000, "SBER", "buy")
    print(f"SBER buy 100 @ 270, ADV=5M:  slippage={s:.4f} bps "
          f"(expected ≈ 2.0447)")
    print(f"  effective_price={effective_price(100, 270, 5_000_000, 'SBER', 'buy'):.4f}")

    # Large order: square-root impact dominates
    s = estimate_slippage(100_000, 270, 5_000_000, "SBER", "buy")
    print(f"\nSBER buy 100k @ 270, ADV=5M: slippage={s:.4f} bps "
          f"(expected ≈ 3.4142)")
    print(f"  effective_price={effective_price(100_000, 270, 5_000_000, 'SBER', 'buy'):.4f}")

    # Sell side (negative)
    s = estimate_slippage(100_000, 270, 5_000_000, "SBER", "sell")
    print(f"\nSBER sell 100k @ 270:       slippage={s:.4f} bps "
          f"(expected ≈ -3.4142)")

    # Different liquidity classes (same participation)
    for ticker, expected_base in [("SBER", 2.0), ("GMKN", 3.0), ("MGNT", 5.0)]:
        adv = 1_000_000
        qty = 100_000  # 10% participation → impact = 10 * sqrt(0.1) ≈ 3.162
        s = estimate_slippage(qty, 100, adv, ticker, "buy")
        print(f"\n{ticker} buy {qty} @ 100, ADV={adv}: slippage={s:.4f} bps")

    # Edge cases — invalid inputs return 0
    print("\nEdge cases (expect 0.0 for all):")
    for args in [
        (0, 270, 5e6, "SBER", "buy"),       # qty=0
        (100, 0, 5e6, "SBER", "buy"),       # price=0
        (100, 270, 0, "SBER", "buy"),       # adv=0
        (100, 270, 5e6, "SBER", "hold"),    # bad side
        (100, 270, 5e6, "UNKNOWN", "buy"),  # unknown ticker → low class (NOT 0)
    ]:
        print(f"  {args}: {estimate_slippage(*args):.4f}")
