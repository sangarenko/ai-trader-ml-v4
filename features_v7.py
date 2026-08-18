#!/usr/bin/env python3
"""Feature pipeline v7 — v4 (22 features) + 14 NEW features = 36 features total.

Adds VWAP / volume profile / order flow / macro / intraday-structure features
to the v4 pipeline. All features are STRICTLY CAUSAL — no look-ahead bias.

NEW features (14):

  Intraday volume profile (5):
      vwap          — VWAP for the trading day, normalized as vwap/close (~1.0)
      vwap_dev       — (close - vwap) / vwap  (signed deviation, ~0)
      poc            — Point of Control (price bin with highest session vol),
                       normalized as poc/close (~1.0)
      vah            — Volume Area High (upper bound of 70% value area),
                       normalized as vah/close (~1.0)
      val            — Volume Area Low (lower bound of 70% value area),
                       normalized as val/close (~1.0)
      All reset at MSK midnight. Causal: use only bars from start_of_day to i.

  Order flow (2):
      cum_delta     — cumulative (buy_vol - sell_vol) / cumulative volume (in [-1,1])
                      buy/sell vol estimated from each bar's (close-low)/(high-low)
      order_imbalance — per-bar (buy_vol - sell_vol) / (buy_vol + sell_vol) (in [-1,1])

  Macro (5):
      usdrub_ret    — USD/RUB daily return (forward-filled from latest completed day)
      brent_ret     — Brent crude oil daily return (merged BR-* futures contracts)
      imoex_ret     — IMOEX index daily return
      imoex_ret_1h  — IMOEX index 1-hour return
      cb_rate       — ЦБ РФ key rate / 100 (e.g. 0.21 for 21%)

  Intraday structure (2):
      intraday_session — 0=open(10-11), 1=mid(11-17), 2=close(17-23:50), /2 → [0,1]
      gap_overnight    — (today_open - prev_close) / prev_close (constant per day)

Macro data sources:
  - USD/RUB : MOEX ISS currency/selt/USD000UTSTOM (daily candles)
  - Brent   : MOEX ISS futures/forts/BR-* contracts merged by front-month
  - IMOEX   : MOEX ISS stock/index/IMOEX (daily + 1h candles)
  - CB rate : hardcoded history (effective dates 2022..2024)
  If any fetch fails, the corresponding feature returns 0.0 (no crash).

Signature:
    compute_features_v7(aligned, all_tickers_data=None, macro_data=None)
        -> (X: np.ndarray[n, 36], feature_names: list[str])
"""
import os
import json
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from features_v4 import compute_features_v4


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
USER_AGENT = "ai-trader-ml/1.0"
_MSK = timezone(timedelta(hours=3))


# ЦБ РФ key rate history (effective date, rate %).
# Source: cbr.ru press releases. Dates are the day the rate became effective
# (board meeting day). Used as a step function — rate at time t is the most
# recent rate whose effective_date <= t.
_CB_RATE_HISTORY = [
    (datetime(2022,  1,  1, tzinfo=_MSK),  8.5),
    (datetime(2022,  2, 28, tzinfo=_MSK), 20.0),  # emergency hike
    (datetime(2022,  4, 11, tzinfo=_MSK), 17.0),
    (datetime(2022,  5, 26, tzinfo=_MSK), 11.0),
    (datetime(2022,  6, 10, tzinfo=_MSK),  9.5),
    (datetime(2022,  7, 15, tzinfo=_MSK),  8.0),
    (datetime(2022,  9, 19, tzinfo=_MSK),  7.5),
    (datetime(2023,  8, 15, tzinfo=_MSK), 12.0),
    (datetime(2023,  9, 15, tzinfo=_MSK), 13.0),
    (datetime(2023, 10, 27, tzinfo=_MSK), 15.0),
    (datetime(2023, 12, 15, tzinfo=_MSK), 16.0),
    (datetime(2024,  7, 26, tzinfo=_MSK), 18.0),
    (datetime(2024,  9, 13, tzinfo=_MSK), 19.0),
    (datetime(2024, 10, 25, tzinfo=_MSK), 21.0),
]


# Candidate historical Brent futures contracts on MOEX FORTS.
# SECID format: BR + <month-code> + <year-digit>  (year-digit 4=2024, 5=2025, 6=2026, 7=2027)
# Month codes: F=Jan G=Feb H=Mar J=Apr K=May M=Jun N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec
# We list ~4 years × 12 months = 48 contracts; most return 0 candles but a few
# cover the requested period. Front-month merging picks the right contract per date.
_BR_CANDIDATE_SECIDS = []
for _y in ("4", "5", "6", "7"):
    for _mc in ("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"):
        _BR_CANDIDATE_SECIDS.append(f"BR{_mc}{_y}")


# --------------------------------------------------------------------------- #
# MOEX ISS fetchers for macro data
# --------------------------------------------------------------------------- #
def _fetch_moex_macro_candles(ticker: str, engine: str, market: str,
                                interval: int, from_date: str) -> List[Dict]:
    """Fetch candles from MOEX ISS for a macro ticker (USD/RUB, IMOEX, BR).

    Returns list of dicts: {"time": ts_ms, "open", "high", "low", "close", "volume"}.
    Timestamps are in MSK (MOEX returns MSK-local time strings).
    """
    all_candles: List[Dict] = []
    start = 0
    while True:
        url = (
            f"https://iss.moex.com/iss/engines/{engine}/markets/{market}/securities/{ticker}"
            f"/candles.json?interval={interval}&from={from_date}&start={start}"
            f"&iss.meta=off&iss.only=candles"
            f"&candles.columns=begin,end,open,high,low,close,volume"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())
            candles = data.get("candles", {}).get("data", [])
            if not candles:
                break
            for c in candles:
                try:
                    dt = datetime.strptime(c[0][:19], "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue
                dt = dt.replace(tzinfo=_MSK)
                ts_ms = int(dt.timestamp() * 1000)
                all_candles.append({
                    "time": ts_ms,
                    "open": float(c[2]) if c[2] is not None else 0.0,
                    "high": float(c[3]) if c[3] is not None else 0.0,
                    "low":  float(c[4]) if c[4] is not None else 0.0,
                    "close": float(c[5]) if c[5] is not None else 0.0,
                    "volume": int(c[6]) if c[6] else 0,
                })
            if len(candles) < 500:
                break
            start += 500
            time.sleep(0.05)  # be polite
        except Exception as e:
            print(f"  {ticker}: fetch error: {e}")
            break
    return all_candles


def _estimate_br_lastdel_ms(secid: str) -> int:
    """Estimate last delivery timestamp (ms) from a FORTS BR SECID like 'BRU6'.

    Expiry is roughly the 15th of the month AFTER the contract month
    (e.g. BRU6 = Sep 2026 contract → expiry mid-October 2026).
    Used only for front-month merging — exact date isn't critical.
    """
    if not secid.startswith("BR") or len(secid) < 4:
        return 0
    mc = secid[2]
    yr_digit = secid[3]
    month_codes = ("F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z")
    if mc not in month_codes or not yr_digit.isdigit():
        return 0
    month_idx = month_codes.index(mc)  # 0..11 → contract month (Jan..Dec)
    year = 2020 + int(yr_digit)        # 4 → 2024, 5 → 2025, 6 → 2026, 7 → 2027
    # Expiry ~15th of the month AFTER the contract month
    if month_idx == 11:  # Dec contract → Jan next year
        exp_year, exp_month = year + 1, 1
    else:
        exp_year, exp_month = year, month_idx + 2
    dt = datetime(exp_year, exp_month, 15, tzinfo=_MSK)
    return int(dt.timestamp() * 1000)


def _fetch_brent_candles(from_date: str) -> List[Dict]:
    """Fetch Brent futures history by merging multiple BR-* FORTS contracts.

    For each date we use the front-month contract = the one whose expiry is
    the closest (but still after) that date. This gives a continuous Brent
    price series. Returns list of {"time", "close"} dicts.
    """
    # For each candidate contract, fetch candles
    contract_candles: List[Tuple[int, List[Dict]]] = []  # (lastdel_ms, candles)
    for secid in _BR_CANDIDATE_SECIDS:
        try:
            candles = _fetch_moex_macro_candles(secid, "futures", "forts", 24, from_date)
            if not candles:
                continue
            lastdel_ms = _estimate_br_lastdel_ms(secid)
            contract_candles.append((lastdel_ms, candles))
        except Exception:
            continue

    if not contract_candles:
        return []

    # Merge: for each date, pick front-month = smallest lastdel > date
    by_date: Dict[int, Tuple[int, float]] = {}  # date_ms -> (lastdel_ms, close)
    for lastdel_ms, candles in contract_candles:
        for c in candles:
            t = c["time"]
            # Skip if contract already expired at time t (avoid stale data)
            if lastdel_ms > 0 and t >= lastdel_ms:
                continue
            if t not in by_date:
                by_date[t] = (lastdel_ms, c["close"])
            else:
                # Prefer the contract with smaller (but positive) lastdel — that's the front month
                prev_lastdel, _ = by_date[t]
                if lastdel_ms > 0 and (prev_lastdel <= 0 or lastdel_ms < prev_lastdel):
                    by_date[t] = (lastdel_ms, c["close"])

    sorted_times = sorted(by_date.keys())
    return [{"time": t, "close": by_date[t][1]} for t in sorted_times]


def fetch_macro_data(days: int = 180) -> Dict[str, np.ndarray]:
    """Fetch USD/RUB, Brent, IMOEX daily candles + CB rate history.

    Returns dict (any key may be missing on fetch failure):
        usdrub_time, usdrub_close      — USD/RUB daily candles
        brent_time, brent_close        — Brent (merged BR-* contracts) daily
        imoex_time, imoex_close         — IMOEX index daily candles
        imoex_1h_time, imoex_1h_close   — IMOEX index 1-hour candles
        cb_rate_history                 — np.array shape [N, 2]: (timestamp_ms, rate%)
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"macro_{days}d.npz")

    if os.path.exists(cache_path):
        age = time.time() - os.path.getmtime(cache_path)
        if age < 86400:
            print(f"  macro: cached (age {age/3600:.1f}h)")
            return dict(np.load(cache_path, allow_pickle=True))

    # +14 days safety margin so we have at least one prior close for return calc
    from_date = (datetime.now(timezone.utc) - timedelta(days=days + 14)).strftime("%Y-%m-%d")

    result: Dict[str, np.ndarray] = {}

    # USD/RUB
    try:
        candles = _fetch_moex_macro_candles("USD000UTSTOM", "currency", "selt", 24, from_date)
        if candles:
            result["usdrub_time"] = np.array([c["time"] for c in candles])
            result["usdrub_close"] = np.array([c["close"] for c in candles], dtype=float)
            print(f"  USD/RUB: {len(candles)} daily candles")
    except Exception as e:
        print(f"  USD/RUB fetch failed: {e}")

    # Brent
    try:
        candles = _fetch_brent_candles(from_date)
        if candles:
            result["brent_time"] = np.array([c["time"] for c in candles])
            result["brent_close"] = np.array([c["close"] for c in candles], dtype=float)
            print(f"  Brent: {len(candles)} daily candles")
    except Exception as e:
        print(f"  Brent fetch failed: {e}")

    # IMOEX daily
    try:
        candles = _fetch_moex_macro_candles("IMOEX", "stock", "index", 24, from_date)
        if candles:
            result["imoex_time"] = np.array([c["time"] for c in candles])
            result["imoex_close"] = np.array([c["close"] for c in candles], dtype=float)
            print(f"  IMOEX: {len(candles)} daily candles")
    except Exception as e:
        print(f"  IMOEX fetch failed: {e}")

    # IMOEX 1h
    try:
        candles = _fetch_moex_macro_candles("IMOEX", "stock", "index", 60, from_date)
        if candles:
            result["imoex_1h_time"] = np.array([c["time"] for c in candles])
            result["imoex_1h_close"] = np.array([c["close"] for c in candles], dtype=float)
            print(f"  IMOEX 1h: {len(candles)} hourly candles")
    except Exception as e:
        print(f"  IMOEX 1h fetch failed: {e}")

    # CB rate history (hardcoded — see _CB_RATE_HISTORY at top of file)
    cb_arr = np.array(
        [[int(dt.timestamp() * 1000), rate] for dt, rate in _CB_RATE_HISTORY],
        dtype=float,
    )
    result["cb_rate_history"] = cb_arr

    if result:
        try:
            np.savez(cache_path, **result)
        except Exception as e:
            print(f"  macro cache save failed: {e}")

    return result


# --------------------------------------------------------------------------- #
# Macro -> base-grid alignment helpers (causal)
# --------------------------------------------------------------------------- #
def _macro_ret(macro_time: Optional[np.ndarray],
                macro_close: Optional[np.ndarray],
                base_time: np.ndarray,
                interval_minutes: int) -> np.ndarray:
    """Forward-fill macro returns onto base grid (strictly causal).

    For each macro bar i, its close becomes available at macro_time[i] +
    interval_minutes (end of the macro bar). At base time t we use the
    most recent macro return whose close time is <= t.

    Returns: np.array shape [n_base], zeros if macro data missing.
    """
    n = len(base_time)
    if macro_time is None or macro_close is None or len(macro_close) < 2:
        return np.zeros(n)
    macro_time = np.asarray(macro_time, dtype=float)
    macro_close = np.asarray(macro_close, dtype=float)
    macro_ret = np.zeros(len(macro_close))
    macro_ret[1:] = (macro_close[1:] - macro_close[:-1]) / (macro_close[:-1] + 1e-10)
    # Close of macro bar i is at macro_time[i] + interval
    macro_close_time = macro_time + interval_minutes * 60 * 1000.0
    idx = np.searchsorted(macro_close_time, base_time, side="right") - 1
    out = np.zeros(n)
    valid = idx >= 0
    idx_safe = np.clip(idx, 0, len(macro_ret) - 1)
    out[valid] = macro_ret[idx_safe[valid]]
    return out


def _cb_rate_array(time5_ms: np.ndarray,
                    cb_history: Optional[np.ndarray]) -> np.ndarray:
    """Step-function CB rate at each base timestamp (in %, e.g. 21.0 for 21%)."""
    n = len(time5_ms)
    out = np.full(n, 8.5)  # default to pre-2022 rate
    if cb_history is None or len(cb_history) == 0:
        return out
    cb_history = np.asarray(cb_history, dtype=float)
    for cutoff_ms, rate in cb_history:
        cutoff_ms = float(cutoff_ms)
        out[time5_ms >= cutoff_ms] = rate
    return out


# --------------------------------------------------------------------------- #
# Day-boundary helpers (shared by VWAP / cum_delta / volume profile / gap)
# --------------------------------------------------------------------------- #
def _compute_day_boundaries(time5: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute day boundaries for MSK-midnight reset (causal).

    Args:
        time5: epoch-ms timestamps of the base (10-min) grid.

    Returns:
        day_start: np.array[int] of shape [n] — index of the FIRST bar of each
            bar's day (so day_start[i] is the index of the first bar of the
            same MSK-calendar-day as bar i).
        prev_day_last_bar: np.array[int] of shape [n] — index of the LAST bar of
            the previous day, or -1 if bar i is on the first day.
    """
    n = len(time5)
    ts_seconds = time5 / 1000.0
    msk_seconds = ts_seconds + 3 * 3600  # MSK = UTC+3
    day_id = (msk_seconds // 86400).astype(np.int64)

    # Indices where the day changes
    day_change = np.concatenate(([True], day_id[1:] != day_id[:-1]))
    day_start_indices = np.where(day_change)[0]

    # For each bar, find which day-start it belongs to (searchsorted)
    bar_idx = np.arange(n)
    day_pos = np.searchsorted(day_start_indices, bar_idx, side="right") - 1
    day_pos = np.clip(day_pos, 0, len(day_start_indices) - 1)
    day_start = day_start_indices[day_pos]

    prev_day_last_bar = day_start - 1  # -1 for first day → "no previous day"
    return day_start, prev_day_last_bar


# --------------------------------------------------------------------------- #
# VWAP — causal, resets at MSK midnight
# --------------------------------------------------------------------------- #
def _compute_vwap(close5: np.ndarray, high5: np.ndarray, low5: np.ndarray,
                   vol5: np.ndarray, day_start: np.ndarray,
                   prev_day_last_bar: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal VWAP that resets at MSK midnight.

    vwap[i] = sum(typical_price * vol from day_start[i] to i) /
              sum(vol from day_start[i] to i)

    Returns:
        vwap_raw      — raw VWAP price level (np.float)
        vwap_norm     — vwap_raw / close5 (~1.0, scale-invariant)
        vwap_dev      — (close5 - vwap_raw) / vwap_raw (signed deviation, ~0)
        cumsum_v_day  — cumulative volume within the day up to each bar
    """
    typical_price = (high5 + low5 + close5) / 3.0
    pv = typical_price * vol5
    cumsum_pv = np.cumsum(pv)
    cumsum_v = np.cumsum(vol5)

    prev_idx_safe = np.where(prev_day_last_bar >= 0, prev_day_last_bar, 0)
    mask_prev = prev_day_last_bar >= 0

    sub_pv = np.where(mask_prev, cumsum_pv[prev_idx_safe], 0.0)
    sub_v = np.where(mask_prev, cumsum_v[prev_idx_safe], 0.0)

    cumsum_pv_day = cumsum_pv - sub_pv
    cumsum_v_day = cumsum_v - sub_v

    vwap_raw = cumsum_pv_day / (cumsum_v_day + 1e-10)
    vwap_norm = vwap_raw / (close5 + 1e-10)
    vwap_dev = (close5 - vwap_raw) / (vwap_raw + 1e-10)

    return vwap_raw, vwap_norm, vwap_dev, cumsum_v_day


# --------------------------------------------------------------------------- #
# Volume profile — POC / VAH / VAL (causal, resets at MSK midnight)
# --------------------------------------------------------------------------- #
def _compute_volume_profile(close5: np.ndarray, high5: np.ndarray, low5: np.ndarray,
                              vol5: np.ndarray, day_start: np.ndarray,
                              prev_day_last_bar: np.ndarray,
                              n_bins: int = 50, value_pct: float = 0.70
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal volume profile (POC, VAH, VAL).

    Bins all base bars into ``n_bins`` price buckets spanning [low5.min(),
    high5.max()] across the entire dataset. For each bar we then take the
    cumulative volume per bin from start_of_day up to that bar (causal).

    POC = bin with the highest cumulative volume.
    VAH/VAL = expand outward from POC until cumulative volume >= 70% of total.

    Returns: (poc_raw, vah_raw, val_raw) — raw price levels (np.float).
    """
    n = len(close5)
    poc_raw = close5.copy()
    vah_raw = high5.copy()
    val_raw = low5.copy()

    price_lo = float(low5.min())
    price_hi = float(high5.max())
    if price_hi - price_lo < 1e-10:
        return poc_raw, vah_raw, val_raw

    bin_edges = np.linspace(price_lo, price_hi, n_bins + 1)
    bin_idx_lo = np.clip(np.searchsorted(bin_edges, low5, side="right") - 1, 0, n_bins - 1)
    bin_idx_hi = np.clip(np.searchsorted(bin_edges, high5, side="right") - 1, 0, n_bins - 1)

    # Build per-bar bin volume contributions: shape [n, n_bins]
    bin_vols = np.zeros((n, n_bins))
    for j in range(n):
        n_overlap = max(int(bin_idx_hi[j] - bin_idx_lo[j] + 1), 1)
        bin_vols[j, bin_idx_lo[j]:bin_idx_hi[j] + 1] = vol5[j] / n_overlap

    # Cumulative within day (subtract previous day's last cumulative)
    cum_vols = np.cumsum(bin_vols, axis=0)  # [n, n_bins]
    prev_idx_safe = np.where(prev_day_last_bar >= 0, prev_day_last_bar, 0)
    mask_prev_2d = (prev_day_last_bar >= 0)[:, None]
    sub_vols = np.where(mask_prev_2d, cum_vols[prev_idx_safe], 0.0)
    cum_vols_day = cum_vols - sub_vols  # [n, n_bins]

    total_vols = cum_vols_day.sum(axis=1)  # [n]
    poc_idx = np.argmax(cum_vols_day, axis=1)  # [n]
    poc_raw = (bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0

    target_vols = total_vols * value_pct

    vah_raw = np.full(n, bin_edges[-1], dtype=float)
    val_raw = np.full(n, bin_edges[0], dtype=float)

    # Expand from POC outward until 70% of total volume accumulated
    for j in range(n):
        if total_vols[j] < 1e-10:
            vah_raw[j] = high5[j]
            val_raw[j] = low5[j]
            continue
        p = poc_idx[j]
        cur = cum_vols_day[j, p]
        lo_idx = p
        hi_idx = p
        # Cap iterations to n_bins to prevent infinite loops
        for _ in range(n_bins):
            if cur >= target_vols[j]:
                break
            below = cum_vols_day[j, lo_idx - 1] if lo_idx > 0 else -1.0
            above = cum_vols_day[j, hi_idx + 1] if hi_idx < n_bins - 1 else -1.0
            if above >= below and above > 0:
                hi_idx += 1
                cur += above
            elif below > 0:
                lo_idx -= 1
                cur += below
            else:
                break
        vah_raw[j] = bin_edges[hi_idx + 1]
        val_raw[j] = bin_edges[lo_idx]

    return poc_raw, vah_raw, val_raw


# --------------------------------------------------------------------------- #
# Order flow — cumulative delta + order imbalance (causal)
# --------------------------------------------------------------------------- #
def _compute_order_flow(close5: np.ndarray, high5: np.ndarray, low5: np.ndarray,
                         vol5: np.ndarray, day_start: np.ndarray,
                         prev_day_last_bar: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Causal order flow proxy (no order book → estimate from candle shape).

    buy_vol  = vol * (close - low) / (high - low)     — fraction of bar's range
                                                            that closed in upper half
    sell_vol = vol * (high - close) / (high - low)

    cum_delta = cumsum(buy_vol - sell_vol) within day, normalized by cumulative
                volume (so result is in [-1, +1]).
    order_imbalance = per-bar (buy_vol - sell_vol) / (buy_vol + sell_vol) (in [-1,+1]).

    When high == low (no range), assume 50/50 buy/sell split.
    """
    rng = high5 - low5
    safe_rng = np.where(rng > 1e-10, rng, 1.0)

    buy_vol = np.where(rng > 1e-10, vol5 * (close5 - low5) / safe_rng, vol5 * 0.5)
    sell_vol = np.where(rng > 1e-10, vol5 * (high5 - close5) / safe_rng, vol5 * 0.5)
    delta = buy_vol - sell_vol

    cumsum_delta = np.cumsum(delta)
    prev_idx_safe = np.where(prev_day_last_bar >= 0, prev_day_last_bar, 0)
    mask_prev = prev_day_last_bar >= 0
    sub_delta = np.where(mask_prev, cumsum_delta[prev_idx_safe], 0.0)
    cum_delta_day = cumsum_delta - sub_delta

    # Cumulative volume within day (recompute for normalization)
    cumsum_v = np.cumsum(vol5)
    sub_v = np.where(mask_prev, cumsum_v[prev_idx_safe], 0.0)
    cumsum_v_day = cumsum_v - sub_v

    cum_delta_norm = cum_delta_day / (cumsum_v_day + 1e-10)
    order_imbalance = (buy_vol - sell_vol) / (buy_vol + sell_vol + 1e-10)

    return cum_delta_norm, order_imbalance


# --------------------------------------------------------------------------- #
# Intraday structure features
# --------------------------------------------------------------------------- #
def _compute_intraday_session(time5: np.ndarray) -> np.ndarray:
    """0=open(10-11 MSK), 1=mid(11-17), 2=close(17-23:50), normalized /2 → [0, 1]."""
    ts_seconds = time5 / 1000.0
    hour_msk = ((ts_seconds // 3600 + 3) % 24).astype(int)
    session = np.ones(len(time5), dtype=int)  # default mid
    session[(hour_msk >= 10) & (hour_msk < 11)] = 0  # open
    session[hour_msk >= 17] = 2                       # close
    return session.astype(float) / 2.0


def _compute_gap_overnight(open5: np.ndarray, close5: np.ndarray,
                            time5: np.ndarray) -> np.ndarray:
    """Causal gap = (today_open - prev_close) / prev_close. Constant per day.

    For the first day there's no previous close → gap = 0 (causal: don't use
    future data, and don't fabricate a gap).
    """
    ts_seconds = time5 / 1000.0
    msk_seconds = ts_seconds + 3 * 3600
    day_id = (msk_seconds // 86400).astype(np.int64)

    unique_days, day_inverse = np.unique(day_id, return_inverse=True)
    n_days = len(unique_days)

    day_first_idx = np.zeros(n_days, dtype=int)
    day_last_idx = np.zeros(n_days, dtype=int)
    for d in range(n_days):
        indices = np.where(day_inverse == d)[0]
        if len(indices) > 0:
            day_first_idx[d] = indices[0]
            day_last_idx[d] = indices[-1]

    day_open_price = open5[day_first_idx]    # [n_days]
    day_close_price = close5[day_last_idx]   # [n_days]

    gap_per_day = np.zeros(n_days)
    if n_days > 1:
        gap_per_day[1:] = (
            (day_open_price[1:] - day_close_price[:-1])
            / (day_close_price[:-1] + 1e-10)
        )

    return gap_per_day[day_inverse]


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def compute_features_v7(
    aligned: Dict[str, np.ndarray],
    all_tickers_data: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    macro_data: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute v7 features = v4 (22) + 14 new = 36 features.

    Args:
        aligned: dict with keys "5min_close", "5min_high", "5min_low",
            "5min_open", "5min_volume", "time", and forward-filled higher-TF
            arrays (same format as ml_data_pipeline.align_timeframes() output).
        all_tickers_data: optional {ticker: aligned_data} for cross-asset v4
            features (market_breadth, sber_gazp_corr).
        macro_data: optional dict with USD/RUB / Brent / IMOEX / CB rate data.
            If None, fetched from MOEX ISS (cached 24h in data_cache/macro_*.npz).

    Returns:
        X: np.ndarray shape (n_bars, 36)
        feature_names: list[str] sorted alphabetically
    """
    # =================================================================== #
    # v4 features (22) — reuse directly from features_v4.py
    # =================================================================== #
    X_v4, names_v4 = compute_features_v4(aligned, all_tickers_data)

    close5 = aligned["5min_close"]
    high5 = aligned["5min_high"]
    low5 = aligned["5min_low"]
    open5 = aligned["5min_open"]
    vol5 = aligned["5min_volume"].astype(float)
    time5 = aligned["time"]

    n = len(close5)

    # =================================================================== #
    # v7 NEW features (14)
    # =================================================================== #
    new_features: Dict[str, np.ndarray] = {}

    # ---- shared: day boundaries (MSK midnight reset) ----
    day_start, prev_day_last_bar = _compute_day_boundaries(time5)

    # ---- VWAP + vwap_dev ----
    vwap_raw, vwap_norm, vwap_dev, _ = _compute_vwap(
        close5, high5, low5, vol5, day_start, prev_day_last_bar
    )
    new_features["vwap"] = vwap_norm
    new_features["vwap_dev"] = vwap_dev

    # ---- Volume profile: POC, VAH, VAL ----
    poc_raw, vah_raw, val_raw = _compute_volume_profile(
        close5, high5, low5, vol5, day_start, prev_day_last_bar
    )
    new_features["poc"] = poc_raw / (close5 + 1e-10)
    new_features["vah"] = vah_raw / (close5 + 1e-10)
    new_features["val"] = val_raw / (close5 + 1e-10)

    # ---- Order flow: cum_delta, order_imbalance ----
    cum_delta, order_imbalance = _compute_order_flow(
        close5, high5, low5, vol5, day_start, prev_day_last_bar
    )
    new_features["cum_delta"] = cum_delta
    new_features["order_imbalance"] = order_imbalance

    # ---- Intraday structure ----
    new_features["intraday_session"] = _compute_intraday_session(time5)
    new_features["gap_overnight"] = _compute_gap_overnight(open5, close5, time5)

    # ---- Macro features ----
    if macro_data is None:
        try:
            macro_data = fetch_macro_data(days=60)
        except Exception as e:
            print(f"  macro fetch failed (using zeros): {e}")
            macro_data = {}

    # USD/RUB daily return
    new_features["usdrub_ret"] = _macro_ret(
        macro_data.get("usdrub_time"),
        macro_data.get("usdrub_close"),
        time5,
        interval_minutes=1440,
    )

    # Brent daily return (merged BR-* contracts)
    new_features["brent_ret"] = _macro_ret(
        macro_data.get("brent_time"),
        macro_data.get("brent_close"),
        time5,
        interval_minutes=1440,
    )

    # IMOEX daily return
    new_features["imoex_ret"] = _macro_ret(
        macro_data.get("imoex_time"),
        macro_data.get("imoex_close"),
        time5,
        interval_minutes=1440,
    )

    # IMOEX 1-hour return
    new_features["imoex_ret_1h"] = _macro_ret(
        macro_data.get("imoex_1h_time"),
        macro_data.get("imoex_1h_close"),
        time5,
        interval_minutes=60,
    )

    # CB rate (normalized to 0..0.25 range, e.g. 21% -> 0.21)
    cb_history = macro_data.get("cb_rate_history")
    cb_rate = _cb_rate_array(time5, cb_history)
    new_features["cb_rate"] = cb_rate / 100.0

    # =================================================================== #
    # Assemble all features (sorted alphabetically for determinism)
    # =================================================================== #
    all_features: Dict[str, np.ndarray] = {}

    # Copy v4 features (X_v4 columns) into the combined dict
    for i, name in enumerate(names_v4):
        all_features[name] = X_v4[:, i]

    # Add v7 new features
    for name, arr in new_features.items():
        all_features[name] = arr

    feature_names = sorted(all_features.keys())
    X = np.column_stack([all_features[name] for name in feature_names])

    # Sanitize: replace NaN/inf with 0, clip to sane range
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -10.0, 10.0)
    # Zero out first 50 bars (warm-up: SMA50 / vol_regime / volume profile needs history)
    if n >= 50:
        X[:50] = 0.0
    else:
        X[:] = 0.0

    return X, feature_names


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from ml_data_pipeline import download_multi_timeframe, align_timeframes

    print("=== features_v7 self-test ===")
    data = download_multi_timeframe("SBER", days=30)
    aligned = align_timeframes(data)
    X, names = compute_features_v7(aligned)
    print(f"\nFeatures: X.shape={X.shape} ({len(names)} features)")
    print("\nFeature names:")
    for nm in names:
        print(f"  {nm}")

    # Sanity check
    print(f"\nNaN count: {np.isnan(X).sum()}, Inf count: {np.isinf(X).sum()}")
    print(f"\nMin/max per column (skip warm-up):")
    for i, nm in enumerate(names):
        col = X[50:, i]
        print(f"  {nm:25s}  min={col.min():.4f}  max={col.max():.4f}  mean={col.mean():.4f}")

    # Verify v4 features are still present
    from features_v4 import compute_features_v4
    X_v4, names_v4 = compute_features_v4(aligned)
    v4_set = set(names_v4)
    v7_set = set(names)
    new_only = v7_set - v4_set
    print(f"\nv4 features: {len(names_v4)}")
    print(f"v7 total features: {len(names)}")
    print(f"NEW v7 features ({len(new_only)}): {sorted(new_only)}")
