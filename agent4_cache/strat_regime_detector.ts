$ cat /opt/ai-trader/src/strategies/regime_detector.ts
--- rc=0 ---
/**
 * regime_detector.ts — Pure-TypeScript 12-regime detector.
 *
 * Faithful port of Python `compute_regime_v2()` from
 *   /root/ai-trader-evolution/ml/meta_labeler_v2.py
 * using indicator formulas from
 *   /root/ai-trader-evolution/fast_mc/fast_backtest_v2.py :: precompute_indicators
 *
 * 12 regimes (index → name):
 *    0 STRONG_TREND_UP        ADX>30  + SMA50>SMA20>SMA14
 *    1 MILD_TREND_UP          ADX 20-30 + ascending SMAs
 *    2 RANGE_TIGHT            ADX<15  + ATR <= median(ATR,100)
 *    3 RANGE_WIDE             ADX<15  + ATR  > median(ATR,100)
 *    4 MILD_TREND_DOWN        ADX 20-30 + descending SMAs
 *    5 STRONG_TREND_DOWN      ADX>30  + SMA50<SMA20<SMA14
 *    6 CRASH                  ret_30 < -1.5% (-1.5% in 30 bars)
 *    7 OVERSOLD_BOUNCE        RSI<25  AND close > SMA5
 *    8 OVERBOUGHT_REVERSAL    RSI>75  AND close < SMA5
 *    9 BREAKOUT_UP            close > max(high[-20:-1])
 *   10 BREAKDOWN              close < min(low[-20:-1])
 *   11 HIGH_VOL_REGIME        ATR > 1.5 × median(ATR, 100)
 *
 * Effective priority (matching Python "last write wins" semantics —
 * each later block overrides whatever was set earlier):
 *    default RANGE_TIGHT
 *    → STRONG_TREND
 *    → MILD_TREND
 *    → RANGE_TIGHT/RANGE_WIDE (ADX<15 path)
 *    → CRASH
 *    → OVERSOLD_BOUNCE / OVERBOUGHT_REVERSAL
 *    → BREAKOUT_UP / BREAKDOWN
 *    → HIGH_VOL_REGIME (final override)
 *    → first 100 bars forced to RANGE_TIGHT (warmup)
 *
 * All indicators are CAUSAL (no future data):
 *  - SMA: cumsum-based, partial mean in warmup
 *  - RSI(14): Wilder smoothing (alpha = 1/14)
 *  - ATR(14): Wilder smoothing on True Range
 *  - ADX(14): proper Wilder formula (DI+/DI- → DX → ADX)
 *  - Donchian(20): excludes current bar (max/min of last 20 bars)
 *  - ATR median(100): rolling median of last 100 ATR values
 *  - ret_30: (close - close[-30]) / close[-30]
 *
 * The detector is intended to run on every tick of the latest N candles
 * (typically 200-500).  It returns the regime at the LAST bar only.
 *
 * No runtime dependencies — pure TypeScript.  Drop-in for the trader
 * server at /opt/ai-trader/src/strategies/regime_detector.ts.
 */

// ─── Public types ─────────────────────────────────────────────────────────────

export interface Candle {
  time: number;     // unix seconds (or any monotonic unit)
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface RegimeResult {
  regime: number;        // 0..11
  regimeName: string;    // STRONG_TREND_UP, ..., HIGH_VOL_REGIME
  confidence: number;    // 0..1, 0 during warmup
  adx: number;           // ADX(14) at last bar
  rsi: number;           // RSI(14) at last bar
}

export const REGIME_NAMES = [
  "STRONG_TREND_UP",
  "MILD_TREND_UP",
  "RANGE_TIGHT",
  "RANGE_WIDE",
  "MILD_TREND_DOWN",
  "STRONG_TREND_DOWN",
  "CRASH",
  "OVERSOLD_BOUNCE",
  "OVERBOUGHT_REVERSAL",
  "BREAKOUT_UP",
  "BREAKDOWN",
  "HIGH_VOL_REGIME",
] as const;

// Regime enum values (mirrors the indices above)
export const enum Regime {
  STRONG_TREND_UP       = 0,
  MILD_TREND_UP         = 1,
  RANGE_TIGHT           = 2,
  RANGE_WIDE            = 3,
  MILD_TREND_DOWN       = 4,
  STRONG_TREND_DOWN     = 5,
  CRASH                 = 6,
  OVERSOLD_BOUNCE       = 7,
  OVERBOUGHT_REVERSAL   = 8,
  BREAKOUT_UP           = 9,
  BREAKDOWN             = 10,
  HIGH_VOL_REGIME       = 11,
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

const EPS = 1e-10;

function clamp(x: number, lo: number, hi: number): number {
  if (Number.isNaN(x)) return lo;
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

/**
 * Causal rolling mean via cumsum.
 *   result[i < w]  = sum(arr[0..i]) / (i+1)     (partial mean — warmup)
 *   result[i >= w] = (cumsum[i] - cumsum[i-w]) / w
 *
 * Matches `rolling_mean()` in precompute_indicators.
 */
function rollingMean(arr: number[], w: number): number[] {
  const n = arr.length;
  const result = new Array<number>(n);
  if (n === 0) return result;
  const cumsum = new Array<number>(n);
  let s = 0;
  for (let i = 0; i < n; i++) {
    s += arr[i];
    cumsum[i] = s;
  }
  for (let i = 0; i < n; i++) {
    if (i < w) {
      // partial mean: i+1 elements averaged (matches ret[:w]/arange(1, w+1))
      result[i] = cumsum[i] / (i + 1);
    } else {
      // full window: w elements averaged (matches (ret[i] - ret[i-w]) / w)
      result[i] = (cumsum[i] - cumsum[i - w]) / w;
    }
  }
  return result;
}

/**
 * Wilder smoothing — EMA with alpha = 1/period.
 *   result[0] = arr[0]
 *   result[i] = alpha*arr[i] + (1-alpha)*result[i-1]
 *
 * Matches `wilder_smooth()` in precompute_indicators (used for RSI, ATR, ADX).
 */
function wilderSmooth(arr: number[], period: number): number[] {
  const n = arr.length;
  const result = new Array<number>(n);
  if (n === 0) return result;
  const alpha = 1.0 / period;
  result[0] = arr[0];
  for (let i = 1; i < n; i++) {
    result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1];
  }
  return result;
}

/**
 * True Range array.
 *   prev_close[0]   = closes[0]
 *   prev_close[i>0]  = closes[i-1]
 *   tr[i] = max(high - low, |high - prev_close|, |low - prev_close|)
 */
function trueRange(highs: number[], lows: number[], closes: number[]): number[] {
  const n = closes.length;
  const tr = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const prevClose = i === 0 ? closes[0] : closes[i - 1];
    const a = highs[i] - lows[i];
    const b = Math.abs(highs[i] - prevClose);
    const c = Math.abs(lows[i] - prevClose);
    tr[i] = Math.max(a, Math.max(b, c));
  }
  return tr;
}

/**
 * Wilder RSI(14).
 *   deltas = diff(closes, prepend=closes[0])   → deltas[0]=0
 *   gains  = max(deltas, 0), losses = max(-deltas, 0)
 *   avg_gain = wilder_smooth(gains, period)
 *   avg_loss = wilder_smooth(losses, period)
 *   rsi = 100 - 100/(1 + avg_gain/(avg_loss+EPS))
 */
function computeRSI(closes: number[], period = 14): number[] {
  const n = closes.length;
  const deltas = new Array<number>(n);
  if (n > 0) deltas[0] = 0;
  for (let i = 1; i < n; i++) {
    deltas[i] = closes[i] - closes[i - 1];
  }
  const gains = new Array<number>(n);
  const losses = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const d = deltas[i];
    gains[i] = d > 0 ? d : 0;
    losses[i] = d < 0 ? -d : 0;
  }
  const avgGain = wilderSmooth(gains, period);
  const avgLoss = wilderSmooth(losses, period);
  const rsi = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const rs = avgGain[i] / (avgLoss[i] + EPS);
    rsi[i] = 100 - 100 / (1 + rs);
  }
  return rsi;
}

/**
 * Wilder ADX(14). Requires n >= 28 (per Python guard).
 *
 *   up_move   = high - prev_high        (prev_high[0] = high[0], causal)
 *   down_move = prev_low - low          (prev_low[0]  = low[0],  causal)
 *   +DM = up_move   if up_move   > down_move AND up_move   > 0 else 0
 *   -DM = down_move if down_move > up_move   AND down_move > 0 else 0
 *   atr_adx  = wilder_smooth(TR, 14)
 *   +DI = 100 * wilder_smooth(+DM, 14) / (atr_adx + EPS)
 *   -DI = 100 * wilder_smooth(-DM, 14) / (atr_adx + EPS)
 *   DX  = 100 * |+DI - -DI| / (+DI + -DI + EPS)
 *   ADX = wilder_smooth(DX, 14)
 *
 * NOTE: Python uses np.roll (wraps last→first), but the first bar is in
 * warmup anyway. We use the causal prev=arr[0] convention which matches
 * Python's behaviour for i >= 1 exactly.
 */
function computeADX(highs: number[], lows: number[], tr: number[]): number[] {
  const n = highs.length;
  const adx = new Array<number>(n).fill(0);
  if (n < 28) return adx;

  const plusDM = new Array<number>(n);
  const minusDM = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const prevHigh = i === 0 ? highs[0] : highs[i - 1];
    const prevLow  = i === 0 ? lows[0]  : lows[i - 1];
    const upMove   = highs[i] - prevHigh;
    const downMove = prevLow - lows[i];
    plusDM[i]  = (upMove > downMove && upMove > 0) ? upMove : 0;
    minusDM[i] = (downMove > upMove && downMove > 0) ? downMove : 0;
  }

  const atrAdx       = wilderSmooth(tr, 14);
  const plusDMSmooth = wilderSmooth(plusDM, 14);
  const minusDMSmooth = wilderSmooth(minusDM, 14);

  const dx = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const plusDI  = 100 * plusDMSmooth[i]  / (atrAdx[i] + EPS);
    const minusDI = 100 * minusDMSmooth[i] / (atrAdx[i] + EPS);
    dx[i] = 100 * Math.abs(plusDI - minusDI) / (plusDI + minusDI + EPS);
  }

  return wilderSmooth(dx, 14);
}

/**
 * Donchian channel (excludes current bar — max/min over last `w` bars).
 *   donchian_high[i] = max(highs[i-w .. i-1])   for i >= w, else 0
 *   donchian_low[i]  = min(lows[i-w  .. i-1])   for i >= w, else 0
 *
 * The "exclude current bar" semantics are critical — including i would
 * make breakout/close > donchian trivially true.
 */
function computeDonchian(highs: number[], lows: number[], w = 20): { high: number[]; low: number[] } {
  const n = highs.length;
  const dh = new Array<number>(n).fill(0);
  const dl = new Array<number>(n).fill(0);
  for (let i = w; i < n; i++) {
    let mx = -Infinity;
    let mn = Infinity;
    for (let j = i - w; j < i; j++) {
      if (highs[j] > mx) mx = highs[j];
      if (lows[j]  < mn) mn = lows[j];
    }
    dh[i] = mx;
    dl[i] = mn;
  }
  return { high: dh, low: dl };
}

/**
 * Rolling median over window `w` (default 100).
 *   result[i < w] = 0
 *   result[i >= w] = median(arr[i-w .. i-1])
 *
 * For even window sizes, returns the mean of the two middle elements
 * (matches numpy.median behaviour).
 */
function computeATRMedian(atr: number[], w = 100): number[] {
  const n = atr.length;
  const med = new Array<number>(n).fill(0);
  if (w % 2 !== 0) {
    // Odd window — single middle element.
    const mid = (w - 1) >> 1;
    for (let i = w; i < n; i++) {
      const win = atr.slice(i - w, i).sort((a, b) => a - b);
      med[i] = win[mid];
    }
  } else {
    // Even window — mean of two middle elements.
    const midA = (w >> 1) - 1;
    const midB = w >> 1;
    for (let i = w; i < n; i++) {
      const win = atr.slice(i - w, i).sort((a, b) => a - b);
      med[i] = (win[midA] + win[midB]) / 2;
    }
  }
  return med;
}

/**
 * 30-bar return (causal):
 *   ret_30[i < 30] = 0
 *   ret_30[i]      = (close[i] - close[i-30]) / (close[i-30] + EPS)
 */
function computeRet30(closes: number[]): number[] {
  const n = closes.length;
  const r = new Array<number>(n).fill(0);
  for (let i = 30; i < n; i++) {
    const prev = closes[i - 30];
    r[i] = (closes[i] - prev) / (prev + EPS);
  }
  return r;
}

// ─── Indicators bundle ────────────────────────────────────────────────────────

export interface Indicators {
  sma5:        number[];
  sma14:       number[];
  sma20:       number[];
  sma50:       number[];
  rsi14:       number[];
  atr:         number[];
  adx:         number[];
  ret30:       number[];
  donchianHigh: number[];
  donchianLow:  number[];
  atrMedian:   number[];
}

/**
 * Compute every indicator needed by detectRegime.
 * Exposed for callers that want to inspect / log the intermediate values.
 */
export function computeIndicators(candles: Candle[]): Indicators {
  const n = candles.length;
  const closes = new Array<number>(n);
  const highs  = new Array<number>(n);
  const lows   = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    closes[i] = candles[i].close;
    highs[i]  = candles[i].high;
    lows[i]   = candles[i].low;
  }

  const sma5  = rollingMean(closes, 5);
  const sma14 = rollingMean(closes, 14);
  const sma20 = rollingMean(closes, 20);
  let sma50: number[];
  if (n >= 50) {
    sma50 = rollingMean(closes, 50);
  } else {
    // Matches Python fallback: np.full(n, closes.mean()) when n < 50
    const mean = closes.length > 0 ? closes.reduce((a, b) => a + b, 0) / closes.length : 0;
    sma50 = new Array<number>(n).fill(mean);
  }

  const rsi14 = computeRSI(closes, 14);

  const tr  = trueRange(highs, lows, closes);
  const atr = wilderSmooth(tr, 14);
  const adx = computeADX(highs, lows, tr);

  const ret30       = computeRet30(closes);
  const donchian    = computeDonchian(highs, lows, 20);
  const atrMedian   = computeATRMedian(atr, 100);

  return {
    sma5, sma14, sma20, sma50,
    rsi14,
    atr, adx,
    ret30,
    donchianHigh: donchian.high,
    donchianLow:  donchian.low,
    atrMedian,
  };
}

// ─── Main entry point ─────────────────────────────────────────────────────────

/**
 * Detect market regime at the LAST bar of `candles`.
 *
 * Implements the exact same priority order as Python
 * `compute_regime_v2()`: each later rule overrides the previous one
 * where its conditions hold.  HIGH_VOL_REGIME is the final override;
 * the first 100 bars of an array are forced to RANGE_TIGHT (warmup).
 */
export function detectRegime(candles: Candle[]): RegimeResult {
  const n = candles.length;

  // Warmup: first 100 bars → RANGE_TIGHT with confidence 0.
  // (Python: `regime[:100] = RANGE_TIGHT` covers indices 0..99 inclusive,
  //  i.e. the first 100 bars.  Since detectRegime returns the LAST bar's
  //  regime, the last bar's index n-1 must be >= 100 to do real detection.
  //  So warmup applies when n <= 100.)
  if (n <= 100) {
    return {
      regime: Regime.RANGE_TIGHT,
      regimeName: REGIME_NAMES[Regime.RANGE_TIGHT],
      confidence: 0,
      adx: 0,
      rsi: 50,
    };
  }

  const ind = computeIndicators(candles);
  const closes = candles.map(c => c.close);
  const i = n - 1;  // index of the bar we classify

  const sma5  = ind.sma5[i];
  const sma14 = ind.sma14[i];
  const sma20 = ind.sma20[i];
  const sma50 = ind.sma50[i];
  const rsi   = ind.rsi14[i];
  const atr   = ind.atr[i];
  const adx   = ind.adx[i];
  const ret30 = ind.ret30[i];
  const dh    = ind.donchianHigh[i];
  const dl    = ind.donchianLow[i];
  const amed  = ind.atrMedian[i];
  const close = closes[i];

  // Trend alignment (with Python's 0.999 / 1.001 floating-point tolerance).
  // up_aligned:   sma50 > sma20*0.999  AND  sma20 > sma14*0.999
  // down_aligned: sma50 < sma20*1.001  AND  sma20 < sma14*1.001
  const upAligned   = (sma50 > sma20 * 0.999) && (sma20 > sma14 * 0.999);
  const downAligned = (sma50 < sma20 * 1.001) && (sma20 < sma14 * 1.001);

  // Default = RANGE_TIGHT (most neutral). Python initialises regime[:]
  // to RANGE_TIGHT before any rule is applied.
  let regime: Regime = Regime.RANGE_TIGHT;
  let confidence = 0;

  // ── 4. STRONG_TREND_UP / STRONG_TREND_DOWN (ADX > 30) ──────────────────────
  if (upAligned && adx > 30) {
    regime = Regime.STRONG_TREND_UP;
    confidence = clamp((adx - 30) / 30, 0.5, 1);
  } else if (downAligned && adx > 30) {
    regime = Regime.STRONG_TREND_DOWN;
    confidence = clamp((adx - 30) / 30, 0.5, 1);
  }

  // ── 5. MILD_TREND_UP / MILD_TREND_DOWN (ADX 20-30) ─────────────────────────
  if (upAligned && adx > 20 && adx <= 30) {
    regime = Regime.MILD_TREND_UP;
    confidence = clamp((adx - 20) / 10, 0.3, 1);
  } else if (downAligned && adx > 20 && adx <= 30) {
    regime = Regime.MILD_TREND_DOWN;
    confidence = clamp((adx - 20) / 10, 0.3, 1);
  }

  // ── 6. RANGE_TIGHT / RANGE_WIDE (ADX < 15, gated by atr_median > 0) ────────
  if (adx < 15 && amed > 0) {
    if (atr > amed) {
      regime = Regime.RANGE_WIDE;
      // confidence = how strong the "range" signal is, with a small boost
      // for how much ATR exceeds the median.
      confidence = clamp((15 - adx) / 15 * 0.7 + (atr / amed - 1) * 0.3, 0.3, 1);
    } else {
      regime = Regime.RANGE_TIGHT;
      confidence = clamp((15 - adx) / 15, 0.3, 1);
    }
  }

  // ── 7. CRASH (ret_30 < -1.5%) ──────────────────────────────────────────────
  if (ret30 < -0.015) {
    regime = Regime.CRASH;
    // At threshold (-1.5%) → 0.5; at -3% → 1.0
    confidence = clamp(Math.abs(ret30) / 0.03, 0.5, 1);
  }

  // ── 8. OVERSOLD_BOUNCE (RSI < 25 AND close > SMA5) ──────────────────────────
  if (rsi < 25 && close > sma5) {
    regime = Regime.OVERSOLD_BOUNCE;
    confidence = clamp((25 - rsi) / 25, 0.5, 1);
  }

  // ── 9. OVERBOUGHT_REVERSAL (RSI > 75 AND close < SMA5) ─────────────────────
  if (rsi > 75 && close < sma5) {
    regime = Regime.OVERBOUGHT_REVERSAL;
    confidence = clamp((rsi - 75) / 25, 0.5, 1);
  }

  // ── 10. BREAKOUT_UP (close > donchian_high, donchian_high > 0) ────────────
  if (dh > 0 && close > dh) {
    regime = Regime.BREAKOUT_UP;
    // Distance above the channel, normalised by ATR.
    confidence = clamp((close - dh) / (atr + EPS), 0.5, 1);
  }

  // ── 11. BREAKDOWN (close < donchian_low, donchian_low > 0) ─────────────────
  if (dl > 0 && close < dl) {
    regime = Regime.BREAKDOWN;
    confidence = clamp((dl - close) / (atr + EPS), 0.5, 1);
  }

  // ── 12. HIGH_VOL_REGIME (ATR > 1.5 × median, median > 0) — FINAL OVERRIDE ──
  if (amed > 0 && atr > 1.5 * amed) {
    regime = Regime.HIGH_VOL_REGIME;
    confidence = clamp((atr / amed - 1.5) / 1.5, 0.5, 1);
  }

  return {
    regime,
    regimeName: REGIME_NAMES[regime],
    confidence,
    adx,
    rsi,
  };
}

// ─── Self-test (commented out — uncomment to run with `npx tsx`) ───────────────
//
// Generates 200 synthetic candles in a clear uptrend (slow drift + noise)
// and verifies that detectRegime returns a valid regime 0..11 with a
// confidence in [0, 1].
//
// ```bash
//   npx tsx regime_detector.ts
// ```
//
/*
function selfTest(): void {
  // Build 200 candles: gentle uptrend with noise.
  const N = 200;
  const candles: Candle[] = [];
  let price = 100;
  let t = Math.floor(Date.now() / 1000);
  for (let i = 0; i < N; i++) {
    // Strong upward drift on the last 50 bars → should trigger a breakout
    // or trend regime.
    const drift = i > 150 ? 0.45 : 0.05;
    const noise = (Math.random() - 0.5) * 0.6;
    const open  = price;
    const close = price + drift + noise;
    const high  = Math.max(open, close) + Math.random() * 0.3;
    const low   = Math.min(open, close) - Math.random() * 0.3;
    const vol   = 1000 + Math.random() * 500;
    candles.push({ time: t + i * 60, open, high, low, close, volume: vol });
    price = close;
  }

  const result = detectRegime(candles);
  console.log("regime      =", result.regime);
  console.log("regimeName  =", result.regimeName);
  console.log("confidence  =", result.confidence.toFixed(4));
  console.log("adx         =", result.adx.toFixed(4));
  console.log("rsi         =", result.rsi.toFixed(4));

  // Assertions
  if (result.regime < 0 || result.regime > 11) {
    throw new Error(`regime out of range: ${result.regime}`);
  }
  if (result.confidence < 0 || result.confidence > 1) {
    throw new Error(`confidence out of [0,1]: ${result.confidence}`);
  }
  if (REGIME_NAMES[result.regime] !== result.regimeName) {
    throw new Error(`regimeName mismatch: ${result.regimeName} vs idx ${result.regime}`);
  }
  console.log("OK — self-test passed");
}

if (require.main === module) {
  selfTest();
}
*/


