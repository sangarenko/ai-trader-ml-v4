// ─── Sniper Strategies ──────────────────────────────────────────────────────
// 5 patient, precise trading strategies that DON'T spam trades.
// Based on research: "overtrading is the silent account killer",
// "confluence — combine 2-3 indicators from different categories",
// "if profitable on trades 1-3, max 3 trades per day".
//
// Each strategy waits for STRONG confirmation signals before acting.
// Target: 1-2 trades per minute maximum, not 100-500 like before.
//
// Strategies:
//   SNIPER_TREND       — trend confirmation (SMA + RSI + 3-candle streak)
//   CONFLUENCE_RSI_MACD — RSI + MACD + SMA confluence (3 indicators)
//   CONFLUENCE_BB_VOL  — Bollinger + volume spike + momentum
//   SNIPER_MOMENTUM    — strong momentum (>2%) + RSI confirmation
//   PATIENT_SWING      — 3 confirming signals required, very rare trades
// ──────────────────────────────────────────────────────────────────────────────

import type { Candle } from './rl-agents'

export interface SniperStrategy {
  name: string
  description: string
  // Returns action: 0=hold, 1=buy long, 2=sell short, 3=close position
  // Only acts when ALL confirmation conditions are met
  decide(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number): number
}

// Helper: compute indicators from candle array at given index
function indicators(candles: Candle[], idx: number) {
  if (idx < 14) return null
  const recent = candles.slice(idx - 14, idx + 1)
  const closes = recent.map(c => c.close)
  const volumes = recent.map(c => c.volume)
  const sma5 = closes.slice(-5).reduce((a, b) => a + b, 0) / 5
  const sma14 = closes.reduce((a, b) => a + b, 0) / closes.length
  let gains = 0, losses = 0
  for (let i = 1; i < recent.length; i++) {
    const change = closes[i] - closes[i - 1]
    if (change > 0) gains += change
    else losses -= change
  }
  const rsi = losses === 0 ? 100 : 100 - 100 / (1 + gains / losses)
  const stdDev = Math.sqrt(closes.reduce((s, c) => s + Math.pow(c - sma14, 2), 0) / closes.length)
  const upper = sma14 + 2 * stdDev
  const lower = sma14 - 2 * stdDev
  const momentum = (closes[closes.length - 1] - closes[0]) / closes[0]
  const volRatio = volumes[volumes.length - 1] / (volumes.reduce((a, b) => a + b, 0) / volumes.length)
  const cur = closes[closes.length - 1]
  // MACD: EMA12 - EMA26 (simplified as SMA12 - SMA26)
  const sma12 = closes.slice(-12).reduce((a, b) => a + b, 0) / Math.min(12, closes.length)
  const sma26 = closes.length >= 26 ? closes.slice(-26).reduce((a, b) => a + b, 0) / 26 : sma14
  const macd = sma12 - sma26
  // Check 3-candle streak (all up or all down)
  const last3 = closes.slice(-3)
  const allUp = last3[0] < last3[1] && last3[1] < last3[2]
  const allDown = last3[0] > last3[1] && last3[1] > last3[2]
  return { sma5, sma14, rsi, upper, lower, momentum, volRatio, cur, macd, allUp, allDown, stdDev }
}

// ─── 1. SNIPER_TREND ─────────────────────────────────────────────────────────
// Waits for: SMA5 > SMA14 (trend up) AND RSI in 40-65 zone (not overbought)
// AND 3 consecutive up candles (momentum confirmation)
// Only buys when ALL 3 conditions met. Very selective.

// ─── 2. CONFLUENCE_RSI_MACD ──────────────────────────────────────────────────
// Requires 3 indicators to agree: RSI + MACD + SMA crossover
// Based on "Multi-Indicator Confluence Momentum Trading Strategy"

// ─── 3. CONFLUENCE_BB_VOL ────────────────────────────────────────────────────
// Requires: price at Bollinger band + volume spike + momentum confirmation
// Based on "Bollinger bounce with volume confirmation"

// ─── 4. SNIPER_MOMENTUM ──────────────────────────────────────────────────────
// Waits for STRONG momentum (>2%) + RSI confirmation + volume
// Very selective — only acts on significant moves

// ─── 5. PATIENT_SWING ────────────────────────────────────────────────────────
// Most conservative — requires 4 conditions to all align:
// 1. SMA crossover  2. RSI in zone  3. MACD confirms  4. Volume spike
// Very rare trades, highest conviction

// ════════════════════════════════════════════════════════════════════════════
// SNIPER TREND VARIATIONS — SniperTrend shows profit, make different versions
// ════════════════════════════════════════════════════════════════════════════

// ─── V2: SHORT FOCUS — primarily shorts, holds longer ───────────────────────
// Like SniperTrend but biased towards shorts, longer holds, wider RSI zone
export class SniperTrendV2ShortStrategy implements SniperStrategy {
  name = 'SniperTrendV2'
  description = 'SniperTrend V2: фокус на шорты, дольше держит, RSI 30-55'

  decide(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number): number {
    const ind = indicators(candles, idx)
    if (!ind) return 0
    if (hasPosition) {
      // Hold longer — only close on strong reversal (SMA cross + RSI extreme)
      if (ind.sma5 > ind.sma14 * 1.003 && ind.rsi > 65) return 3
      return 0
    }
    // SHORT: easier entry — RSI 30-55, only 2 down candles needed
    if (ind.sma5 < ind.sma14 * 0.999 && ind.rsi > 30 && ind.rsi < 55 && ind.allDown) {
      return 2
    }
    // LONG: stricter — only when very oversold
    if (ind.sma5 > ind.sma14 * 1.002 && ind.rsi > 25 && ind.rsi < 40 && ind.allUp) {
      return 1
    }
    return 0
  }
}

// ─── V3: AGGRESSIVE — more trades, lower threshold ──────────────────────────
// Trades more often than V1, catches smaller trends

// ─── V4: MACD CONFIRM — SniperTrend + MACD confirmation ─────────────────────
// Adds MACD as 4th confirmation — higher quality entries

// ─── V5: MEAN REVERSION — RSI rebuilt from scratch ──────────────────────────
// Replaces RSI-Revert: proper RSI mean reversion with Bollinger + trend filter
export class RSIReversionV2Strategy implements SniperStrategy {
  name = 'RSI-ReversionV2'
  description = 'RSI V2: RSI<25 + lower BB + volume, RSI>75 + upper BB = short'

  decide(candles: Candle[], idx: number, hasPosition: boolean, _stepsHeld?: number): number {
    const ind = indicators(candles, idx)
    if (!ind) return 0
    if (hasPosition) {
      // Close when RSI returns to neutral (45-55)
      if (ind.rsi > 45 && ind.rsi < 55) return 3
      return 0
    }
    // BUY: RSI oversold (<25) + price near lower Bollinger + volume
    if (ind.rsi < 25 && ind.cur < ind.lower * 1.005 && ind.volRatio > 1.1) {
      return 1
    }
    // SHORT: RSI overbought (>75) + price near upper Bollinger + volume
    if (ind.rsi > 75 && ind.cur > ind.upper * 0.995 && ind.volRatio > 1.1) {
      return 2
    }
    return 0
  }
}

// ─── V6: BB BOUNCE V2 — rebuilt ConfluenceBV ────────────────────────────────
// Better Bollinger bounce: requires 2 confirmations (BB + RSI extreme)

// ─── V5: TRAINABLE V2 ────────────────────────────────────────────────────────
// V2's logic is optimal (backtest proven). V5 keeps it EXACTLY but exposes
// parameters that can be tuned via evolutionary training (12 hours overnight).
// Trained params loaded from /opt/ai-trader/scripts/sniper-v5-params.json
export class SniperTrendV5Strategy implements SniperStrategy {
  name = 'SniperTrendV5'
  description = 'SniperTrend V5: V2 logic + trainable params (RSI/SMA/hold), evolution-optimized'

  // Trainable parameters (defaults = V2 values, will be overwritten by training)
  private params = {
    short_rsi_min: 30, short_rsi_max: 55,
    long_rsi_min: 25, long_rsi_max: 40,
    close_rsi_short: 65, close_rsi_long: 35,
    sma_short: 0.999, sma_long: 1.002,
    close_sma_short: 1.003, close_sma_long: 0.997,
    min_hold_steps: 3,
  }

  constructor() {
    this.loadParams()
  }

  private loadParams() {
    try {
      const fs = require('fs')
      const path = '/opt/ai-trader/scripts/sniper-v5-params.json'
      if (fs.existsSync(path)) {
        const data = JSON.parse(fs.readFileSync(path, 'utf-8'))
        this.params = { ...this.params, ...data }
        console.log(`[SniperTrendV5] Loaded trained params: min_hold=${this.params.min_hold_steps}, short_rsi=${this.params.short_rsi_min}-${this.params.short_rsi_max}`)
      }
    } catch { /* use defaults */ }
  }

  decide(candles: Candle[], idx: number, hasPosition: boolean, stepsHeld?: number): number {
    const ind = indicators(candles, idx)
    if (!ind) return 0

    if (hasPosition) {
      // FIX: min_hold_steps — don't close before N steps (trained parameter!)
      const held = stepsHeld ?? 0
      if (held < this.params.min_hold_steps) return 0
      // V2 close logic + trainable thresholds
      if (ind.sma5 > ind.sma14 * this.params.close_sma_short && ind.rsi > this.params.close_rsi_short) return 3
      if (ind.sma5 < ind.sma14 * this.params.close_sma_long && ind.rsi < this.params.close_rsi_long) return 3
      return 0
    }
    // SHORT: V2 entry with trainable RSI/SMA
    if (ind.sma5 < ind.sma14 * this.params.sma_short
        && ind.rsi > this.params.short_rsi_min
        && ind.rsi < this.params.short_rsi_max
        && ind.allDown) {
      return 2
    }
    // LONG: V2 entry with trainable RSI/SMA
    if (ind.sma5 > ind.sma14 * this.params.sma_long
        && ind.rsi > this.params.long_rsi_min
        && ind.rsi < this.params.long_rsi_max
        && ind.allUp) {
      return 1
    }
    return 0
  }
}

export const SNIPER_STRATEGIES: Record<string, () => SniperStrategy> = {
  // NEW: SniperTrend variations
  'SniperTrendV2': () => new SniperTrendV2ShortStrategy(),
  'SniperTrendV5': () => new SniperTrendV5Strategy(),
  // NEW: Rebuilt RSI and BB
  'RSIReversionV2': () => new RSIReversionV2Strategy(),
}

// V5 class moved above (before SNIPER_STRATEGIES)
