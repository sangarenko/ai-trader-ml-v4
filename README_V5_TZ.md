# AI Trader ML v5 — Полное ТЗ обучения

**Статус:** Проектируется (после анализа v1/v2/v3/v4 косяков)
**Дата:** 2026-08-18
**Цель:** Обучить ML модель которая РЕАЛЬНО зарабатывает на MOEX sandbox

---

## 📋 Контекст — что было до этого

### Версии ML которые уже были:

| Версия | Архитектура | Результат | Проблема |
|---|---|---|---|
| V1 (ml_predict.ts) | XGBoost binary, 31 features, P>0.65 LONG | +84₽ за 2 дня ✅ | Fires too often, long-biased |
| V2 (ml_predict_v2.ts) | Regime-aware thresholds + seasonality | -124₽ | Bad thresholds (0.55-0.85) |
| V3 (meta_selector.ts) | Multi-class 22 strategies | -207k OOS | 22 classes = 5% baseline |
| V4 (meta_selector_v4.ts) | 12 regimes × binary classifier | +118k OOS ✅ | Label threshold = commission |

### Археология (3 агента изучили код):

**Что нашли:**
1. **Label threshold = 0.001 = commission roundtrip** → модель учит breakeven, не alpha
2. **commFilterMult = 0 во ВСЕХ ботах** → не отсеивает убыточные сделки
3. **features_v4.py (22 чистых фичи) — не подключен**, обучение использует старые 31 (10 дубликатов)
4. **Per-ticker chronological split** → cross-ticker leakage risk
5. **V4 SHORT bias 68.5%** → модель предсказывает SHORT чаще чем LONG
6. **Coordinated signals** → 4 бота купили MGNT одновременно, -248₽
7. **T-Bank sandbox 50004/35001** → missed trades + phantom positions
8. **ADX/RSI formula inconsistency** → regime_detector.ts (Wilder) vs ml_features.py (simplified)
9. **Higher-TF mismatch** → Python real 1h/1d, TS approximates from 5min
10. **No walk-forward validation** ever done

---

## 🎯 Принципы v5 (NON-NEGOTIABLE)

### 1. Commission-aware labels
```python
COMMISSION_PER_SIDE = 0.0005  # 0.05%
ROUNDTRIP_COMMISSION = 0.001  # 0.1%
PROFIT_MARGIN = 0.001  # 0.1% minimum profit after commission
LABEL_THRESHOLD = ROUNDTRIP_COMMISSION + PROFIT_MARGIN  # = 0.002 (0.2%)

# y = 1 if forward_return > 0.002 else 0
# Это значит: сделка открывается ТОЛЬКО если ожидаемая прибыль > комиссия + margin
```

### 2. Clean features (22, not 31)
Использовать `features_v4.py` (Agent 1 уже написал, но не подключил):
- Returns: ret_1, ret_5, ret_10, ret_30, ret_5_log
- Indicators: rsi14 (NOT rsi2), sma5_sma14, sma14_sma20, sma20_sma50 (ratios only)
- Bollinger: bb_pct_b, bb_width
- MACD: macd_hist ONLY (not line+signal+hist)
- ATR/Stoch/Volume: atr_pct, stoch_k, vol_ratio
- Higher TF: 1h_ret, 1d_ret
- Time: hour, day_of_week
- NEW: market_breadth (% tickers up), sber_gazp_corr (cross-asset), vol_regime (ATR percentile), trend_strength (ADX/100)

### 3. Walk-forward validation (de Prado)
```
Split: NOT random, NOT per-ticker chronological
Use DATE-PURGED global split:
  - Train: days 1-126 (70%)
  - Val: days 127-153 (15%)
  - Test: days 154-180 (15%)
All tickers in same date range — no cross-ticker leakage.
```

### 4. Per-regime binary classifiers (V4 architecture, PROVEN)
- 12 regimes (rule-based detection, 95% accuracy)
- Each regime: separate XGBoost binary classifier
- 10 regimes have enough data (>1000 samples)
- 2 regimes (OVERSOLD_BOUNCE, OVERBOUGHT_REVERSAL) → rule-based fallback

### 5. Inference thresholds (HIGHER, not lower)
```
LONG_THRESHOLD = 0.65  (was 0.6 in V4)
SHORT_THRESHOLD = 0.35  (was 0.4 in V4)
EXIT_LONG = 0.45  (close if P drops below 0.45)
EXIT_SHORT = 0.55  (close if P rises above 0.55)
```
Higher threshold = fewer trades but higher precision.

### 6. Risk-manager config (MUST be set)
```json
{
  "filters": {
    "commFilterMult": 1.5,  // was 0 — SKIP if expGross < 1.5 × commission
    "maxTradesPerHour": 5,   // was 10-30 — reduce overtrading
    "holdTicks": 36,         // 3 hours minimum hold (was 60 ticks = 5h)
    "cooldownTicks": 12      // 1 hour between trades (was 3-6)
  },
  "positionSize": 0.08,     // 8% of balance per trade (was 0.10-0.15)
  "maxPositionCost": 800    // max 800₽ per position (was 1500-2000)
}
```

---

## 🏗️ Архитектура v5

```
┌─────────────────────────────────────────────────────────┐
│  LIVE TRADING (trader server 2.26.122.152)             │
│  ┌──────────────────────────────────────────────────┐   │
│  │  10 bots × 10 accounts (10000₽ each)            │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ...           │   │
│  │  │ V1     │ │ V2     │ │ V3     │               │   │
│  │  │ acc#1  │ │ acc#2  │ │ acc#3  │               │   │
│  │  └────────┘ └────────┘ └────────┘               │   │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ...           │   │
│  │  │ V4     │ │ V5     │ │ P01-P05│               │   │
│  │  │ acc#4  │ │ acc#5  │ │ acc6-10│               │   │
│  │  └────────┘ └────────┘ └────────┘               │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          ↑
                    ML models (.json)
                          ↑
┌─────────────────────────────────────────────────────────┐
│  TRAINING (evolution server 2.26.123.205)              │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Pipeline v5 (5 часов):                          │   │
│  │  1. Download MOEX data (11 tickers × 365 days)  │   │
│  │  2. Compute features_v4 (22 clean features)     │   │
│  │  3. Compute regime (12 regimes, rule-based)      │   │
│  │  4. Compute labels (threshold=0.002, comm-aware)│   │
│  │  5. Date-purged split (70/15/15)                 │   │
│  │  6. Train 12 XGBoost binary classifiers         │   │
│  │  7. Walk-forward backtest (realistic)            │   │
│  │  8. Export to .json (pure-TS inference)         │   │
│  │  9. Deploy to trader server                      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Account distribution (10 accounts)

| Account | Bot | Strategy | Balance | Notes |
|---|---|---|---|---|
| #1 | ML-Trader-V1 | ml_predict.ts | 10000₽ | XGBoost 200 trees, P>0.65 |
| #2 | ML-Trader-V2 | ml_predict_v2.ts | 10000₽ | Regime-aware + seasonality |
| #3 | ML-Trader-V3 | meta_selector.ts | 10000₽ | Multi-class 22 strategies |
| #4 | MetaSelectorV4 | meta_selector_v4.ts | 10000₽ | 12 regimes × binary |
| #5 | MetaSelectorV5 | meta_selector_v5.ts | 10000₽ | **NEW: clean features + comm-aware** |
| #6 | P01+P02 | random_hold_short | 10000₽ | Top-1 + Top-2 from Monte Carlo |
| #7 | P03+P04 | random_hold_short | 10000₽ | Top-3 + Top-4 |
| #8 | P05+P06 | v2_short | 10000₽ | Top-5 + Top-6 |
| #9 | P07+P08 | random_hold_short | 10000₽ | Top-7 + Top-8 |
| #10 | P09+P10 | random_hold_short | 10000₽ | Top-9 + Top-10 |

---

## 🔧 Workflow v5 (5 часов = 18000 секунд)

### Phase 1: Data preparation (30 min)
- Download 365 days of MOEX data (was 180)
- 11 tickers × 365 days × 5min = ~57000 candles each
- Cache to .npz

### Phase 2: Feature engineering (30 min)
- Use `features_v4.py` (22 clean features)
- Compute cross-asset features (market_breadth, sber_gazp_corr)
- Verify no NaN/Inf, no look-ahead bias

### Phase 3: Labeling (15 min)
- `y = 1 if forward_return > 0.002 else 0` (comm-aware)
- horizon = 6 bars (30 min)
- Drop last 6 bars (no forward data)

### Phase 4: Regime detection (15 min)
- 12 regimes via `compute_regime_v2()` (rule-based, 95% accuracy)
- Per-regime sample count (need >1000 per regime)
- 10 regimes train, 2 fallback

### Phase 5: Training (60 min)
- 12 XGBoost binary classifiers
- Date-purged split: 70% train / 15% val / 15% test
- Strong regularization (max_depth=3, reg_lambda=10)
- class_weight for imbalance
- Early stopping on val logloss

### Phase 6: Walk-forward backtest (60 min)
- Simulate: train on days 1-126, predict days 127-153
- Then: train on days 1-153, predict days 154-180
- Realistic: commission 0.05% per side, position size 8%, hold 3h
- Compare to: buy&hold, v1, v4

### Phase 7: Export + Deploy (30 min)
- Export each model to .json (XGBoost raw format)
- Update `meta_selector_v5.ts` (clean features, higher thresholds)
- Deploy to trader server
- Wire to account #5

### Phase 8: Live verification (30 min)
- Wait for first 10 trades
- Check: predictions match backtest?
- Check: commFilterMult=1.5 active?
- Check: no coordinated signals (4 bots same ticker)

---

## ✅ Success criteria

| Metric | Target | V4 actual | V5 goal |
|---|---|---|---|
| Val precision @ P>0.65 | >70% | 70-80% | 75-85% |
| OOS P&L (180 days) | >0 | +118k | +150k+ |
| OOS win rate | >60% | 63% | 68%+ |
| Live P&L (first day) | >0 | -25₽ | >0 |
| Live win rate | >55% | ? | 60%+ |
| Commission as % of gross | <30% | ? | <20% |

---

## 🚫 Что НЕ делать (lessons learned)

1. **НЕ использовать threshold=0.001** (равен комиссии)
2. **НЕ ставить commFilterMult=0** (не отсеивает убыточные)
3. **НЕ использовать 31 features** (10 дубликатов)
4. **НЕ делать per-ticker split** (cross-ticker leakage)
5. **НЕ использовать 22-class classifier** (5% baseline)
6. **НЕ делать positionSize > 0.10** (overtrading)
7. **НЕ делать maxTradesPerHour > 10** (overtrading)
8. **НЕ использовать rsi2** (дубликат rsi14)
9. **НЕ использовать macd_line + macd_signal** (дубликат macd_hist)
10. **НЕ использовать SMA absolutes** (price_sma20, price_sma50) — только ratios

---

## 📁 Files (v5)

### Training (evolution server):
- `/root/ai-trader-evolution/ml/train_v5.py` — main training script
- `/root/ai-trader-evolution/ml/features_v4.py` — 22 clean features (already exists)
- `/root/ai-trader-evolution/ml/walk_forward_v5.py` — walk-forward backtest
- `/root/ai-trader-evolution/ml/meta_models_v2/regime_v5_*.json` — 12 models

### Inference (trader server):
- `/opt/ai-trader/src/strategies/meta_selector_v5.ts` — v5 strategy
- `/opt/ai-trader/src/strategies/regime_detector.ts` — already exists
- `/opt/ai-trader/src/strategies/xgboost_binary_ts.ts` — already exists
- `/opt/ai-trader/src/strategies/regime_v5_*.json` — 12 models (uploaded)

### Configs:
- `/opt/ai-trader/config/bots/bot-meta-selector-v5.json` — v5 bot config
- All bot configs updated: commFilterMult=1.5, positionSize=0.08

---

## 🎯 Итог

V5 = V4 architecture + clean features + comm-aware labels + higher thresholds + risk-manager fix.

Ожидаемый результат: +150k₽ OOS (vs V4 +118k₽), live win rate 60%+ (vs V4 ~50%).
