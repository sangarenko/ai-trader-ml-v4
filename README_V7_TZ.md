# ML v7 — Полное ТЗ

**Цель:** V6 + все нюансы кроме LLM (slippage, portfolio risk, order book, ensemble, macro, walk-forward)

## 🎯 Улучшения над V6

### 1. Slippage model (КРИТИЧНО)
- Market order: price = close × (1 + slippage_bps × sign × sqrt(qty/adv))
- slippage_bps = 5 (0.05%) для ликвидных, 20 для неликвидных
- ADV (Average Daily Volume) = mean(volume[-20:])
- Limit order: 50% fill probability, fills at limit price

### 2. Portfolio-level risk
- Correlation matrix между ботами (на основе исторических P&L)
- Concurrent exposure limit: max 3 positions in same ticker
- Portfolio VaR: 95% confidence, max -2% daily loss
- Beta hedging: if portfolio beta > 1.0, reduce positions

### 3. VWAP + Order book features
- VWAP (Volume Weighted Average Price) for day
- VWAP deviation: (close - vwap) / vwap
- Order book imbalance (если MOEX API отдаёт)
- Volume profile (POC, VAH, VAL)

### 4. Ensemble (XGBoost + LightGBM + CatBoost)
- 3 модели обучаются на одних данных
- Stacking: мета-модель (логистическая регрессия) над предсказаниями
- Final prediction = average of 3 (или weighted by val precision)

### 5. Macro features
- Ключевая ставка ЦБ РФ (раз в месяц)
- Курс USD/RUB (дневной return)
- Цены нефти Brent (дневной return)
- IMOEX index return (1d, 1h)
- SBER vs IMOEX beta

### 6. Walk-forward automation
- Каждое воскресенье 02:00 МСК: retrain на свежих данных
- Сравнить OOS precision с предыдущей моделью
- Auto-rollback если precision упала > 10%

### 7. Drift detection
- PSI (Population Stability Index) на features
- Если PSI > 0.2 → trigger retraining
- Real-time monitoring через dashboard

### 8. Better features
- Volume profile (POC, VAH, VAL)
- Cumulative delta (buy volume - sell volume)
- Order flow imbalance
- Intraday seasonality (open/close patterns)

### 9. Risk limits
- Daily loss limit: -2% → stop trading for day
- Drawdown limit: -5% → reduce position size by 50%
- Sector exposure: max 30% in one sector
- Overnight gap risk: close all positions before market close

## 🏗️ Архитектура V7

```
Data (MOEX + macro)
    ↓
Features v7 (35+ features)
    ↓
Regime detection (12 regimes, same as V6)
    ↓
Ensemble (3 models per regime = 36 models total)
    ↓
Portfolio risk manager (correlation, VaR, limits)
    ↓
Slippage-aware execution
    ↓
Live trading + drift monitoring
```

## 📊 Account distribution (10 accounts)

Same as V6:
- acc1: ML-V1 (ml_predict)
- acc2: ML-V2 (ml_predict_v2)
- acc3: ML-V3 (meta_selector)
- acc4: ML-V4 (meta_selector_v4)
- acc5: ML-V5 (v5 models)
- acc6: ML-V6 (v6 models)
- acc7-10: P01-P08 (Monte Carlo)

V7 replaces V6 on acc6 (or add as ML-V7 on acc6 if we want to keep V6 for comparison).

## 📁 Files

### Training (evolution server):
- `features_v7.py` — 35+ features (VWAP, order flow, macro)
- `train_v7.py` — ensemble training (3 models per regime)
- `backtest_v7.py` — slippage-aware backtest
- `walk_forward_v7.py` — automated retraining

### Inference (trader server):
- `meta_selector_v7.ts` — ensemble inference + portfolio risk
- `slippage_model.ts` — slippage calculation
- `portfolio_risk.ts` — correlation matrix + VaR
