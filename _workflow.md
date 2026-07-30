# ML Trading Workflow — 4 этапа

## Этап 1: Ускоренный backtest engine (1M моделей/час)

**Проблема:** Сейчас 3000 моделей за 2 часа = 25 моделей/мин. Need 1M/час = 16000/мин.
**Решение:** Vectorized backtest (NumPy arrays + pre-computed indicators)

```
pre-compute indicators ONCE (2 сек)
→ for each model: evaluate signals as boolean arrays (0.001 сек)
→ vectorized P&L calculation (0.001 сек)
→ 1M models × 0.003 сек = 50 мин
```

Files:
- `fast_backtest.py` — vectorized backtest engine
- `fast_monte_carlo.py` — 1M model sweep

## Этап 2: Record ALL results (лучшие + худшие)

**Цель:** Для ML нужны примеры и хороших, и плохих моделей.

```
For each of 1M models:
  - strategy structure
  - 7 parameters (entry_sma_mult, rsi_min, rsi_max, ...)
  - train P&L, val P&L, test P&L
  - trades count, win rate, sortino
  - label: profitable (1) / not profitable (0)
```

Output: `results/all_1m_models.parquet` (ML-ready dataset)

## Этап 3: ML model learns "what makes a strategy profitable"

**Вход:** 1M rows × (strategy_type + 7 params + market_features)
**Выход:** P(strategy profitable) for new unseen params

```
X = [strategy_encoded, entry_sma_mult, rsi_min, rsi_max, take_profit, 
     hold_ticks, exit_sma_mult, position_size, 
     avg_volatility, avg_trend, avg_volume]
y = 1 if val_pnl > 0 AND test_pnl > 0 else 0

Model: XGBoost classifier
  P(profit) > 0.7 → high confidence profitable
```

The ML model learns: "multi_timeframe with entry_sma_mult ~0.995, 
rsi_min ~25, hold_ticks ~120 → 78% chance of profit"

## Этап 4: ML generates new optimal parameters

```
For each strategy structure:
  1. Generate 10000 random param sets
  2. Predict P(profit) for each using ML model
  3. Take top 100 with P(profit) > 0.7
  4. Validate with real backtest
  5. Deploy best 10 to live trading
```

This is **Bayesian optimization via ML** — instead of random search,
ML guides us to profitable regions of parameter space.
