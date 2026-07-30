# 19 STRATEGIES × 60 BOTS — FULL PACKAGE

## 19 стратегий (10 новых из research + 9 существующих)

| # | Стратегия | Источник | Backtest | Ботов |
|---|---|---|---|---|
| 1 | multi_timeframe | Multi-TF: SHORT on HT downtrend + 3 down candles | Monte Carlo winner, val+350 test+180 | 6 |
| 2 | wiseplat_triple_sma | WISEPLAT 177%: LONG SMA9/14 crossover + SMA20 filter | GitHub WISEPLAT, 177% backtest | 3 |
| 3 | turtle_donchian | Turtle: 20-period Donchian breakout | Richard Dennis 1980s, 20-80% yearly | 3 |
| 4 | rsi_extremes | RSI<25 long, RSI>75 short | Welles Wilder 1978, +5-15% on range | 3 |
| 5 | bollinger_bounce | BB lower+RSI<30 long, BB upper+RSI>70 short | John Bollinger, +5-10% on range | 3 |
| 6 | macd_trend | MACD+ADX>25 trend follow | Gerald Appel 1979, +10-30% on trend | 3 |
| 7 | vwap_reversion | VWAP intraday mean reversion | Institutional, +3-8% stable days | 3 |
| 8 | momentum_volume | ROC>2% + volume surge momentum | Our idea, +5-15% volatile | 3 |
| 9 | stoch_oscillator | Stochastic %K + RSI | George Lane 1950s | 3 |
| 10 | connors_rsi2 | Connors RSI(2)<10 long, 77% win rate | Connors, 77% win, 30% annual since 1999 | 3 |
| 11 | zscore_reversion | Z-Score<-2 long, 131% return | AlgoCraft, 131% return, Sharpe 2.11 | 3 |
| 12 | supertrend | Supertrend ATR(10,3), 67% accuracy | QuantifiedStrategies, 67% accuracy | 3 |
| 13 | bollinger_squeeze | BB squeeze breakout, R:R 2:1+ | QuantifiedStrategies, 58% profitable | 3 |
| 14 | atr_bands | ATR volatility bands mean reversion | 33yr backtest profitable | 3 |
| 15 | heikin_ashi | Heikin-Ashi+SMA50, DD 29% vs 52% | QuantifiedStrategies, noise-filtered | 3 |
| 16 | dual_thrust | Dual Thrust breakout | Michael Chalek, intraday classic | 3 |
| 17 | awesome_oscillator | AO+MACD momentum confirmation | je-suis-tm GitHub, fewer whipsaws | 3 |
| 18 | golden_cross | 50/200 SMA crossover, $100k→$7.2M/66yr | QuantifiedStrategies, 66-year backtest | 3 |
| 19 | orb | Opening Range Breakout 5-min, Sharpe 2.81 | ConcretumGroup, Sharpe 2.81 | 3 |

**Итого: 60 ботов, все на shared аккаунт 171315d1-fb00-4edd-b1d4-f42eefe6339a**

## Новые стратегии (10)

1. **connors_rsi2** — Connors RSI(2), 77% win rate, 30% annual
2. **zscore_reversion** — Z-Score mean reversion, 131% return
3. **supertrend** — Supertrend ATR(10,3), 67% accuracy
4. **bollinger_squeeze** — BB squeeze breakout, R:R 2:1+
5. **atr_bands** — ATR volatility bands, 33yr backtest
6. **heikin_ashi** — HA+SMA50, DD 29% vs 52%
7. **dual_thrust** — Dual Thrust breakout, intraday
8. **awesome_oscillator** — AO+MACD momentum
9. **golden_cross** — 50/200 SMA, $100k→$7.2M/66yr
10. **orb** — Opening Range Breakout, Sharpe 2.81

## Для деплоя

1. `strategies/*.ts` → `/opt/ai-trader/src/strategies/`
2. Добавить case в `base.ts` для каждой новой стратегии
3. `configs/*.json` → `/opt/ai-trader/config/bots/`
4. `sandbox-accounts.json` → `/opt/ai-trader/scripts/`
5. `systemctl restart ai-trader-worker.service`
