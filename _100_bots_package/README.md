# 100 BOTS PACKAGE — DIVERSE STRATEGIES FOR LIVE TESTING

## Состав

| Группа | Ботов | Описание |
|---|---|---|
| P01-P30 (profitable) | 30 | Топ-модели из Monte Carlo (val>0, test>0) |
| D31-D90 (diverse) | 60 | 22 стратегии × 3 вариации с разными params |
| X91-X100 (bad) | 10 | Намеренно плохие конфиги (control group) |

**Итого: 100 ботов, все на shared аккаунт 171315d1-fb00-4edd-b1d4-f42eefe6339a**

## Стратегии (22 разные)

- random_hold_short: 27 ботов
- donchian_breakout: 9 ботов
- multi_timeframe: 4 ботов
- v2_short: 4 ботов
- macd_trend: 4 ботов
- vwap_reversion: 4 ботов
- momentum_volume: 4 ботов
- supertrend: 4 ботов
- v2_inverted: 3 ботов
- mean_reversion: 3 ботов
- trend_follow: 3 ботов
- bb_reversion: 3 ботов
- stoch_oscillator: 3 ботов
- connors_rsi2: 3 ботов
- zscore_reversion: 3 ботов
- bollinger_squeeze: 3 ботов
- atr_bands: 3 ботов
- heikin_ashi: 3 ботов
- dual_thrust: 3 ботов
- awesome_oscillator: 3 ботов
- golden_cross: 1 ботов
- rsi_extremes: 1 ботов
- bollinger_bounce: 1 ботов
- turtle_donchian: 1 ботов

## Для деплоя

1. `strategies/*.ts` → `/opt/ai-trader/src/strategies/`
2. Добавить case в `base.ts`
3. `configs/*.json` → `/opt/ai-trader/config/bots/`
4. `sandbox-accounts.json` → `/opt/ai-trader/scripts/`
5. `systemctl restart ai-trader-worker.service`
