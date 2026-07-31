# 60 BOTS PACKAGE — READY FOR DEPLOYMENT

## Что это

60 ботов с 8 стратегиями для live тестирования на T-Bank sandbox.
Все на shared аккаунте 171315d1 (virtualBalance=10000 каждый).

## Стратегии

### MultiTimeframe (multi_timeframe) — 9 ботов

- **Источник:** Monte Carlo winner (our evolution)
- **Backtest:** val +106..+350 RUB, test +13..+425 RUB (6mo MOEX)
- **Логика:** SHORT при даунтренде старшего TF (SMA20<SMA14) + 3 red свечи + RSI 30-55. LONG зеркально.

### WiseplatTripleSma (wiseplat_triple_sma) — 8 ботов

- **Источник:** WISEPLAT GitHub (Strategy 04, 177% backtest)
- **Backtest:** 177% на истории (автор)
- **Логика:** LONG only: SMA9 пересекает SMA14 снизу вверх + SMA20<SMA14. Выход RSI<30.

### TurtleDonchian (turtle_donchian) — 8 ботов

- **Источник:** Richard Dennis Turtle Trading (1980s)
- **Backtest:** 20-80% годовых на трендовых рынках
- **Логика:** LONG при пробое 20-периодного max + объём. SHORT при пробое min. Выход на противоположном пробое.

### RsiExtremes (rsi_extremes) — 8 ботов

- **Источник:** Welles Wilder RSI (1978)
- **Backtest:** На флэте +5-15%, на тренде -10-20%
- **Логика:** LONG при RSI<25. SHORT при RSI>75. Выход при RSI=50.

### BollingerBounce (bollinger_bounce) — 8 ботов

- **Источник:** John Bollinger (1980s)
- **Backtest:** На флэте +5-10%, на тренде убыточна
- **Логика:** LONG при lower BB + RSI<30. SHORT при upper BB + RSI>70. Выход к SMA20.

### MacdTrend (macd_trend) — 8 ботов

- **Источник:** Gerald Appel MACD (1979)
- **Backtest:** На тренде +10-30%, на флэте убыточна
- **Логика:** LONG при MACD>signal + ADX>25. SHORT зеркально. Выход при развороте MACD.

### VwapReversion (vwap_reversion) — 8 ботов

- **Источник:** Институциональные VWAP стратегии
- **Backtest:** +3-8% в стабильные дни
- **Логика:** SHORT при цене>VWAP*1.005+RSI>60. LONG при цене<VWAP*0.995+RSI<40. Выход к VWAP.

### MomentumVolume (momentum_volume) — 2 ботов

- **Источник:** Наша идея: momentum + volume
- **Backtest:** +5-15% на волатильных акциях
- **Логика:** LONG при ROC>2% + объём>1.5x + RSI>50. SHORT зеркально. Выход при развороте ROC.

## Состав

- **MC01-MC10** (10 ботов) — топ-модели из Monte Carlo (val>0 AND test>0 на 6 месяцах MOEX)
- **BT11-BT60** (50 ботов) — 7 стратегий × ~7 вариаций с разными параметрами

## Для деплоя (другому ИИ)

1. Скопировать `strategies/*.ts` в `/opt/ai-trader/src/strategies/`
2. Добавить case в `base.ts` для каждой новой стратегии
3. Скопировать `configs/*.json` в `/opt/ai-trader/config/bots/`
4. Обновить `sandbox-accounts.json`
5. Перезапустить `ai-trader-worker.service`
