# HANDOFF: New Bot Strategies for Deployment

## Для другого ИИ: что здесь лежит

Эта папка содержит **новые стратегии** для добавления в AI Trader.
Другой ИИ должен:

1. Скопировать `strategies-code/*.ts` в `/opt/ai-trader/src/strategies/`
2. Добавить `case` в `/opt/ai-trader/src/strategies/base.ts` для каждой стратегии
3. Скопировать `configs/*.json` в `/opt/ai-trader/config/bots/`
4. Обновить `/opt/ai-trader/scripts/sandbox-accounts.json` — назначить bot_name → account_id
5. Перезапустить `ai-trader-worker.service`

## Доступные аккаунты (sandbox-accounts.json)

```
slot_6: 067b7bc8-67c9-45bd-b3ed-17fd96a5b723  (10000 RUB)
slot_7: c87fc2d2-9a20-4a01-b24a-0fc555bb2381  (10000 RUB)
slot_8: 7a4076f5-f01d-464c-990c-523dce3d11f9  (10000 RUB)
slot_9: 67c21126-5296-4f1a-9cb1-fa9868038688  (10000 RUB)
```

Также можно заменить MT-Winner1..5 (если они не делают сделок):
```
MT-Winner1: 72636d0a-1653-4ba9-8f16-d4edf02bb2d3
MT-Winner2: e5075efb-f1f9-4ef9-8695-2abdac44541e
MT-Winner3: 43438735-47e6-4616-b575-7b3473983249
MT-Winner4: 478f3196-048e-41b1-afe6-7432ccd02fff
MT-Winner5: d74a3eec-3fde-4392-8a6b-004cd2d08f56
```

## Стратегии (9 новых)

| # | Файл | Стратегия | Описание |
|---|---|---|---|
| 1 | wiseplat_triple_sma.ts | wiseplat_triple_sma | WISEPLAT 177% (LONG only, SMA9/14/20 crossover) |
| 2 | turtle_donchian.ts | turtle_donchian | Turtle Trading (20-period breakout) |
| 3 | rsi_extremes.ts | rsi_extremes | RSI<25 long, RSI>75 short |
| 4 | bollinger_bounce.ts | bollinger_bounce | BB + RSI mean reversion |
| 5 | macd_trend.ts | macd_trend | MACD + ADX trend follow |
| 6 | vwap_reversion.ts | vwap_reversion | VWAP intraday reversion |
| 7 | momentum_volume.ts | momentum_volume | ROC + volume surge |
| 8 | stoch_oscillator.ts | stoch_oscillator | Stochastic %K + RSI |
| 9 | multi_timeframe.ts | multi_timeframe | Monte Carlo winner (уже на сервере) |

## Configs (9 bot configs)

Каждый конфиг назначает бота на один из свободных аккаунтов.
Параметры — default values, другой ИИ может крутить.

## Важно

- candleInterval: "5min" (как у текущих ботов)
- tickers: все 11 MOEX акций
- filters: commFilterMult=1.2, cooldownTicks=12, maxTradesPerHour=10 (как у V2)
- positionSize: 0.3 (30% от баланса на позицию)
- maxPositionCost: 3000 RUB
