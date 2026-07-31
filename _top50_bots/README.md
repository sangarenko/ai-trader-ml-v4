# TOP 50 BOTS — BEST MONTE CARLO MODELS

## Результаты за 6 месяцев на 10к RUB

| # | Стратегия | Val P&L | Test P&L | Сумма | Доходность |
|---|---|---|---|---|---|
| 1 | random_hold_short | +4256 | +667 | +4923 | 49.2% |
| 2 | random_hold_short | +4167 | +728 | +4895 | 48.9% |
| 3 | random_hold_short | +3986 | +728 | +4714 | 47.1% |
| 4 | random_hold_short | +2846 | +1565 | +4411 | 44.1% |
| 5 | v2_short | +2966 | +1332 | +4298 | 43.0% |
| 6 | v2_short | +3446 | +785 | +4231 | 42.3% |
| 7 | random_hold_short | +3319 | +891 | +4210 | 42.1% |
| 8 | random_hold_short | +3609 | +527 | +4136 | 41.4% |
| 9 | random_hold_short | +4085 | +50 | +4135 | 41.4% |
| 10 | random_hold_short | +4039 | +89 | +4128 | 41.3% |
| 11 | random_hold_short | +3742 | +366 | +4108 | 41.1% |
| 12 | random_hold_short | +3568 | +514 | +4082 | 40.8% |
| 13 | random_hold_short | +3756 | +249 | +4006 | 40.1% |
| 14 | random_hold_short | +3351 | +618 | +3969 | 39.7% |
| 15 | random_hold_short | +3014 | +946 | +3960 | 39.6% |
| 16 | random_hold_short | +3114 | +817 | +3932 | 39.3% |
| 17 | random_hold_short | +3037 | +799 | +3835 | 38.4% |
| 18 | random_hold_short | +3184 | +619 | +3802 | 38.0% |
| 19 | v2_short | +3471 | +278 | +3750 | 37.5% |
| 20 | v2_short | +3118 | +631 | +3748 | 37.5% |

... и ещё 30 ботов (всего 50)

## Стратегии

- random_hold_short: 37 ботов
- v2_short: 13 ботов

## Для деплоя

1. `configs/*.json` -> `/opt/ai-trader/config/bots/`
2. `sandbox-accounts.json` -> `/opt/ai-trader/scripts/`
3. `systemctl restart ai-trader-worker.service`
