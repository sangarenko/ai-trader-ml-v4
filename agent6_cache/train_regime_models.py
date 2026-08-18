#!/usr/bin/env python3
"""Regime-Aware ML Trainer.

Обучает 6 XGBoost моделей (3 режима × 2 направления) для предсказания
сигналов long/short в зависимости от состояния рынка:

  TREND_UP   — SMA50 > SMA20 > SMA14 + ADX > 20  (растёт)
  TREND_DOWN — SMA50 < SMA20 < SMA14 + ADX > 20  (падает)
  RANGE      — SMA50 ≈ SMA20 ≈ SMA14 + ADX < 20  (флэт)

Каждая модель обучается ТОЛЬКО на барах своего режима. Это позволяет
модели выучивать разные закономерности для разных состояний рынка:

  - В TREND_UP:   momentum-стратегии, пробои вверх
  - В TREND_DOWN: mean reversion, откаты после падения
  - В RANGE:      перекупленность/перепроданность, Bollinger bounce

Фичи (34 шт.):
  - Базовые 31 из ml_features.py (returns, SMA, RSI, Bollinger, MACD, ATR,
    Volume, Stochastic, higher TF, time, ADX)
  - +3 НОВЫЕ сезонные:
    - day_of_month (1-31 / 31)
    - month (1-12 / 12)
    - season (0=win, 1=spr, 2=sum, 3=aut / 3)
    - is_dividend_season (1 для апр-мая и июл-авг, MOEX)

Выходные файлы (в /root/ai-trader-evolution/ml/models/):
  ml_trend_up_long.json
  ml_trend_up_short.json
  ml_trend_down_long.json
  ml_trend_down_short.json
  ml_range_long.json
  ml_range_short.json
  ml_regime_metadata.json

Также обновляет старые ml_model_180d_long.json / _short.json
(обученные на ВСЕХ данных — как fallback для неизвестного режима).

Usage:
  python3 train_regime_models.py --days 180
  python3 train_regime_models.py --days 365  # больше данных
  python3 train_regime_models.py --days 180 --tickers SBER,GAZP
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score

from ml_data_pipeline import download_multi_timeframe, align_timeframes, TICKERS
from ml_features import compute_features, compute_labels

OUTPUT_DIR = Path("/root/ai-trader-evolution/ml/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = "/var/log/ai-trader-regime-train.log"


def log(msg: str) -> None:
    """Лог с timestamp МСК."""
    msk = timezone(timedelta(hours=3))
    ts = datetime.now(msk).strftime("%Y-%m-%d %H:%M:%S МСК")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ─── Новые фичи: сезонность ─────────────────────────────────────────────────

def compute_seasonal_features(time_ms: np.ndarray) -> dict:
    """Сезонные фичи: день месяца, месяц, сезон, дивидендный сезон MOEX.

    Args:
        time_ms: массив timestamp в миллисекундах

    Returns:
        dict с 4 фичами (normalized 0-1):
          - day_of_month
          - month
          - season (0=win, 1=spr, 2=sum, 3=aut)
          - is_dividend_season (бинарная: апр-май, июл-авг на MOEX)
    """
    msk = timezone(timedelta(hours=3))
    dts = [datetime.fromtimestamp(t / 1000, tz=msk) for t in time_ms]

    months = np.array([d.month for d in dts], dtype=float)
    days_of_month = np.array([d.day for d in dts], dtype=float)

    # Season: winter=0 (Dec-Feb), spring=1 (Mar-May),
    #         summer=2 (Jun-Aug), autumn=3 (Sep-Nov)
    seasons = np.zeros(len(dts), dtype=float)
    for i, m in enumerate(months):
        if m in (12, 1, 2):    seasons[i] = 0
        elif m in (3, 4, 5):   seasons[i] = 1
        elif m in (6, 7, 8):   seasons[i] = 2
        else:                  seasons[i] = 3

    # Дивидендный сезон на MOEX: апрель-май (годовые) + июль-август (промежуточные)
    is_div = np.where(((months >= 4) & (months <= 5)) |
                      ((months >= 7) & (months <= 8)), 1.0, 0.0)

    return {
        "day_of_month": days_of_month / 31.0,
        "month": months / 12.0,
        "season": seasons / 3.0,
        "is_dividend_season": is_div,
    }


# ─── Определение режима рынка ────────────────────────────────────────────────

def compute_regime(aligned: dict) -> np.ndarray:
    """Разметка каждого бара по режиму: TREND_UP / TREND_DOWN / RANGE.

    Использует SMA14, SMA20, SMA50 + ADX:
      - TREND_UP   если SMA50 > SMA20 > SMA14 (восходящий порядок) И ADX > 20
      - TREND_DOWN если SMA50 < SMA20 < SMA14 (нисходящий порядок) И ADX > 20
      - RANGE во всех остальных случаях (SMA «сплетены» или ADX низкий)

    Возвращает массив int (0=RANGE, 1=TREND_UP, 2=TREND_DOWN).
    """
    close5 = aligned["5min_close"]
    n = len(close5)

    # Causal SMA
    def causal_sma(arr, w):
        c = np.cumsum(arr, dtype=float)
        result = np.empty(n)
        result[:w] = c[:w] / np.arange(1, min(w, n) + 1)[:n]
        if n > w: result[w:] = (c[w:] - c[:-w]) / w
        return result

    sma14 = causal_sma(close5, 14)
    sma20 = causal_sma(close5, 20)
    sma50 = causal_sma(close5, 50)

    # ADX (упрощённый, как в ml_features.py)
    deltas = np.diff(close5, prepend=close5[0])
    up_moves = np.where(deltas > 0, 1.0, 0.0)
    down_moves = np.where(deltas < 0, 1.0, 0.0)
    def rolling_mean(arr, w):
        ret = np.cumsum(arr)
        ret[w:] = ret[w:] - ret[:-w]
        return ret / w
    adx = np.abs(rolling_mean(up_moves, 14) - rolling_mean(down_moves, 14)) * 100

    # Regime logic
    up_trend = (sma50 > sma20 * 0.999) & (sma20 > sma14 * 0.999) & (adx > 20)
    down_trend = (sma50 < sma20 * 1.001) & (sma20 < sma14 * 1.001) & (adx > 20)

    regime = np.zeros(n, dtype=int)  # 0 = RANGE
    regime[up_trend] = 1             # TREND_UP
    regime[down_trend] = 2           # TREND_DOWN

    # Первые 50 баров — недостаточно истории, считаем RANGE
    regime[:50] = 0

    return regime


REGIME_NAMES = {0: "range", 1: "trend_up", 2: "trend_down"}


# ─── Расширенные фичи (базовые + сезонные) ───────────────────────────────────

def compute_features_v2(aligned: dict):
    """Базовые 31 фича + 4 сезонные = 35 фичей.

    Возвращает: (X, feature_names, regime_array)
    """
    X_base, names_base = compute_features(aligned)

    seasonal = compute_seasonal_features(aligned["time"])

    # Объединяем: базовые + новые (сортировка по имени для детерминизма)
    all_features = {}
    for i, name in enumerate(names_base):
        all_features[name] = X_base[:, i]
    for name, vals in seasonal.items():
        all_features[name] = vals

    feature_names = sorted(all_features.keys())
    X = np.column_stack([all_features[name] for name in feature_names])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -10, 10)
    X[:50] = 0

    regime = compute_regime(aligned)
    return X, feature_names, regime


# ─── Экспорт XGBoost → JSON (как в export_xgboost_json.py) ────────────────────

def export_xgboost_json(model: xgb.XGBClassifier, feature_names: list) -> dict:
    """Экспорт модели в JSON-формат, понятный TypeScript-стратегии.

    Использует booster.get_dump(dump_format='json') — это стандартный способ
    получить деревья XGBoost в виде JSON-строк.

    Структура:
      {
        "n_trees": int,
        "n_features": int,
        "feature_names": [...],
        "base_score": float,
        "trees": [{ nodeid, split, split_condition, yes, no, children, leaf }]
      }
    """
    booster = model.get_booster()

    # Get trees as JSON strings (one per tree)
    trees_json = booster.get_dump(dump_format='json')
    parsed_trees = [json.loads(t) for t in trees_json]

    # Base score (intercept)
    base_score = booster.attr('base_score')
    if base_score is None:
        base_score = 0.5
    else:
        base_score = float(base_score)

    return {
        "n_trees": len(parsed_trees),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "base_score": base_score,
        "trees": parsed_trees,
        "description": "XGBoost binary classifier (regime-aware) exported for TypeScript inference",
        "usage": "raw_score = base_score + sum(tree_predictions); prob = sigmoid(raw_score)",
    }


# ─── Обучение одной модели ───────────────────────────────────────────────────

def train_one_model(X_train, y_train, X_val, y_val, X_test, y_test,
                    name: str, label_balance: str = "long") -> dict:
    """Обучает XGBoost классификатор, возвращает метрики + модель."""
    log(f"  [{name}] train: {len(y_train)} bars, pos={y_train.sum()} ({y_train.mean()*100:.1f}%)")
    log(f"  [{name}] val:   {len(y_val)} bars, pos={y_val.sum()} ({y_val.mean()*100:.1f}%)")

    if y_train.sum() < 50:
        log(f"  [{name}] ⚠️ слишком мало позитивных примеров ({y_train.sum()}) — пропускаю")
        return None

    # Compute scale_pos_weight for imbalanced classes
    neg = len(y_train) - y_train.sum()
    pos = y_train.sum()
    spw = max(1.0, neg / max(1, pos))

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",  # fast on CPU
        random_state=42,
        n_jobs=2,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Метрики
    y_pred_test = (model.predict_proba(X_test)[:, 1] > 0.5).astype(int)
    precision = precision_score(y_test, y_pred_test, zero_division=0)
    recall = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)

    # Precision at top-K (high-confidence predictions)
    y_proba_test = model.predict_proba(X_test)[:, 1]
    for threshold in [0.6, 0.65, 0.7, 0.8]:
        mask = y_proba_test > threshold
        if mask.sum() > 10:
            p_at_t = precision_score(y_test[mask], y_pred_test[mask], zero_division=0)
            log(f"  [{name}] precision@{threshold}: {p_at_t*100:.1f}% (n={mask.sum()})")

    log(f"  [{name}] TEST: precision={precision*100:.1f}% recall={recall*100:.1f}% f1={f1*100:.1f}%")

    return {
        "model": model,
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "train_size": int(len(y_train)),
            "train_pos": int(y_train.sum()),
            "test_size": int(len(y_test)),
            "test_pos": int(y_test.sum()),
        },
    }


# ─── Главный цикл ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180, help="History depth (days)")
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated tickers (default: all 11)")
    parser.add_argument("--horizon", type=int, default=6, help="Forward horizon in candles (6=30min)")
    parser.add_argument("--threshold", type=float, default=0.001, help="Min return to label as up/down (0.001=0.1%)")
    args = parser.parse_args()

    tickers = args.tickers.split(",") if args.tickers else TICKERS
    log("=" * 70)
    log(f"🚀 Regime-Aware ML Trainer — старт")
    log(f"   days={args.days}, tickers={len(tickers)}, horizon={args.horizon}, threshold={args.threshold}")
    log("=" * 70)

    # ─── Шаг 1: Загрузка данных ───
    log("\n[Шаг 1/5] Загрузка данных (MOEX ISS API, кэш 1 день)...")
    all_data = {}
    for i, ticker in enumerate(tickers):
        log(f"  [{i+1}/{len(tickers)}] {ticker}...")
        try:
            data = download_multi_timeframe(ticker, days=args.days)
            if "5min_close" not in data:
                log(f"    SKIP: нет 5min данных")
                continue
            all_data[ticker] = data
        except Exception as e:
            log(f"    ERROR: {e}")
    log(f"  Загружено {len(all_data)}/{len(tickers)} тикеров")

    # ─── Шаг 2: Вычисление фичей + режимов ───
    log("\n[Шаг 2/5] Вычисление фичей + разметка режимов...")
    all_X, all_y_long, all_y_short, all_regime, all_ticker_id = [], [], [], [], []
    feature_names = None

    for i, (ticker, data) in enumerate(all_data.items()):
        log(f"  [{i+1}/{len(all_data)}] {ticker}: align + features...")
        aligned = align_timeframes(data)
        X, names, regime = compute_features_v2(aligned)
        y_long, y_short = compute_labels(aligned, horizon=args.horizon, threshold=args.threshold)

        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            log(f"    ⚠️ разные фичи! {len(names)} vs {len(feature_names)} — пропускаю")
            continue

        all_X.append(X)
        all_y_long.append(y_long)
        all_y_short.append(y_short)
        all_regime.append(regime)
        all_ticker_id.append(np.full(len(X), i))

        # Статистика по режимам для этого тикера
        for r_id, r_name in REGIME_NAMES.items():
            n_r = (regime == r_id).sum()
            log(f"    {r_name:12s}: {n_r} баров ({n_r/len(regime)*100:.1f}%), long={y_long[regime==r_id].sum()}, short={y_short[regime==r_id].sum()}")

    X = np.vstack(all_X)
    y_long = np.concatenate(all_y_long)
    y_short = np.concatenate(all_y_short)
    regime = np.concatenate(all_regime)
    ticker_id = np.concatenate(all_ticker_id)

    log(f"\n  Всего: {len(X)} баров, {len(feature_names)} фичей")
    log(f"  Regime distribution:")
    for r_id, r_name in REGIME_NAMES.items():
        n_r = (regime == r_id).sum()
        log(f"    {r_name:12s}: {n_r} ({n_r/len(regime)*100:.1f}%)")

    # ─── Шаг 3: Хронологический split (train 70% / val 15% / test 15%) ───
    log("\n[Шаг 3/5] Разделение train/val/test (хронологически)...")
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    # ВАЖНО: данные отсортированы по времени (по тикерам), но внутри тикера — хронологично.
    # Для чистоты эксперимента перемешаем тикеры, но сохраним хронологию внутри.
    # Простой подход: последние 15% — test, предыдущие 15% — val, первые 70% — train.
    # (Это не идеально, т.к. разные тикеры могут иметь разную длину, но приемлемо.)

    # Лучше: split внутри каждого тикера отдельно, потом объединить
    train_idx, val_idx, test_idx = [], [], []
    offset = 0
    for i, (ticker, data) in enumerate(all_data.items()):
        n_t = len(all_X[i])
        t_train_end = int(n_t * 0.70)
        t_val_end = int(n_t * 0.85)
        train_idx.extend(range(offset, offset + t_train_end))
        val_idx.extend(range(offset + t_train_end, offset + t_val_end))
        test_idx.extend(range(offset + t_val_end, offset + n_t))
        offset += n_t

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    test_idx = np.array(test_idx)

    log(f"  Train: {len(train_idx)} баров")
    log(f"  Val:   {len(val_idx)} баров")
    log(f"  Test:  {len(test_idx)} баров")

    # ─── Шаг 4: Обучение 6 моделей (3 режима × 2 направления) ───
    log("\n[Шаг 4/5] Обучение моделей по режимам...")

    models = {}  # {(regime_id, direction): model_dict}
    for r_id, r_name in REGIME_NAMES.items():
        for direction, y in [("long", y_long), ("short", y_short)]:
            model_name = f"{r_name}_{direction}"

            # Фильтруем данные по режиму
            r_mask_train = regime[train_idx] == r_id
            r_mask_val = regime[val_idx] == r_id
            r_mask_test = regime[test_idx] == r_id

            X_tr = X[train_idx[r_mask_train]]
            y_tr = y[train_idx[r_mask_train]]
            X_va = X[val_idx[r_mask_val]]
            y_va = y[val_idx[r_mask_val]]
            X_te = X[test_idx[r_mask_test]]
            y_te = y[test_idx[r_mask_test]]

            if len(X_tr) < 200:
                log(f"\n  [{model_name}] ⚠️ мало данных ({len(X_tr)}) — пропускаю")
                continue

            log(f"\n  [{model_name}] обучаю...")
            result = train_one_model(X_tr, y_tr, X_va, y_va, X_te, y_te,
                                     name=model_name, label_balance=direction)
            if result is not None:
                models[(r_id, direction)] = result

    # Также обучаем ОБЩУЮ модель (fallback) — на всех данных без разделения по режиму
    log(f"\n  [all_long] обучаю fallback...")
    fallback_long = train_one_model(
        X[train_idx], y_long[train_idx],
        X[val_idx], y_long[val_idx],
        X[test_idx], y_long[test_idx],
        name="all_long", label_balance="long"
    )
    log(f"\n  [all_short] обучаю fallback...")
    fallback_short = train_one_model(
        X[train_idx], y_short[train_idx],
        X[val_idx], y_short[val_idx],
        X[test_idx], y_short[test_idx],
        name="all_short", label_balance="short"
    )

    # ─── Шаг 5: Сохранение моделей в JSON ───
    log("\n[Шаг 5/5] Сохранение моделей...")

    # Сначала бэкап старых моделей
    backup_dir = OUTPUT_DIR / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(exist_ok=True)
    for old_file in ["ml_model_180d_long.json", "ml_model_180d_short.json",
                     "ml_model_180d.pkl", "ml_model_180d_metadata.json"]:
        old_path = OUTPUT_DIR / old_file
        if old_path.exists():
            import shutil
            shutil.copy(old_path, backup_dir / old_file)
            log(f"  backup: {old_file} → {backup_dir}")

    # Сохраняем regime-модели
    saved_models = {}
    for (r_id, direction), result in models.items():
        r_name = REGIME_NAMES[r_id]
        filename = f"ml_{r_name}_{direction}.json"
        filepath = OUTPUT_DIR / filename
        model_json = export_xgboost_json(result["model"], feature_names)
        with open(filepath, "w") as f:
            json.dump(model_json, f)
        log(f"  ✓ {filename}: {model_json['n_trees']} trees, "
            f"precision={result['metrics']['precision']*100:.1f}%")
        saved_models[f"{r_name}_{direction}"] = result["metrics"]

    # Сохраняем fallback (заменяет ml_model_180d_*.json)
    if fallback_long:
        model_json = export_xgboost_json(fallback_long["model"], feature_names)
        with open(OUTPUT_DIR / "ml_model_180d_long.json", "w") as f:
            json.dump(model_json, f)
        saved_models["all_long"] = fallback_long["metrics"]
        log(f"  ✓ ml_model_180d_long.json: {model_json['n_trees']} trees (fallback)")
    if fallback_short:
        model_json = export_xgboost_json(fallback_short["model"], feature_names)
        with open(OUTPUT_DIR / "ml_model_180d_short.json", "w") as f:
            json.dump(model_json, f)
        saved_models["all_short"] = fallback_short["metrics"]
        log(f"  ✓ ml_model_180d_short.json: {model_json['n_trees']} trees (fallback)")

    # Метаданные
    metadata = {
        "trained_at": datetime.now(timezone(timedelta(hours=3))).isoformat(),
        "days_of_history": args.days,
        "tickers": list(all_data.keys()),
        "horizon_candles": args.horizon,
        "horizon_minutes": args.horizon * 5,  # 5min candles
        "threshold": args.threshold,
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "regimes": {
            REGIME_NAMES[r_id]: {
                "long": saved_models.get(f"{REGIME_NAMES[r_id]}_long", {}),
                "short": saved_models.get(f"{REGIME_NAMES[r_id]}_short", {}),
            }
            for r_id in REGIME_NAMES
        },
        "fallback": {
            "long": saved_models.get("all_long", {}),
            "short": saved_models.get("all_short", {}),
        },
        "regime_logic": {
            "trend_up":   "SMA50 > SMA20 > SMA14 AND ADX > 20",
            "trend_down": "SMA50 < SMA20 < SMA14 AND ADX > 20",
            "range":      "otherwise (SMA mixed OR ADX < 20)",
        },
        "seasonal_features": [
            "day_of_month (1-31 / 31)",
            "month (1-12 / 12)",
            "season (0=win, 1=spr, 2=sum, 3=aut / 3)",
            "is_dividend_season (1 for Apr-May, Jul-Aug on MOEX)",
        ],
    }
    with open(OUTPUT_DIR / "ml_regime_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log(f"  ✓ ml_regime_metadata.json")

    # Также сохраняем .pkl для бэктеста
    if fallback_long and fallback_short:
        import joblib
        pkl_data = {
            "long": fallback_long["model"],
            "short": fallback_short["model"],
            "regime_models": {f"{REGIME_NAMES[r_id]}_{d}": models.get((r_id, d), {}).get("model")
                              for r_id in REGIME_NAMES for d in ["long", "short"]},
            "feature_names": feature_names,
            "metadata": metadata,
        }
        joblib.dump(pkl_data, OUTPUT_DIR / "ml_regime_models.pkl")
        log(f"  ✓ ml_regime_models.pkl (для бэктестов)")

    # Финальный отчёт
    log("\n" + "=" * 70)
    log("📊 ИТОГИ ОБУЧЕНИЯ")
    log("=" * 70)
    log(f"  Features: {len(feature_names)}")
    log(f"  Train/Val/Test: {len(train_idx)} / {len(val_idx)} / {len(test_idx)} баров")
    log("")
    for r_name in ["range", "trend_up", "trend_down"]:
        for direction in ["long", "short"]:
            key = f"{r_name}_{direction}"
            if key in saved_models:
                m = saved_models[key]
                log(f"  {r_name:12s} {direction:5s}: precision={m['precision']*100:5.1f}%  "
                    f"recall={m['recall']*100:5.1f}%  train_n={m['train_size']} test_n={m['test_size']}")
    log("")
    for direction in ["long", "short"]:
        key = f"all_{direction}"
        if key in saved_models:
            m = saved_models[key]
            log(f"  {'FALLBACK':12s} {direction:5s}: precision={m['precision']*100:5.1f}%  "
                f"recall={m['recall']*100:5.1f}%  train_n={m['train_size']} test_n={m['test_size']}")
    log("")
    log(f"📁 Модели сохранены в: {OUTPUT_DIR}")
    log(f"📁 Бэкап старых: {backup_dir}")
    log(f"📝 Лог: {LOG_FILE}")
    log("=" * 70)
    log("✅ Готово!")


if __name__ == "__main__":
    main()
