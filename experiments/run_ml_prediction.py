"""Evaluates workload demand forecasting models on realistic cloud trace data."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is on sys.path when executed directly as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.trace_loader import (
    generate_alibaba_style_trace,
    load_csv_trace,
    save_csv_trace,
)
from ml.features import TimeWindowBucket, bucketize_trace
from ml.metrics import mae, r2_score, rmse, smape
from ml.predictor import (
    DemandPredictor,
    EWMAForecaster,
    LinearARForecaster,
    NaiveLastValuePredictor,
)

SAMPLE_TRACE_PATH = PROJECT_ROOT / "datasets" / "sample_alibaba_trace.csv"
WINDOW_SIZE = 30.0
TRAIN_RATIO = 0.70


def prepare_data() -> Tuple[List[TimeWindowBucket], List[TimeWindowBucket]]:
    """Loads trace, bucketizes into fixed-width time windows, and performs chronological train/test split."""
    if not SAMPLE_TRACE_PATH.exists():
        print(f"Generating synthetic Alibaba-style trace at {SAMPLE_TRACE_PATH}...")
        raw_workloads = generate_alibaba_style_trace(n=1000, seed=42)
        save_csv_trace(raw_workloads, SAMPLE_TRACE_PATH)

    workloads = load_csv_trace(SAMPLE_TRACE_PATH)
    buckets = bucketize_trace(workloads, window_size=WINDOW_SIZE)

    split_idx = int(len(buckets) * TRAIN_RATIO)
    train_buckets = buckets[:split_idx]
    test_buckets = buckets[split_idx:]
    return train_buckets, test_buckets


def evaluate_models() -> List[Dict[str, Any]]:
    """Evaluates Naive, EWMA, and Linear AR models using walk-forward prediction across the test set."""
    train_buckets, test_buckets = prepare_data()
    all_buckets = train_buckets + test_buckets
    split_idx = len(train_buckets)

    models: List[Tuple[str, DemandPredictor]] = [
        ("Naive Last-Value", NaiveLastValuePredictor()),
        ("EWMA (Holt Trend)", EWMAForecaster(alpha=0.35, beta=0.1)),
        ("Linear AR (Ridge)", LinearARForecaster(lags=4, ridge_lambda=2.0)),
    ]

    metrics_list: List[Dict[str, Any]] = []

    for name, model in models:
        # Fit on training data
        model.fit(train_buckets)

        y_true_cpu: List[float] = []
        y_pred_cpu: List[float] = []
        y_true_ram: List[float] = []
        y_pred_ram: List[float] = []
        y_true_cnt: List[float] = []
        y_pred_cnt: List[float] = []

        # Walk-forward rolling evaluation
        for t in range(split_idx, len(all_buckets)):
            history = all_buckets[:t]
            target = all_buckets[t]

            forecast = model.predict_next(history)

            y_true_cpu.append(target.cpu_demand)
            y_pred_cpu.append(forecast.predicted_cpu)

            y_true_ram.append(target.ram_demand)
            y_pred_ram.append(forecast.predicted_ram)

            y_true_cnt.append(float(target.arrival_count))
            y_pred_cnt.append(forecast.predicted_arrivals)

        metrics_list.append(
            {
                "model": name,
                "test_windows": len(test_buckets),
                "cpu_mae": round(mae(y_true_cpu, y_pred_cpu), 3),
                "cpu_rmse": round(rmse(y_true_cpu, y_pred_cpu), 3),
                "cpu_smape_pct": round(smape(y_true_cpu, y_pred_cpu), 2),
                "cpu_r2": round(r2_score(y_true_cpu, y_pred_cpu), 3),
                "ram_mae": round(mae(y_true_ram, y_pred_ram), 3),
                "ram_rmse": round(rmse(y_true_ram, y_pred_ram), 3),
                "ram_smape_pct": round(smape(y_true_ram, y_pred_ram), 2),
                "ram_r2": round(r2_score(y_true_ram, y_pred_ram), 3),
                "arrivals_mae": round(mae(y_true_cnt, y_pred_cnt), 3),
            }
        )

    return metrics_list


def display_metrics_table(metrics: List[Dict[str, Any]]) -> None:
    """Prints an aligned ASCII comparison table for the forecasting benchmarks."""
    header = (
        f"{'Forecasting Model':<20} | "
        f"{'CPU MAE':>7} | {'CPU RMSE':>8} | {'CPU sMAPE':>9} | {'CPU R2':>6} | "
        f"{'RAM MAE':>7} | {'RAM RMSE':>8} | {'RAM sMAPE':>9} | {'RAM R2':>6} | "
        f"{'Arr MAE':>7}"
    )
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print(" CloudPilot ML Workload Demand Forecasting Benchmark Results (30s Windows)")
    print("=" * len(header))
    print(header)
    print(separator)

    for m in metrics:
        print(
            f"{m['model']:<20} | "
            f"{m['cpu_mae']:>7.3f} | "
            f"{m['cpu_rmse']:>8.3f} | "
            f"{m['cpu_smape_pct']:>8.2f}% | "
            f"{m['cpu_r2']:>6.3f} | "
            f"{m['ram_mae']:>7.3f} | "
            f"{m['ram_rmse']:>8.3f} | "
            f"{m['ram_smape_pct']:>8.2f}% | "
            f"{m['ram_r2']:>6.3f} | "
            f"{m['arrivals_mae']:>7.3f}"
        )
    print(separator + "\n")


def save_results(metrics: List[Dict[str, Any]], results_dir: str = "results") -> None:
    """Exports metrics to JSON and CSV formats."""
    out_dir = PROJECT_ROOT / results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "ml_prediction_metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    csv_path = out_dir / "ml_prediction_metrics.csv"
    if metrics:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)

    print(f"Metrics successfully saved to:\n  - {json_path}\n  - {csv_path}")


def main() -> None:
    metrics = evaluate_models()
    display_metrics_table(metrics)
    save_results(metrics)


if __name__ == "__main__":
    main()

