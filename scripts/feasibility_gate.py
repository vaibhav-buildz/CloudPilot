"""Phase 2 - Step 0: Alibaba Trace Feasibility Gate Analysis.

Evaluates correlation between requested (plan) and actual observed (max) resource usage,
computes over/under-allocation ratios, and benchmarks a chronological 70/30 predictive model
(GradientBoostingRegressor) against the naive baseline 'usage = request'.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder


def compute_feasibility_metrics(
    data_path: Path,
    output_json: Path,
    train_sample_cap: int = 100000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute feasibility metrics, ratios, correlations, and ML benchmark."""
    if not data_path.exists():
        raise FileNotFoundError(f"Input joined data file not found: {data_path}")

    print(f"Loading joined dataset from {data_path}...")
    t0 = time.perf_counter()
    df = pd.read_csv(data_path)
    load_time = time.perf_counter() - t0
    total_rows = len(df)
    print(f"Loaded {total_rows:,} rows in {load_time:.2f}s")

    # 1. Resource Ratios
    df["ratio_cpu"] = df["cpu_max_cores"] / df["plan_cpu_cores"]
    df["ratio_mem"] = df["mem_max"] / df["plan_mem"]

    percentiles = [5, 25, 50, 75, 95, 99]
    cpu_percentiles = {
        f"p{p}": float(np.percentile(df["ratio_cpu"], p)) for p in percentiles
    }
    mem_percentiles = {
        f"p{p}": float(np.percentile(df["ratio_mem"], p)) for p in percentiles
    }

    share_cpu_under_0_5 = float((df["ratio_cpu"] < 0.5).mean())
    share_cpu_over_1_0 = float((df["ratio_cpu"] > 1.0).mean())
    share_mem_under_0_5 = float((df["ratio_mem"] < 0.5).mean())
    share_mem_over_1_0 = float((df["ratio_mem"] > 1.0).mean())

    # 2. Spearman Correlation
    spearman_cpu_res = stats.spearmanr(df["plan_cpu_cores"], df["cpu_max_cores"])
    spearman_mem_res = stats.spearmanr(df["plan_mem"], df["mem_max"])

    spearman_cpu = float(spearman_cpu_res.statistic)
    spearman_cpu_pvalue = float(spearman_cpu_res.pvalue)
    spearman_mem = float(spearman_mem_res.statistic)
    spearman_mem_pvalue = float(spearman_mem_res.pvalue)

    # 3. Chronological 70/30 Split
    print("Sorting chronologically by arrival_time...")
    df_sorted = df.sort_values("arrival_time").reset_index(drop=True)

    df_sorted["hour_of_arrival"] = (df_sorted["arrival_time"] // 3600) % 24

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_sorted["task_type_enc"] = encoder.fit_transform(df_sorted[["task_type"]])

    features = [
        "plan_cpu_cores",
        "plan_mem",
        "instance_num",
        "task_type_enc",
        "hour_of_arrival",
    ]
    X = df_sorted[features]
    y_cpu = df_sorted["cpu_max_cores"]
    y_mem = df_sorted["mem_max"]

    split_idx = int(total_rows * 0.70)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_cpu, y_test_cpu = y_cpu.iloc[:split_idx], y_cpu.iloc[split_idx:]
    y_train_mem, y_test_mem = y_mem.iloc[:split_idx], y_mem.iloc[split_idx:]

    train_rows = len(X_train)
    test_rows = len(X_test)
    print(f"Split completed: Train={train_rows:,} rows (70%), Test={test_rows:,} rows (30%)")

    # Baseline: usage = request
    base_pred_cpu = X_test["plan_cpu_cores"]
    base_pred_mem = X_test["plan_mem"]

    base_mae_cpu = float(mean_absolute_error(y_test_cpu, base_pred_cpu))
    base_r2_cpu = float(r2_score(y_test_cpu, base_pred_cpu))

    base_mae_mem = float(mean_absolute_error(y_test_mem, base_pred_mem))
    base_r2_mem = float(r2_score(y_test_mem, base_pred_mem))

    # GBR Model training
    # For quick, efficient training on large tabular data, sample training set if capped
    if train_sample_cap and train_rows > train_sample_cap:
        rng = np.random.RandomState(random_state)
        sample_indices = rng.choice(train_rows, size=train_sample_cap, replace=False)
        X_train_fit = X_train.iloc[sample_indices]
        y_train_cpu_fit = y_train_cpu.iloc[sample_indices]
        y_train_mem_fit = y_train_mem.iloc[sample_indices]
        fit_rows = train_sample_cap
    else:
        X_train_fit = X_train
        y_train_cpu_fit = y_train_cpu
        y_train_mem_fit = y_train_mem
        fit_rows = train_rows

    print(f"Training GradientBoostingRegressor for CPU on {fit_rows:,} samples...")
    t_cpu = time.perf_counter()
    gbr_cpu = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=random_state)
    gbr_cpu.fit(X_train_fit, y_train_cpu_fit)
    pred_cpu = gbr_cpu.predict(X_test)
    fit_time_cpu = time.perf_counter() - t_cpu

    gbr_mae_cpu = float(mean_absolute_error(y_test_cpu, pred_cpu))
    gbr_r2_cpu = float(r2_score(y_test_cpu, pred_cpu))

    print(f"Training GradientBoostingRegressor for Memory on {fit_rows:,} samples...")
    t_mem = time.perf_counter()
    gbr_mem = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=random_state)
    gbr_mem.fit(X_train_fit, y_train_mem_fit)
    pred_mem = gbr_mem.predict(X_test)
    fit_time_mem = time.perf_counter() - t_mem

    gbr_mae_mem = float(mean_absolute_error(y_test_mem, pred_mem))
    gbr_r2_mem = float(r2_score(y_test_mem, pred_mem))

    results: Dict[str, Any] = {
        "dataset": {
            "path": str(data_path),
            "total_instances": total_rows,
            "train_instances": train_rows,
            "test_instances": test_rows,
            "training_fit_sample_size": fit_rows,
        },
        "ratio_cpu": {
            "percentiles": cpu_percentiles,
            "share_under_0_5": round(share_cpu_under_0_5, 4),
            "share_over_1_0": round(share_cpu_over_1_0, 4),
        },
        "ratio_mem": {
            "percentiles": mem_percentiles,
            "share_under_0_5": round(share_mem_under_0_5, 4),
            "share_over_1_0": round(share_mem_over_1_0, 4),
        },
        "spearman_correlation": {
            "cpu": {
                "statistic": round(spearman_cpu, 4),
                "pvalue": spearman_cpu_pvalue,
            },
            "memory": {
                "statistic": round(spearman_mem, 4),
                "pvalue": spearman_mem_pvalue,
            },
        },
        "chronological_evaluation": {
            "cpu": {
                "baseline_usage_equals_request": {
                    "mae": round(base_mae_cpu, 4),
                    "r2": round(base_r2_cpu, 4),
                },
                "gbr": {
                    "mae": round(gbr_mae_cpu, 4),
                    "r2": round(gbr_r2_cpu, 4),
                    "train_time_sec": round(fit_time_cpu, 2),
                },
            },
            "memory": {
                "baseline_usage_equals_request": {
                    "mae": round(base_mae_mem, 4),
                    "r2": round(base_r2_mem, 4),
                },
                "gbr": {
                    "mae": round(gbr_mae_mem, 4),
                    "r2": round(gbr_r2_mem, 4),
                    "train_time_sec": round(fit_time_mem, 2),
                },
            },
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Exported metrics to {output_json}")

    return results


def print_report(res: Dict[str, Any]) -> None:
    """Print structured tables of feasibility gate metrics."""
    print("\n" + "=" * 78)
    print("PHASE 2 - STEP 0: FEASIBILITY GATE ANALYSIS REPORT")
    print("=" * 78)

    # Table 1: Ratios and Percentiles
    print("\n[1] USAGE / REQUEST RATIO DISTRIBUTIONS")
    print("-" * 78)
    print(f"{'Metric':<18} | {'p5':<8} | {'p25':<8} | {'p50':<8} | {'p75':<8} | {'p95':<8} | {'p99':<8}")
    print("-" * 78)
    c_p = res["ratio_cpu"]["percentiles"]
    m_p = res["ratio_mem"]["percentiles"]
    print(f"{'ratio_cpu':<18} | {c_p['p5']:<8.4f} | {c_p['p25']:<8.4f} | {c_p['p50']:<8.4f} | {c_p['p75']:<8.4f} | {c_p['p95']:<8.4f} | {c_p['p99']:<8.4f}")
    print(f"{'ratio_mem':<18} | {m_p['p5']:<8.4f} | {m_p['p25']:<8.4f} | {m_p['p50']:<8.4f} | {m_p['p75']:<8.4f} | {m_p['p95']:<8.4f} | {m_p['p99']:<8.4f}")
    print("-" * 78)
    print(f"ratio_cpu: < 0.5 (over-allocated) = {res['ratio_cpu']['share_under_0_5'] * 100:.2f}%, > 1.0 (under-requested) = {res['ratio_cpu']['share_over_1_0'] * 100:.2f}%")
    print(f"ratio_mem: < 0.5 (over-allocated) = {res['ratio_mem']['share_under_0_5'] * 100:.2f}%, > 1.0 (under-requested) = {res['ratio_mem']['share_over_1_0'] * 100:.2f}%")

    # Table 2: Spearman Correlations
    print("\n[2] SPEARMAN CORRELATION (REQUEST vs OBSERVED PEAK)")
    print("-" * 78)
    print(f"{'Resource':<18} | {'Spearman Rho':<15} | {'p-value':<15}")
    print("-" * 78)
    sp_c = res["spearman_correlation"]["cpu"]
    sp_m = res["spearman_correlation"]["memory"]
    print(f"{'CPU (plan vs max)':<18} | {sp_c['statistic']:<15.4f} | {sp_c['pvalue']:<15.4e}")
    print(f"{'RAM (plan vs max)':<18} | {sp_m['statistic']:<15.4f} | {sp_m['pvalue']:<15.4e}")
    print("-" * 78)

    # Table 3: Chronological 70/30 Model Evaluation
    print("\n[3] CHRONOLOGICAL 70/30 SPLIT EVALUATION (TEST METRICS)")
    print("-" * 78)
    print(f"{'Target':<14} | {'Model':<30} | {'MAE':<10} | {'R²':<10}")
    print("-" * 78)
    ev_c = res["chronological_evaluation"]["cpu"]
    ev_m = res["chronological_evaluation"]["memory"]
    print(f"{'cpu_max_cores':<14} | {'Baseline (usage = request)':<30} | {ev_c['baseline_usage_equals_request']['mae']:<10.4f} | {ev_c['baseline_usage_equals_request']['r2']:<10.4f}")
    print(f"{'cpu_max_cores':<14} | {'GradientBoostingRegressor':<30} | {ev_c['gbr']['mae']:<10.4f} | {ev_c['gbr']['r2']:<10.4f}")
    print("-" * 78)
    print(f"{'mem_max':<14} | {'Baseline (usage = request)':<30} | {ev_m['baseline_usage_equals_request']['mae']:<10.4f} | {ev_m['baseline_usage_equals_request']['r2']:<10.4f}")
    print(f"{'mem_max':<14} | {'GradientBoostingRegressor':<30} | {ev_m['gbr']['mae']:<10.4f} | {ev_m['gbr']['r2']:<10.4f}")
    print("=" * 78 + "\n")


def main():
    root = Path(__file__).resolve().parent.parent
    data_path = root / "datasets" / "raw" / "joined_usage.csv"
    output_json = root / "results" / "feasibility_gate.json"

    res = compute_feasibility_metrics(data_path, output_json)
    print_report(res)


if __name__ == "__main__":
    main()
