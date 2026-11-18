"""Phase 2 - Gate 2: Does ML beat simple rules? (Memory & CPU right-sizing).

Diagnostics, submission-time feature engineering, disjoint job-level split (70/15/15),
baseline vs. ML model evaluation on log-ratio targets across margins [1.0, 1.1, 1.25, 1.5, 2.0],
coverage/wastage/savings analysis, bootstrap job-level 95% CIs, and CPU comparison table.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error


def parse_dag_info(task_name: Any) -> Tuple[str, int]:
    """Parse stage prefix (M/R/J/other) and DAG stage index from task_name."""
    s = str(task_name).strip()
    if not s:
        return "other", 0
    prefix = s[0] if s[0] in ["M", "R", "J"] else "other"
    m = re.match(r"^[MRJ](\d+)", s)
    stage_idx = int(m.group(1)) if m else 0
    return prefix, stage_idx


def run_gate2(
    data_path: Path,
    output_json: Path,
    train_sample_cap: int = 100000,
    random_state: int = 42,
) -> Dict[str, Any]:
    print("=" * 80)
    print("PHASE 2 - GATE 2: MEMORY & CPU RIGHT-SIZING FEASIBILITY ANALYSIS")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # A. Data diagnostics
    # -------------------------------------------------------------------------
    print("\n--- SECTION A: DATA DIAGNOSTICS ---")
    t0 = time.perf_counter()
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} raw joined rows in {time.perf_counter() - t0:.2f}s")

    # 1. Value counts (top 10) & share of top value for plan_cpu and plan_mem
    df["plan_cpu_raw"] = (df["plan_cpu_cores"] * 100.0).round().astype(int)
    top10_cpu = df["plan_cpu_raw"].value_counts().head(10)
    top_cpu_val = top10_cpu.index[0]
    top_cpu_count = int(top10_cpu.iloc[0])
    top_cpu_share = top_cpu_count / len(df)

    top10_mem = df["plan_mem"].value_counts().head(10)
    top_mem_val = float(top10_mem.index[0])
    top_mem_count = int(top10_mem.iloc[0])
    top_mem_share = top_mem_count / len(df)

    print("\nTop 10 plan_cpu (100 = 1 full core):")
    for val, count in top10_cpu.items():
        print(f"  {val:>6} : {count:>8,} instances ({count / len(df) * 100:>5.2f}%)")
    cpu_dominated = bool(top_cpu_val == 100 and top_cpu_share > 0.50)
    print(f"Top plan_cpu value is {top_cpu_val} ({top_cpu_share * 100:.2f}% share).")
    print(f"Is plan_cpu dominated by a default (100 cores / 1 core)? {cpu_dominated}")

    print("\nTop 10 plan_mem (normalized relative to node capacity):")
    for val, count in top10_mem.items():
        print(f"  {val:>6.2f} : {count:>8,} instances ({count / len(df) * 100:>5.2f}%)")
    print(f"Top plan_mem value is {top_mem_val:.2f} ({top_mem_share * 100:.2f}% share).")

    # 2. Distinct jobs, distinct tasks, median instances per task, jobs per day-bucket
    n_distinct_jobs = int(df["job_name"].nunique())
    task_groups = df.groupby(["job_name", "task_name"]).size()
    n_distinct_tasks = len(task_groups)
    med_instances_per_task = float(task_groups.median())

    print(f"\nDistinct jobs:            {n_distinct_jobs:,}")
    print(f"Distinct tasks (job,task): {n_distinct_tasks:,}")
    print(f"Median instances per task: {med_instances_per_task:.1f}")

    job_min_times = df.groupby("job_name")["arrival_time"].min()
    min_t = job_min_times.min()
    max_t = job_min_times.max()
    bins = pd.cut(job_min_times, bins=24)
    jobs_per_bin = bins.value_counts().sort_index()

    print("\nJobs per time bucket (24 equal-width bins across trace):")
    bin_counts_dict = {}
    for interval, count in jobs_per_bin.items():
        interval_str = f"[{interval.left:.0f}, {interval.right:.0f}]"
        bin_counts_dict[interval_str] = int(count)
        print(f"  {interval_str:<22} : {count:>5} jobs")

    # 3. Drop rows with plan_mem > 1.0 and clip 99.5th percentiles
    raw_len = len(df)
    df_clean = df[df["plan_mem"] <= 1.0].copy()
    dropped_plan_mem_gt_1 = raw_len - len(df_clean)
    print(f"\nDropped rows with plan_mem > 1.0: {dropped_plan_mem_gt_1:,} rows ({dropped_plan_mem_gt_1 / raw_len * 100:.2f}%)")

    p995_mem_max = float(np.percentile(df_clean["mem_max"], 99.5))
    p995_mem_avg = float(np.percentile(df_clean["mem_avg"], 99.5))
    p995_cpu_max = float(np.percentile(df_clean["cpu_max_cores"], 99.5))

    print(f"99.5th percentile clip thresholds:")
    print(f"  mem_max:       {p995_mem_max:.4f}")
    print(f"  mem_avg:       {p995_mem_avg:.4f}")
    print(f"  cpu_max_cores: {p995_cpu_max:.4f}")

    df_clean["mem_max"] = df_clean["mem_max"].clip(upper=p995_mem_max)
    df_clean["mem_avg"] = df_clean["mem_avg"].clip(upper=p995_mem_avg)
    df_clean["cpu_max_cores"] = df_clean["cpu_max_cores"].clip(upper=p995_cpu_max)

    # -------------------------------------------------------------------------
    # B. Features known AT SUBMISSION only
    # -------------------------------------------------------------------------
    print("\n--- SECTION B: FEATURE ENGINEERING (SUBMISSION-TIME ONLY) ---")
    hour = (df_clean["arrival_time"] // 3600) % 24
    df_clean["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    df_clean["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    dag_info = df_clean["task_name"].apply(parse_dag_info)
    df_clean["stage_prefix"] = [x[0] for x in dag_info]
    df_clean["dag_stage_idx"] = [x[1] for x in dag_info]

    df_clean["job_size"] = df_clean.groupby("job_name")["task_name"].transform("nunique")
    df_clean["mem_cpu_ratio"] = df_clean["plan_mem"] / df_clean["plan_cpu_cores"]

    # Target log ratios
    df_clean["log_ratio_mem"] = np.log(df_clean["mem_max"] / df_clean["plan_mem"])
    df_clean["log_ratio_cpu"] = np.log(df_clean["cpu_max_cores"] / df_clean["plan_cpu_cores"])

    feature_cols_num = [
        "plan_cpu_cores",
        "plan_mem",
        "instance_num",
        "dag_stage_idx",
        "job_size",
        "mem_cpu_ratio",
        "hour_sin",
        "hour_cos",
    ]
    feature_cols_cat = ["task_type", "stage_prefix"]
    print(f"Features: Numerical={feature_cols_num}, Categorical={feature_cols_cat}")

    # -------------------------------------------------------------------------
    # C. Split by JOB
    # -------------------------------------------------------------------------
    print("\n--- SECTION C: DISJOINT JOB-LEVEL CHRONOLOGICAL SPLIT ---")
    job_order = df_clean.groupby("job_name")["arrival_time"].min().sort_values()
    ordered_jobs = job_order.index.tolist()
    total_jobs = len(ordered_jobs)

    n_train_jobs = int(total_jobs * 0.70)
    n_val_jobs = int(total_jobs * 0.15)

    train_job_set = set(ordered_jobs[:n_train_jobs])
    val_job_set = set(ordered_jobs[n_train_jobs : n_train_jobs + n_val_jobs])
    test_job_set = set(ordered_jobs[n_train_jobs + n_val_jobs :])

    # Assert disjoint
    assert train_job_set.isdisjoint(val_job_set), "Train and validation jobs overlap!"
    assert train_job_set.isdisjoint(test_job_set), "Train and test jobs overlap!"
    assert val_job_set.isdisjoint(test_job_set), "Validation and test jobs overlap!"
    print("Disjoint job assertion passed: No job appears in more than one partition.")

    train_df = df_clean[df_clean["job_name"].isin(train_job_set)].copy()
    val_df = df_clean[df_clean["job_name"].isin(val_job_set)].copy()
    test_df = df_clean[df_clean["job_name"].isin(test_job_set)].copy()

    print(f"Train partition:      {len(train_job_set):>5,} jobs | {len(train_df):>9,} instances ({len(train_df) / len(df_clean) * 100:.1f}%)")
    print(f"Validation partition: {len(val_job_set):>5,} jobs | {len(val_df):>9,} instances ({len(val_df) / len(df_clean) * 100:.1f}%)")
    print(f"Test partition:       {len(test_job_set):>5,} jobs | {len(test_df):>9,} instances ({len(test_df) / len(df_clean) * 100:.1f}%)")

    # Fit encoder on train only
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = ohe.fit_transform(train_df[feature_cols_cat])
    X_val_cat = ohe.transform(val_df[feature_cols_cat])
    X_test_cat = ohe.transform(test_df[feature_cols_cat])

    X_train = np.hstack([train_df[feature_cols_num].values, X_train_cat])
    X_val = np.hstack([val_df[feature_cols_num].values, X_val_cat])
    X_test = np.hstack([test_df[feature_cols_num].values, X_test_cat])

    # Subsample train for tree fitting
    rng = np.random.RandomState(random_state)
    if train_sample_cap and len(X_train) > train_sample_cap:
        fit_indices = rng.choice(len(X_train), size=train_sample_cap, replace=False)
        X_fit = X_train[fit_indices]
        y_fit_mem = train_df["log_ratio_mem"].values[fit_indices]
        y_fit_cpu = train_df["log_ratio_cpu"].values[fit_indices]
    else:
        X_fit = X_train
        y_fit_mem = train_df["log_ratio_mem"].values
        y_fit_cpu = train_df["log_ratio_cpu"].values

    # -------------------------------------------------------------------------
    # D & E. MEMORY TARGET AND MODELS
    # -------------------------------------------------------------------------
    print("\n--- SECTION D & E: MEMORY RIGHT-SIZING MODELS & EVALUATION ---")
    # Baselines
    b1_ratio_mem = float((train_df["mem_max"] / train_df["plan_mem"]).median())

    b2_dict_mem = (
        train_df.groupby(["task_type", "stage_prefix"])
        .apply(lambda g: float((g["mem_max"] / g["plan_mem"]).median()))
        .to_dict()
    )

    test_plan_mem = test_df["plan_mem"].values
    test_mem_max = test_df["mem_max"].values
    val_plan_mem = val_df["plan_mem"].values
    val_mem_max = val_df["mem_max"].values

    # Tune DecisionTree on validation set
    print("Tuning DecisionTreeRegressor max_depth on validation set...")
    best_dt_depth_mem = 3
    best_dt_val_mae_mem = float("inf")
    for depth in [3, 5, 7, 10, 15]:
        dt = DecisionTreeRegressor(max_depth=depth, random_state=random_state)
        dt.fit(X_fit, y_fit_mem)
        val_pred = val_plan_mem * np.exp(dt.predict(X_val))
        val_mae = mean_absolute_error(val_mem_max, val_pred)
        print(f"  max_depth={depth:>2}: validation MAE={val_mae:.4f}")
        if val_mae < best_dt_val_mae_mem:
            best_dt_val_mae_mem = val_mae
            best_dt_depth_mem = depth
    print(f"Selected DecisionTree max_depth: {best_dt_depth_mem}")

    # Build memory models
    mem_models: Dict[str, Any] = {
        "B0_RequestBaseline": None,
        "B1_GlobalMedianRatio": None,
        "B2_PerTaskType": None,
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=best_dt_depth_mem, random_state=random_state),
        "RandomForest": RandomForestRegressor(n_estimators=30, max_depth=10, n_jobs=-1, random_state=random_state),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=40, max_depth=4, random_state=random_state),
        "GBR_quantile_90": GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=40, max_depth=4, random_state=random_state),
        "GBR_quantile_95": GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=40, max_depth=4, random_state=random_state),
    }

    # Generate predictions on test set
    mem_preds: Dict[str, np.ndarray] = {}
    for name, m in mem_models.items():
        t_start = time.perf_counter()
        if name == "B0_RequestBaseline":
            pred = test_plan_mem.copy()
        elif name == "B1_GlobalMedianRatio":
            pred = test_plan_mem * b1_ratio_mem
        elif name == "B2_PerTaskType":
            ratios = test_df.apply(
                lambda r: b2_dict_mem.get((r["task_type"], r["stage_prefix"]), b1_ratio_mem),
                axis=1,
            ).values
            pred = test_plan_mem * ratios
        else:
            m.fit(X_fit, y_fit_mem)
            pred_log = m.predict(X_test)
            pred = test_plan_mem * np.exp(pred_log)
        mem_preds[name] = pred
        dur = time.perf_counter() - t_start
        print(f"  Fitted & predicted {name} in {dur:.2f}s")

    # Error metrics on Test (original units)
    mem_error_metrics: Dict[str, Dict[str, float]] = {}
    print("\nMemory Models: Test Set Error Metrics (original units, clipped):")
    print(f"{'Model':<22} | {'MAE':<8} | {'RMSE':<8} | {'MSE':<8} | {'MedAE':<8}")
    print("-" * 62)
    for name, pred in mem_preds.items():
        mae = float(mean_absolute_error(test_mem_max, pred))
        mse = float(mean_squared_error(test_mem_max, pred))
        rmse = float(np.sqrt(mse))
        medae = float(median_absolute_error(test_mem_max, pred))
        mem_error_metrics[name] = {
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "medae": round(medae, 4),
        }
        print(f"{name:<22} | {mae:<8.4f} | {rmse:<8.4f} | {mse:<8.4f} | {medae:<8.4f}")

    # Margins evaluation on test set
    margins = [1.0, 1.1, 1.25, 1.5, 2.0]
    mem_margin_results: List[Dict[str, Any]] = []

    print("\nMemory Models: Policy x Margin Performance Matrix:")
    print(f"{'Policy':<22} | {'Margin':<6} | {'Coverage':<10} | {'Saved %':<10} | {'Wastage %':<10}")
    print("-" * 68)

    total_plan_mem_test = float(test_plan_mem.sum())

    for name, pred in mem_preds.items():
        for m in margins:
            reserve = pred * m
            covered_mask = reserve >= test_mem_max
            cov = float(covered_mask.mean())
            saved = float(1.0 - (reserve.sum() / total_plan_mem_test))
            if covered_mask.any():
                wastage = float(
                    (reserve[covered_mask] - test_mem_max[covered_mask]).mean()
                    / test_plan_mem[covered_mask].mean()
                )
            else:
                wastage = 0.0

            row = {
                "policy": name,
                "margin": m,
                "coverage": round(cov, 4),
                "saved_pct": round(saved * 100.0, 2),
                "wastage_pct": round(wastage * 100.0, 2),
                "reserve_vector": reserve,  # retained for bootstrap, stripped before JSON export
            }
            mem_margin_results.append(row)
            print(f"{name:<22} | {m:<6.2f} | {cov * 100:>8.2f}% | {saved * 100:>8.2f}% | {wastage * 100:>8.2f}%")

    # Identify policies saving the most memory at coverage targets (90%, 95%, 99%)
    def find_best_policy(target_cov: float, policy_filter=None) -> Dict[str, Any]:
        candidates = [
            r for r in mem_margin_results
            if r["coverage"] >= target_cov and (policy_filter is None or policy_filter(r["policy"]))
        ]
        if not candidates:
            return {"policy": "None", "margin": None, "coverage": 0.0, "saved_pct": -999.0}
        candidates.sort(key=lambda x: x["saved_pct"], reverse=True)
        return candidates[0]

    best_at_90 = find_best_policy(0.90)
    best_at_95 = find_best_policy(0.95)
    best_at_99 = find_best_policy(0.99)

    print("\nOptimal Policy per Coverage Target (Memory):")
    print(f"  Target >= 90%: {best_at_90['policy']} (margin {best_at_90['margin']}) -> Saved: {best_at_90['saved_pct']:.2f}%, Coverage: {best_at_90['coverage']*100:.2f}%")
    print(f"  Target >= 95%: {best_at_95['policy']} (margin {best_at_95['margin']}) -> Saved: {best_at_95['saved_pct']:.2f}%, Coverage: {best_at_95['coverage']*100:.2f}%")
    print(f"  Target >= 99%: {best_at_99['policy']} (margin {best_at_99['margin']}) -> Saved: {best_at_99['saved_pct']:.2f}%, Coverage: {best_at_99['coverage']*100:.2f}%")

    # Bootstrap 95% CI by JOB for memory saved at 95% coverage
    # Simple baselines: B0, B1, B2
    simple_baseline_names = {"B0_RequestBaseline", "B1_GlobalMedianRatio", "B2_PerTaskType"}
    best_simple_95 = find_best_policy(0.95, policy_filter=lambda p: p in simple_baseline_names)
    best_ml_95 = find_best_policy(0.95, policy_filter=lambda p: p not in simple_baseline_names)

    print(f"\nCandidates for 95% Coverage Bootstrap:")
    print(f"  Best Simple Baseline: {best_simple_95['policy']} (m={best_simple_95['margin']}) saved={best_simple_95['saved_pct']:.2f}%")
    print(f"  Best ML Model:       {best_ml_95['policy']} (m={best_ml_95['margin']}) saved={best_ml_95['saved_pct']:.2f}%")

    # Pre-aggregate per job for instant 200 resamples
    test_jobs_list = sorted(list(test_job_set))
    job_indices_map = test_df.groupby("job_name").indices

    # Compute per-job sums of plan_mem and reserve for both policies
    job_plan_mem = np.zeros(len(test_jobs_list))
    job_reserve_simple = np.zeros(len(test_jobs_list))
    job_reserve_ml = np.zeros(len(test_jobs_list))

    simple_res_vec = best_simple_95["reserve_vector"]
    ml_res_vec = best_ml_95["reserve_vector"]

    for i, j_name in enumerate(test_jobs_list):
        idxs = job_indices_map[j_name]
        job_plan_mem[i] = test_plan_mem[idxs].sum()
        job_reserve_simple[i] = simple_res_vec[idxs].sum()
        job_reserve_ml[i] = ml_res_vec[idxs].sum()

    n_test_jobs = len(test_jobs_list)
    b_simple_savings: List[float] = []
    b_ml_savings: List[float] = []

    boot_rng = np.random.RandomState(random_state)
    for _ in range(200):
        sample_job_idx = boot_rng.choice(n_test_jobs, size=n_test_jobs, replace=True)
        sum_plan = job_plan_mem[sample_job_idx].sum()
        sum_res_simple = job_reserve_simple[sample_job_idx].sum()
        sum_res_ml = job_reserve_ml[sample_job_idx].sum()

        b_simple_savings.append(float(1.0 - (sum_res_simple / sum_plan)) * 100.0)
        b_ml_savings.append(float(1.0 - (sum_res_ml / sum_plan)) * 100.0)

    ci_simple_95 = (float(np.percentile(b_simple_savings, 2.5)), float(np.percentile(b_simple_savings, 97.5)))
    ci_ml_95 = (float(np.percentile(b_ml_savings, 2.5)), float(np.percentile(b_ml_savings, 97.5)))

    # Overlap check
    ci_overlap = not (ci_ml_95[1] < ci_simple_95[0] or ci_simple_95[1] < ci_ml_95[0])

    print(f"\nBootstrap 95% Confidence Intervals (by JOB, 200 resamples):")
    print(f"  {best_simple_95['policy']} (m={best_simple_95['margin']}): [{ci_simple_95[0]:.2f}%, {ci_simple_95[1]:.2f}%]")
    print(f"  {best_ml_95['policy']} (m={best_ml_95['margin']}): [{ci_ml_95[0]:.2f}%, {ci_ml_95[1]:.2f}%]")
    print(f"  Do the 95% CIs overlap? {ci_overlap}")

    # Strip raw reserve vectors from results
    for r in mem_margin_results:
        r.pop("reserve_vector", None)

    # -------------------------------------------------------------------------
    # F. CPU ANALYSIS
    # -------------------------------------------------------------------------
    print("\n--- SECTION F: CPU RIGHT-SIZING ANALYSIS ---")
    test_plan_cpu = test_df["plan_cpu_cores"].values
    test_cpu_max = test_df["cpu_max_cores"].values
    total_plan_cpu_test = float(test_plan_cpu.sum())

    b1_ratio_cpu = float((train_df["cpu_max_cores"] / train_df["plan_cpu_cores"]).median())
    b2_dict_cpu = (
        train_df.groupby(["task_type", "stage_prefix"])
        .apply(lambda g: float((g["cpu_max_cores"] / g["plan_cpu_cores"]).median()))
        .to_dict()
    )

    cpu_models: Dict[str, Any] = {
        "B0_RequestBaseline": None,
        "B1_GlobalMedianRatio": None,
        "B2_PerTaskType": None,
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=random_state),
        "RandomForest": RandomForestRegressor(n_estimators=30, max_depth=10, n_jobs=-1, random_state=random_state),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=40, max_depth=4, random_state=random_state),
        "GBR_quantile_90": GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=40, max_depth=4, random_state=random_state),
        "GBR_quantile_95": GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=40, max_depth=4, random_state=random_state),
    }

    cpu_preds: Dict[str, np.ndarray] = {}
    for name, m in cpu_models.items():
        if name == "B0_RequestBaseline":
            pred = test_plan_cpu.copy()
        elif name == "B1_GlobalMedianRatio":
            pred = test_plan_cpu * b1_ratio_cpu
        elif name == "B2_PerTaskType":
            ratios = test_df.apply(
                lambda r: b2_dict_cpu.get((r["task_type"], r["stage_prefix"]), b1_ratio_cpu),
                axis=1,
            ).values
            pred = test_plan_cpu * ratios
        else:
            m.fit(X_fit, y_fit_cpu)
            pred_log = m.predict(X_test)
            pred = test_plan_cpu * np.exp(pred_log)
        cpu_preds[name] = pred

    cpu_margin_results: List[Dict[str, Any]] = []
    print("\nCPU Models: Policy x Margin Compact Matrix:")
    print(f"{'Policy':<22} | {'Margin':<6} | {'Coverage':<10} | {'Saved %':<10} | {'Wastage %':<10}")
    print("-" * 68)

    for name, pred in cpu_preds.items():
        for m in margins:
            reserve = pred * m
            covered_mask = reserve >= test_cpu_max
            cov = float(covered_mask.mean())
            saved = float(1.0 - (reserve.sum() / total_plan_cpu_test))
            if covered_mask.any():
                wastage = float(
                    (reserve[covered_mask] - test_cpu_max[covered_mask]).mean()
                    / test_plan_cpu[covered_mask].mean()
                )
            else:
                wastage = 0.0

            row = {
                "policy": name,
                "margin": m,
                "coverage": round(cov, 4),
                "saved_pct": round(saved * 100.0, 2),
                "wastage_pct": round(wastage * 100.0, 2),
            }
            cpu_margin_results.append(row)
            print(f"{name:<22} | {m:<6.2f} | {cov * 100:>8.2f}% | {saved * 100:>8.2f}% | {wastage * 100:>8.2f}%")

    cpu_candidates_90 = [r for r in cpu_margin_results if r["coverage"] >= 0.90]
    cpu_positive_savings_90 = [r for r in cpu_candidates_90 if r["saved_pct"] > 0]
    print(f"\nCPU models reaching >= 90% coverage with real savings (> 0%): {len(cpu_positive_savings_90)}")

    # -------------------------------------------------------------------------
    # G. Final Summary & JSON Export
    # -------------------------------------------------------------------------
    final_output = {
        "diagnostics": {
            "top_plan_cpu": {"value": int(top_cpu_val), "count": top_cpu_count, "share": round(top_cpu_share, 4)},
            "top_plan_mem": {"value": float(top_mem_val), "count": top_mem_count, "share": round(top_mem_share, 4)},
            "plan_cpu_dominated_by_default": cpu_dominated,
            "distinct_jobs": n_distinct_jobs,
            "distinct_tasks": n_distinct_tasks,
            "median_instances_per_task": med_instances_per_task,
            "jobs_per_24_time_bins": bin_counts_dict,
            "dropped_plan_mem_gt_1": dropped_plan_mem_gt_1,
            "p995_clip_values": {
                "mem_max": p995_mem_max,
                "mem_avg": p995_mem_avg,
                "cpu_max_cores": p995_cpu_max,
            },
        },
        "split_counts": {
            "train_jobs": len(train_job_set),
            "train_instances": len(train_df),
            "val_jobs": len(val_job_set),
            "val_instances": len(val_df),
            "test_jobs": len(test_job_set),
            "test_instances": len(test_df),
        },
        "memory": {
            "error_metrics": mem_error_metrics,
            "policy_margin_matrix": mem_margin_results,
            "best_policies": {
                "cov_90": {
                    "policy": best_at_90["policy"],
                    "margin": best_at_90["margin"],
                    "coverage": best_at_90["coverage"],
                    "saved_pct": best_at_90["saved_pct"],
                },
                "cov_95": {
                    "policy": best_at_95["policy"],
                    "margin": best_at_95["margin"],
                    "coverage": best_at_95["coverage"],
                    "saved_pct": best_at_95["saved_pct"],
                },
                "cov_99": {
                    "policy": best_at_99["policy"],
                    "margin": best_at_99["margin"],
                    "coverage": best_at_99["coverage"],
                    "saved_pct": best_at_99["saved_pct"],
                },
            },
            "bootstrap_ci_95": {
                "best_simple_baseline": {
                    "policy": best_simple_95["policy"],
                    "margin": best_simple_95["margin"],
                    "ci_95_saved_pct": [round(ci_simple_95[0], 2), round(ci_simple_95[1], 2)],
                },
                "best_ml_model": {
                    "policy": best_ml_95["policy"],
                    "margin": best_ml_95["margin"],
                    "ci_95_saved_pct": [round(ci_ml_95[0], 2), round(ci_ml_95[1], 2)],
                },
                "cis_overlap": ci_overlap,
            },
        },
        "cpu": {
            "policy_margin_matrix": cpu_margin_results,
            "models_reaching_90_cov_positive_savings": len(cpu_positive_savings_90),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved structured metrics to {output_json}")

    # Section Summary
    print("\n" + "=" * 80)
    print("SHORT SUMMARY: MEMORY AT 95% COVERAGE")
    print("=" * 80)
    # Check memory saved at 95% coverage for requested policies
    # RequestBaseline at 95% coverage:
    r_base_95 = next((r for r in mem_margin_results if r["policy"] == "B0_RequestBaseline" and r["coverage"] >= 0.95), None)
    r_b1_95 = next((r for r in mem_margin_results if r["policy"] == "B1_GlobalMedianRatio" and r["coverage"] >= 0.95), None)
    r_b2_95 = next((r for r in mem_margin_results if r["policy"] == "B2_PerTaskType" and r["coverage"] >= 0.95), None)

    print(f"1. RequestBaseline (margin {r_base_95['margin'] if r_base_95 else 'N/A'}): "
          f"Memory saved = {r_base_95['saved_pct'] if r_base_95 else 'Did not reach 95%'}% "
          f"(coverage: {r_base_95['coverage']*100 if r_base_95 else 0:.2f}%)")

    print(f"2. GlobalMedianRatio: "
          f"{'Memory saved = ' + str(r_b1_95['saved_pct']) + '%' if r_b1_95 else 'Did not reach 95% coverage within evaluated margins [1.0, 2.0]'}")

    print(f"3. PerTaskType: "
          f"{'Memory saved = ' + str(r_b2_95['saved_pct']) + '%' if r_b2_95 else 'Did not reach 95% coverage within evaluated margins [1.0, 2.0]'}")

    print(f"4. Best ML Model ({best_ml_95['policy']}, margin {best_ml_95['margin']}): "
          f"Memory saved = {best_ml_95['saved_pct']:.2f}% "
          f"(coverage: {best_ml_95['coverage']*100:.2f}%)")
    print("=" * 80 + "\n")

    return final_output


def main():
    root = Path(__file__).resolve().parent.parent
    data_path = root / "datasets" / "raw" / "joined_usage.csv"
    output_json = root / "results" / "feasibility_gate2.json"

    run_gate2(data_path, output_json)


if __name__ == "__main__":
    main()
