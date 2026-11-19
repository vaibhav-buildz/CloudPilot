"""Phase 2 - Gate 3b: Robustness Fixes for CPU Right-Sizing.

Comprehensive evaluation of CPU right-sizing policies under three splits:
(a) TIME split: 60% Train / 15% Validation / 25% Test per container
(b) APP split: 70% Train / 15% Validation / 15% Test grouped by app_du
(c) APP+TIME split: Train apps on early period, unseen apps on late period

Features validation-only margin selection (grid 1.00..2.00 step 0.05),
app-level 500-draw bootstrap CIs and paired difference CIs, severity metrics,
GBM feature importance, ablation analysis, and memory censoring diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURE_COLS = [
    "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6",
    "lag_7", "lag_8", "lag_9", "lag_10", "lag_11", "lag_12",
    "roll_mean_1h", "roll_std_1h", "roll_max_1h",
    "roll_mean_6h", "roll_std_6h", "roll_max_6h",
    "roll_mean_24h", "roll_std_24h", "roll_max_24h",
    "roll_p99_7d", "lag_48", "seasonal_naive", "ewma_demand",
    "hour_sin", "hour_cos", "req_cpu_cores"
]


def load_data(raw_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load container_meta.csv and container_usage_sample2.csv."""
    meta_path = raw_dir / "container_meta.csv"
    usage_path = raw_dir / "container_usage_sample2.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    if not usage_path.exists():
        raise FileNotFoundError(f"Missing {usage_path}")

    df_meta = pd.read_csv(
        meta_path,
        header=None,
        names=["container_id", "machine_id", "time_stamp", "app_du", "status", "cpu_request", "cpu_limit", "mem_size"],
    )
    meta_unique = df_meta.drop_duplicates(subset=["container_id"], keep="last").copy()
    for col in ["cpu_request", "cpu_limit", "mem_size"]:
        meta_unique[col] = pd.to_numeric(meta_unique[col], errors="coerce")

    df_usage = pd.read_csv(usage_path)
    return meta_unique, df_usage


def prepare_windowed_dataset(
    meta_df: pd.DataFrame, usage_df: pd.DataFrame, window_sec: int = 1800
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Aggregate records into 30-min windows and construct lag, rolling, and temporal features."""
    t0 = time.perf_counter()
    meta_map = meta_df.set_index("container_id")

    usage_df["cpu_limit"] = usage_df["container_id"].map(meta_map["cpu_limit"])
    usage_df["cpu_request"] = usage_df["container_id"].map(meta_map["cpu_request"])
    usage_df["mem_size"] = usage_df["container_id"].map(meta_map["mem_size"])
    usage_df["app_du"] = usage_df["container_id"].map(meta_map["app_du"])

    usage_df["demand_cpu_cores"] = (usage_df["cpu_limit"] / 100.0) * (usage_df["cpu_util_percent"] / 100.0)
    usage_df["demand_mem"] = usage_df["mem_size"] * (usage_df["mem_util_percent"] / 100.0)
    usage_df["win_idx"] = (usage_df["time_stamp"] // window_sec).astype(int)

    # 1. Memory censoring diagnostic
    mem_censored_per_c = usage_df.groupby("container_id")["mem_util_percent"].apply(lambda s: (s >= 100).mean())
    censoring_stats = {
        "min": round(float(mem_censored_per_c.min()) * 100, 2),
        "median": round(float(mem_censored_per_c.median()) * 100, 2),
        "p90": round(float(mem_censored_per_c.quantile(0.90)) * 100, 2),
        "max": round(float(mem_censored_per_c.max()) * 100, 2),
        "share_containers_gt_50pct_censored": round(float((mem_censored_per_c > 0.50).mean()) * 100, 2),
    }

    # 2. Aggregate per container per window
    agg = (
        usage_df.groupby(["container_id", "win_idx"])
        .agg({
            "demand_cpu_cores": "max",
            "demand_mem": "max",
            "cpu_request": "first",
            "cpu_limit": "first",
            "mem_size": "first",
            "app_du": "first",
            "time_stamp": "min",
        })
        .reset_index()
    )

    agg["req_cpu_cores"] = agg["cpu_request"] / 100.0

    # 3. Regularize grid per container & build lag features
    window_dfs = []
    for cid, grp in agg.groupby("container_id"):
        min_w = grp["win_idx"].min()
        max_w = grp["win_idx"].max()
        full_grid = pd.DataFrame({"win_idx": np.arange(min_w, max_w + 1)})
        m = pd.merge(full_grid, grp, on="win_idx", how="left")

        # Forward fill up to 2 windows
        m["demand_cpu_ffill"] = m["demand_cpu_cores"].ffill(limit=2)
        m["req_cpu_cores"] = m["req_cpu_cores"].ffill()
        m["app_du"] = m["app_du"].ffill()
        m["container_id"] = cid
        m["valid_eval"] = ~m["demand_cpu_ffill"].isna()

        s = m["demand_cpu_ffill"]

        # Lags 1 to 12 (strictly past data, no bfill, no future values)
        for l in range(1, 13):
            m[f"lag_{l}"] = s.shift(l)

        # Rolling 1h (2 wins), 6h (12 wins), 24h (48 wins)
        m["roll_mean_1h"] = s.shift(1).rolling(2, min_periods=1).mean()
        m["roll_std_1h"] = s.shift(1).rolling(2, min_periods=1).std().fillna(0.0)
        m["roll_max_1h"] = s.shift(1).rolling(2, min_periods=1).max()

        m["roll_mean_6h"] = s.shift(1).rolling(12, min_periods=1).mean()
        m["roll_std_6h"] = s.shift(1).rolling(12, min_periods=1).std().fillna(0.0)
        m["roll_max_6h"] = s.shift(1).rolling(12, min_periods=1).max()

        m["roll_mean_24h"] = s.shift(1).rolling(48, min_periods=1).mean()
        m["roll_std_24h"] = s.shift(1).rolling(48, min_periods=1).std().fillna(0.0)
        m["roll_max_24h"] = s.shift(1).rolling(48, min_periods=1).max()

        # Rolling p99 of past 7 days (336 windows)
        m["roll_p99_7d"] = s.shift(1).rolling(336, min_periods=1).quantile(0.99)

        # Same time yesterday (lag 48) and Seasonal Naive (max of lag 47, 48, 49)
        m["lag_48"] = s.shift(48)
        lag_47 = s.shift(47)
        lag_49 = s.shift(49)
        m["seasonal_naive"] = np.maximum(np.maximum(lag_47, m["lag_48"]), lag_49)

        # EWMA of window max
        m["ewma_demand"] = s.shift(1).ewm(span=10, min_periods=1).mean()

        # Diurnal features
        hour_of_day = (m["win_idx"] * (window_sec / 3600.0)) % 24.0
        m["hour_sin"] = np.sin(2.0 * np.pi * hour_of_day / 24.0)
        m["hour_cos"] = np.cos(2.0 * np.pi * hour_of_day / 24.0)

        # Target: demand at t+1
        m["target_next"] = s.shift(-1)

        # Drop undefined-lag rows per container (dropping warmup period where lags are undefined)
        m_clean = m.dropna(subset=FEATURE_COLS + ["target_next"]).copy()
        if len(m_clean) > 0:
            window_dfs.append(m_clean)

    full_df = pd.concat(window_dfs, ignore_index=True)
    dt = time.perf_counter() - t0
    print(f"Constructed windowed dataset: {len(full_df):,} window instances across {len(window_dfs):,} containers in {dt:.2f}s")
    print(f"Lag-feature integrity: NO bfill used. Undefined-lag rows dropped per container. Zero future-value leakage.")
    return full_df, censoring_stats


def simulate_rule_threshold(
    cids: np.ndarray, demands: np.ndarray, requests: np.ndarray
) -> np.ndarray:
    """Sequential simulation of RuleThreshold autoscaler."""
    n = len(demands)
    reserves = np.zeros(n)
    curr_cid = None
    r_curr = 0.0
    consec_high = 0
    consec_low = 0

    for i in range(n):
        cid_i = cids[i]
        d_i = demands[i]
        req_i = requests[i]

        if cid_i != curr_cid:
            curr_cid = cid_i
            r_curr = req_i
            consec_high = 0
            consec_low = 0

        reserves[i] = r_curr
        util_i = d_i / max(r_curr, 1e-4)
        if util_i > 0.80:
            consec_high += 1
            consec_low = 0
            if consec_high >= 2:
                r_curr = min(r_curr * 1.25, req_i * 1.50)
                consec_high = 0
        elif util_i < 0.40:
            consec_low += 1
            consec_high = 0
            if consec_low >= 6:
                r_curr = max(r_curr * 0.80, req_i * 0.10)
                consec_low = 0
        else:
            consec_high = 0
            consec_low = 0

    return reserves


def evaluate_split_experiments(
    full_df: pd.DataFrame,
    split_name: str,
    feature_cols: List[str],
    margin_grid: np.ndarray,
    target_violations: List[float] = [0.05, 0.01, 0.001],
    train_sample_cap: int = 50000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train ML models and evaluate all policies across validation and test partitions."""
    print(f"\n--- Running Split Evaluation: {split_name} ---")
    t0 = time.perf_counter()

    # 1. Assign split masks
    if split_name == "TIME":
        # 60% Train, 15% Validation, 25% Test per container
        def assign_time(group):
            n = len(group)
            c1 = int(n * 0.60)
            c2 = int(n * 0.75)
            mask_tr = np.zeros(n, dtype=bool)
            mask_val = np.zeros(n, dtype=bool)
            mask_te = np.zeros(n, dtype=bool)
            mask_tr[:c1] = True
            mask_val[c1:c2] = True
            mask_te[c2:] = True
            group["is_train"] = mask_tr
            group["is_val"] = mask_val
            group["is_test"] = mask_te
            return group

        df_s = full_df.groupby("container_id", group_keys=False).apply(assign_time)
        train_mask = df_s["is_train"]
        val_mask = df_s["is_val"]
        test_mask = df_s["is_test"]

    elif split_name == "APP":
        # 70% Train, 15% Validation, 15% Test grouped by app_du
        unique_apps = sorted(full_df["app_du"].dropna().unique())
        rng_app = np.random.RandomState(random_state)
        shuffled = rng_app.permutation(unique_apps)
        n_tr = int(len(shuffled) * 0.70)
        n_val = int(len(shuffled) * 0.15)
        train_apps = set(shuffled[:n_tr])
        val_apps = set(shuffled[n_tr : n_tr + n_val])
        test_apps = set(shuffled[n_tr + n_val :])

        assert train_apps.isdisjoint(val_apps) and train_apps.isdisjoint(test_apps) and val_apps.isdisjoint(test_apps)

        train_mask = full_df["app_du"].isin(train_apps)
        val_mask = full_df["app_du"].isin(val_apps)
        test_mask = full_df["app_du"].isin(test_apps)

    elif split_name == "APP_TIME":
        # Train apps on early period (first 75%), test unseen apps on late period (last 25%)
        unique_apps = sorted(full_df["app_du"].dropna().unique())
        rng_app = np.random.RandomState(random_state)
        shuffled = rng_app.permutation(unique_apps)
        n_tr = int(len(shuffled) * 0.70)
        n_val = int(len(shuffled) * 0.15)
        train_apps = set(shuffled[:n_tr])
        val_apps = set(shuffled[n_tr : n_tr + n_val])
        test_apps = set(shuffled[n_tr + n_val :])

        def assign_time_app(group):
            n = len(group)
            cutoff = int(n * 0.75)
            mask_early = np.zeros(n, dtype=bool)
            mask_late = np.zeros(n, dtype=bool)
            mask_early[:cutoff] = True
            mask_late[cutoff:] = True
            group["is_early"] = mask_early
            group["is_late"] = mask_late
            return group

        df_s = full_df.groupby("container_id", group_keys=False).apply(assign_time_app)
        train_mask = df_s["app_du"].isin(train_apps) & df_s["is_early"]
        val_mask = df_s["app_du"].isin(val_apps) & df_s["is_early"]
        test_mask = df_s["app_du"].isin(test_apps) & df_s["is_late"]

    # Filter out invalid / NaN rows
    valid_mask = full_df["valid_eval"] & ~full_df["target_next"].isna()
    train_data = full_df[train_mask & valid_mask].dropna(subset=feature_cols + ["target_next"]).copy()
    val_data = full_df[val_mask & valid_mask].dropna(subset=feature_cols + ["target_next"]).copy()
    test_data = full_df[test_mask & valid_mask].dropna(subset=feature_cols + ["target_next"]).copy()

    # Train-only imputation stats (strictly computed on train partition, zero future leakage)
    train_medians = train_data[feature_cols].median()
    train_data[feature_cols] = train_data[feature_cols].fillna(train_medians)
    val_data[feature_cols] = val_data[feature_cols].fillna(train_medians)
    test_data[feature_cols] = test_data[feature_cols].fillna(train_medians)

    print(f"  Rows: Train={len(train_data):,}, Val={len(val_data):,}, Test={len(test_data):,}")
    print(f"  Imputation: strictly train-only medians. Remaining NaNs: Train={train_data[feature_cols].isna().sum().sum()}, Val={val_data[feature_cols].isna().sum().sum()}, Test={test_data[feature_cols].isna().sum().sum()}")

    X_train = train_data[feature_cols].values
    y_train = train_data["target_next"].values
    X_val = val_data[feature_cols].values
    y_val = val_data["target_next"].values
    X_test = test_data[feature_cols].values
    y_test = test_data["target_next"].values

    req_val = val_data["req_cpu_cores"].values
    req_test = test_data["req_cpu_cores"].values

    # Fit ML models (subsample train for trees if necessary)
    if len(X_train) > train_sample_cap:
        rng_fit = np.random.RandomState(random_state)
        sub_idx = rng_fit.choice(len(X_train), size=train_sample_cap, replace=False)
        X_fit = X_train[sub_idx]
        y_fit = y_train[sub_idx]
    else:
        X_fit = X_train
        y_fit = y_train

    ml_models = {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=random_state),
        "RandomForest": RandomForestRegressor(n_estimators=30, max_depth=10, n_jobs=-1, random_state=random_state),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=40, max_depth=4, random_state=random_state),
        "GBR_quantile_95": GradientBoostingRegressor(loss="quantile", alpha=0.95, n_estimators=40, max_depth=4, random_state=random_state),
        "GBR_quantile_99": GradientBoostingRegressor(loss="quantile", alpha=0.99, n_estimators=40, max_depth=4, random_state=random_state),
    }

    preds_val: Dict[str, np.ndarray] = {}
    preds_test: Dict[str, np.ndarray] = {}
    model_objects: Dict[str, Any] = {}

    for name, model in ml_models.items():
        t_m = time.perf_counter()
        model.fit(X_fit, y_fit)
        model_objects[name] = model
        preds_val[name] = np.maximum(model.predict(X_val), 0.0)
        preds_test[name] = np.maximum(model.predict(X_test), 0.0)
        print(f"  Trained {name} in {time.perf_counter() - t_m:.2f}s")

    # Own-History Heuristics
    preds_val["Naive"] = val_data["roll_max_6h"].values
    preds_test["Naive"] = test_data["roll_max_6h"].values

    preds_val["RollingMax48"] = val_data["roll_max_24h"].values
    preds_test["RollingMax48"] = test_data["roll_max_24h"].values

    preds_val["RollingP99_7d"] = val_data["roll_p99_7d"].values
    preds_test["RollingP99_7d"] = test_data["roll_p99_7d"].values

    preds_val["SeasonalNaive"] = val_data["seasonal_naive"].values
    preds_test["SeasonalNaive"] = test_data["seasonal_naive"].values

    preds_val["EWMA"] = val_data["ewma_demand"].values
    preds_test["EWMA"] = test_data["ewma_demand"].values

    # Fleet-wide training statistics (consistent definition across all three splits)
    g_peak = float(train_data["demand_cpu_ffill"].max())
    g_p95 = float(train_data["demand_cpu_ffill"].quantile(0.95))
    g_p99 = float(train_data["demand_cpu_ffill"].quantile(0.99))

    preds_val["FleetPeakOfTraining"] = np.full(len(val_data), g_peak)
    preds_test["FleetPeakOfTraining"] = np.full(len(test_data), g_peak)

    preds_val["FleetPercentile95"] = np.full(len(val_data), g_p95)
    preds_test["FleetPercentile95"] = np.full(len(test_data), g_p95)

    preds_val["FleetPercentile99"] = np.full(len(val_data), g_p99)
    preds_test["FleetPercentile99"] = np.full(len(test_data), g_p99)

    # Per-container training statistics (in TIME: container's own training history; in APP/APP_TIME: unseen containers default to req_cpu_cores)
    if split_name == "TIME":
        peak_map = train_data.groupby("container_id")["demand_cpu_ffill"].max().to_dict()
        p95_map = train_data.groupby("container_id")["demand_cpu_ffill"].quantile(0.95).to_dict()
        p99_map = train_data.groupby("container_id")["demand_cpu_ffill"].quantile(0.99).to_dict()

        preds_val["PeakOfTraining"] = val_data["container_id"].map(peak_map).fillna(val_data["req_cpu_cores"]).values
        preds_test["PeakOfTraining"] = test_data["container_id"].map(peak_map).fillna(test_data["req_cpu_cores"]).values

        preds_val["Percentile95"] = val_data["container_id"].map(p95_map).fillna(val_data["req_cpu_cores"]).values
        preds_test["Percentile95"] = test_data["container_id"].map(p95_map).fillna(test_data["req_cpu_cores"]).values

        preds_val["Percentile99"] = val_data["container_id"].map(p99_map).fillna(val_data["req_cpu_cores"]).values
        preds_test["Percentile99"] = test_data["container_id"].map(p99_map).fillna(test_data["req_cpu_cores"]).values
    else:
        # In APP and APP_TIME, containers are unseen in training; per-container training percentile falls back to request
        preds_val["PeakOfTraining"] = val_data["req_cpu_cores"].values
        preds_test["PeakOfTraining"] = test_data["req_cpu_cores"].values

        preds_val["Percentile95"] = val_data["req_cpu_cores"].values
        preds_test["Percentile95"] = test_data["req_cpu_cores"].values

        preds_val["Percentile99"] = val_data["req_cpu_cores"].values
        preds_test["Percentile99"] = test_data["req_cpu_cores"].values

    # Static & RuleThreshold
    preds_val["Static"] = req_val
    preds_test["Static"] = req_test

    rule_val_res = simulate_rule_threshold(val_data["container_id"].values, y_val, req_val)
    rule_test_res = simulate_rule_threshold(test_data["container_id"].values, y_test, req_test)

    # Helper: compute detailed test metrics
    def compute_test_metrics(reserve_vec: np.ndarray, pred_base: np.ndarray = None) -> Dict[str, Any]:
        viol_mask = y_test > reserve_vec
        viol_rate = float(viol_mask.mean())
        mean_saved = float(1.0 - (reserve_vec.mean() / req_test.mean()))
        wastage = float(np.mean(reserve_vec - y_test) / req_test.mean())

        # Forecast errors if pred_base is available
        if pred_base is not None:
            mae = float(mean_absolute_error(y_test, pred_base))
            mse = float(mean_squared_error(y_test, pred_base))
            rmse = float(np.sqrt(mse))
        else:
            mae = mse = rmse = None

        # Severity metrics
        if viol_mask.any():
            d_viol = y_test[viol_mask]
            r_viol = reserve_vec[viol_mask]
            rel_shortfall = (d_viol - r_viol) / np.maximum(d_viol, 1e-4)
            mean_shortfall = float(rel_shortfall.mean())
            p95_shortfall = float(np.percentile(rel_shortfall, 95))
            throttling_factor = float(np.mean(d_viol / np.maximum(r_viol, 1e-4)))
        else:
            mean_shortfall = p95_shortfall = 0.0
            throttling_factor = 1.0

        # Container-level violation metrics
        test_df_eval = pd.DataFrame({
            "container_id": test_data["container_id"].values,
            "viol": viol_mask,
        })
        viol_per_c = test_df_eval.groupby("container_id")["viol"].sum()
        share_containers_any_viol = float((viol_per_c > 0).mean())

        # Longest run of consecutive violating windows across containers
        max_runs = []
        for cid, grp in test_df_eval.groupby("container_id"):
            v_arr = grp["viol"].values
            max_run = 0
            curr_run = 0
            for v in v_arr:
                if v:
                    curr_run += 1
                    if curr_run > max_run:
                        max_run = curr_run
                else:
                    curr_run = 0
            max_runs.append(max_run)
        longest_run = int(max(max_runs)) if max_runs else 0

        return {
            "violation_rate": round(viol_rate, 4),
            "saved_pct": round(mean_saved * 100.0, 2),
            "wastage_pct": round(wastage * 100.0, 2),
            "mae": round(mae, 4) if mae is not None else None,
            "mse": round(mse, 4) if mse is not None else None,
            "rmse": round(rmse, 4) if rmse is not None else None,
            "mean_shortfall": round(mean_shortfall, 4),
            "p95_shortfall": round(p95_shortfall, 4),
            "share_c_any_viol": round(share_containers_any_viol * 100.0, 2),
            "longest_consec_viol": longest_run,
            "mean_throttling_factor": round(throttling_factor, 2),
        }

    # 2. Validation Margin Grid Search & Calibration Check
    policy_names = list(preds_test.keys())
    tradeoff_rows: List[Dict[str, Any]] = []
    target_results: Dict[str, Dict[str, Any]] = {}

    for target in target_violations:
        target_results[f"target_{int(target*100)}pct" if target >= 0.01 else "target_0_1pct"] = {}

    for pol in policy_names:
        p_val = preds_val[pol]
        p_test = preds_test[pol]

        if pol in ["Static", "RuleThreshold"]:
            grid = [1.0]
        else:
            grid = margin_grid

        # Full margin sweep for tradeoff curves
        for m in grid:
            if pol == "RuleThreshold":
                r_te = rule_test_res
            else:
                r_te = p_test * m
            m_res = compute_test_metrics(r_te, p_test if pol not in ["Static", "RuleThreshold"] else None)
            tradeoff_rows.append({
                "split": split_name,
                "policy": pol,
                "margin": m,
                **m_res,
            })

        # Selection per target
        for target in target_violations:
            tgt_key = f"target_{int(target*100)}pct" if target >= 0.01 else "target_0_1pct"
            best_m = None
            best_val_viol = None

            for m in grid:
                if pol == "RuleThreshold":
                    r_va = rule_val_res
                else:
                    r_va = p_val * m
                v_rate = float((y_val > r_va).mean())
                if v_rate <= target:
                    best_m = m
                    best_val_viol = v_rate
                    break

            met_val_target = (best_m is not None)

            if best_m is None:
                # If target not achieved, choose largest margin (4.00)
                best_m = float(grid[-1])
                r_va = rule_val_res if pol == "RuleThreshold" else p_val * best_m
                best_val_viol = float((y_val > r_va).mean())

            r_te_selected = rule_test_res if pol == "RuleThreshold" else p_test * best_m
            t_res = compute_test_metrics(r_te_selected, p_test if pol not in ["Static", "RuleThreshold"] else None)
            exceeds_2x = bool(t_res["violation_rate"] > 2.0 * target)

            r_va_selected = rule_val_res if pol == "RuleThreshold" else p_val * best_m
            val_saved_pct = float(1.0 - (r_va_selected.mean() / req_val.mean())) * 100.0

            target_results[tgt_key][pol] = {
                "chosen_margin": best_m,
                "val_violation": round(best_val_viol, 4),
                "val_saved_pct": round(val_saved_pct, 2),
                "test_violation": t_res["violation_rate"],
                "test_saved_pct": t_res["saved_pct"],
                "met_val_target": met_val_target,
                "exceeds_2x_target": exceeds_2x,
                "true_generalization_failure": bool(met_val_target and exceeds_2x),
                "full_metrics": t_res,
                "reserve_vector": r_te_selected,
            }

    dt_total = time.perf_counter() - t0
    print(f"Completed {split_name} evaluation in {dt_total:.2f}s")

    return {
        "split": split_name,
        "tradeoff_df": pd.DataFrame(tradeoff_rows),
        "target_results": target_results,
        "test_data": test_data,
        "best_gbm_model": model_objects.get("GradientBoosting"),
    }


def run_bootstrap_apps(
    res_split: Dict[str, Any], target_key: str = "target_1pct", n_draws: int = 500
) -> Dict[str, Any]:
    """Run 500-draw bootstrap by app_du comparing best ML vs feasible versions of rules and point-ML models.

    Model selection for 'best ML' and 'best rule' is performed STRICTLY on the VALIDATION set:
    selecting the candidate satisfying validation violation <= target with maximum validation CPU savings.
    Policies that never reach the validation target within the margin grid are flagged as dropped.
    """
    test_df = res_split["test_data"]
    target_dict = res_split["target_results"][target_key]
    req_v = test_df["req_cpu_cores"].values
    split_name = res_split["split"]

    ml_names = ["LinearRegression", "DecisionTree", "RandomForest", "GradientBoosting", "GBR_quantile_95", "GBR_quantile_99"]
    rule_names = [
        "Naive", "RollingMax48", "RollingP99_7d", "SeasonalNaive", "EWMA",
        "PeakOfTraining", "Percentile95", "Percentile99",
        "FleetPeakOfTraining", "FleetPercentile95", "FleetPercentile99",
        "RuleThreshold"
    ]

    target_val = 0.01 if target_key == "target_1pct" else (0.05 if target_key == "target_5pct" else 0.001)

    # STRICT VALIDATION-BASED SELECTION (never test set):
    def select_best_on_val(candidates: List[str]) -> str:
        meeting_target = [p for p in candidates if p in target_dict and target_dict[p]["met_val_target"]]
        if not meeting_target:
            meeting_target = [p for p in candidates if p in target_dict]
        return max(meeting_target, key=lambda p: (target_dict[p]["val_saved_pct"], -target_dict[p]["val_violation"]))

    best_ml_name = select_best_on_val(ml_names)
    best_rule_name = select_best_on_val(rule_names)

    best_ml_saved = target_dict[best_ml_name]["test_saved_pct"]
    best_ml_test_viol = target_dict[best_ml_name]["test_violation"]
    r_ml = target_dict[best_ml_name]["reserve_vector"]

    unique_apps = test_df["app_du"].dropna().unique()
    app_indices = test_df.groupby("app_du").indices
    app_req_sums = np.array([req_v[app_indices[a]].sum() for a in unique_apps])
    app_ml_sums = np.array([r_ml[app_indices[a]].sum() for a in unique_apps])

    candidate_policies = [
        ("best_rule", best_rule_name),
        ("EWMA", "EWMA"),
        ("SeasonalNaive", "SeasonalNaive"),
        ("RollingMax48", "RollingMax48"),
        ("RollingP99_7d", "RollingP99_7d"),
        ("FleetPercentile99", "FleetPercentile99"),
        ("Percentile99", "Percentile99"),
        ("LinearRegression", "LinearRegression"),
        ("DecisionTree", "DecisionTree"),
        ("RandomForest", "RandomForest"),
        ("GradientBoosting", "GradientBoosting"),
        ("GBR_quantile_99", "GBR_quantile_99"),
    ]

    policy_eval_info: Dict[str, Dict[str, Any]] = {}
    for label, pol in candidate_policies:
        if pol not in target_dict:
            continue
        p_info = target_dict[pol]
        met_val = p_info["met_val_target"]

        # Drop Percentile99 on APP and APP_TIME (unseen containers have no training history)
        if pol == "Percentile99" and split_name != "TIME":
            policy_eval_info[label] = {
                "policy": pol,
                "feasible": False,
                "drop_reason": "Dropped on unseen apps (per-container training usage undefined)",
                "val_violation": p_info["val_violation"],
                "test_violation": p_info["test_violation"],
                "test_saved_pct": p_info["test_saved_pct"],
                "chosen_margin": p_info["chosen_margin"],
            }
            continue

        if not met_val:
            policy_eval_info[label] = {
                "policy": pol,
                "feasible": False,
                "drop_reason": f"Never reached <=1% validation target even at m=4.00 (val_viol={p_info['val_violation']*100:.2f}%)",
                "val_violation": p_info["val_violation"],
                "test_violation": p_info["test_violation"],
                "test_saved_pct": p_info["test_saved_pct"],
                "chosen_margin": p_info["chosen_margin"],
            }
            continue

        r_vec = p_info["reserve_vector"]
        app_p_sums = np.array([r_vec[app_indices[a]].sum() for a in unique_apps])
        policy_eval_info[label] = {
            "policy": pol,
            "feasible": True,
            "drop_reason": None,
            "val_violation": p_info["val_violation"],
            "test_violation": p_info["test_violation"],
            "test_saved_pct": p_info["test_saved_pct"],
            "chosen_margin": p_info["chosen_margin"],
            "app_sums": app_p_sums,
            "boot_saved": [],
            "boot_diff": [],
        }

    n_apps = len(unique_apps)
    rng_b = np.random.RandomState(42)

    b_ml = []
    for _ in range(n_draws):
        idx = rng_b.choice(n_apps, size=n_apps, replace=True)
        sum_req = app_req_sums[idx].sum()
        s_ml = (1.0 - (app_ml_sums[idx].sum() / sum_req)) * 100.0
        b_ml.append(s_ml)

        for label, info in policy_eval_info.items():
            if info["feasible"]:
                s_p = (1.0 - (info["app_sums"][idx].sum() / sum_req)) * 100.0
                info["boot_saved"].append(s_p)
                info["boot_diff"].append(s_ml - s_p)

    ci_ml = [round(float(np.percentile(b_ml, 2.5)), 2), round(float(np.percentile(b_ml, 97.5)), 2)]

    policy_results: Dict[str, Dict[str, Any]] = {}
    for label, info in policy_eval_info.items():
        if info["feasible"]:
            b_s = info["boot_saved"]
            b_d = info["boot_diff"]
            ci_s = [round(float(np.percentile(b_s, 2.5)), 2), round(float(np.percentile(b_s, 97.5)), 2)]
            ci_d = [round(float(np.percentile(b_d, 2.5)), 2), round(float(np.percentile(b_d, 97.5)), 2)]
            spans_zero = bool(ci_d[0] <= 0 <= ci_d[1])
            policy_results[label] = {
                "policy": info["policy"],
                "feasible": True,
                "chosen_margin": info["chosen_margin"],
                "val_violation": info["val_violation"],
                "test_violation": info["test_violation"],
                "test_saved_pct": info["test_saved_pct"],
                "saved_ci_95": ci_s,
                "diff_mean": round(float(np.mean(b_d)), 2),
                "diff_ci_95": ci_d,
                "diff_spans_zero": spans_zero,
            }
        else:
            policy_results[label] = {
                "policy": info["policy"],
                "feasible": False,
                "drop_reason": info["drop_reason"],
                "chosen_margin": info["chosen_margin"],
                "val_violation": info["val_violation"],
                "test_violation": info["test_violation"],
                "test_saved_pct": info["test_saved_pct"],
            }

    best_rule_res = policy_results.get("best_rule", {})
    rule_ci = best_rule_res.get("saved_ci_95", [0.0, 0.0])
    rule_diff_ci = best_rule_res.get("diff_ci_95", [0.0, 0.0])
    cis_overlap = not (ci_ml[1] < rule_ci[0] or rule_ci[1] < ci_ml[0])
    ml_strictly_superior = bool(rule_diff_ci[0] > 0)

    return {
        "split": split_name,
        "best_ml_policy": best_ml_name,
        "best_ml_margin": target_dict[best_ml_name]["chosen_margin"],
        "best_ml_val_viol": target_dict[best_ml_name]["val_violation"],
        "best_ml_test_viol": best_ml_test_viol,
        "best_ml_saved_pct": best_ml_saved,
        "ml_ci_95": ci_ml,
        "best_rule_policy": best_rule_name,
        "best_rule_margin": target_dict[best_rule_name]["chosen_margin"],
        "best_rule_val_viol": target_dict[best_rule_name]["val_violation"],
        "best_rule_test_viol": target_dict[best_rule_name]["test_violation"],
        "best_rule_saved_pct": target_dict[best_rule_name]["test_saved_pct"],
        "rule_ci_95": rule_ci,
        "paired_diff_mean": best_rule_res.get("diff_mean", 0.0),
        "paired_diff_ci_95": rule_diff_ci,
        "cis_overlap": cis_overlap,
        "ml_strictly_superior": ml_strictly_superior,
        "policy_results": policy_results,
    }


def run_ablation_analysis(
    full_df: pd.DataFrame, feature_cols: List[str]
) -> Dict[str, Any]:
    """Ablation analysis on feature subsets for GradientBoostingRegressor."""
    print("\n--- Running Feature Ablation Analysis ---")
    lag_cols = [c for c in feature_cols if "lag" in c or "roll" in c or "seasonal" in c or "ewma" in c]
    lag_cal_cols = lag_cols + ["hour_sin", "hour_cos"]
    full_cols = feature_cols

    ablation_subsets = {
        "lag_only": lag_cols,
        "lag_calendar": lag_cal_cols,
        "lag_request_full": full_cols,
    }

    # TIME split for ablation
    def assign_time(group):
        n = len(group)
        cutoff = int(n * 0.75)
        m_tr = np.zeros(n, dtype=bool)
        m_tr[:cutoff] = True
        group["is_train"] = m_tr
        return group

    df_s = full_df.groupby("container_id", group_keys=False).apply(assign_time)
    valid_mask = full_df["valid_eval"] & ~full_df["target_next"].isna()
    tr_data = full_df[df_s["is_train"] & valid_mask].dropna(subset=full_cols + ["target_next"]).copy()
    te_data = full_df[~df_s["is_train"] & valid_mask].dropna(subset=full_cols + ["target_next"]).copy()

    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(tr_data), size=min(40000, len(tr_data)), replace=False)
    y_tr_sub = tr_data["target_next"].values[sub_idx]
    y_te = te_data["target_next"].values

    ablation_results = {}
    for name, cols in ablation_subsets.items():
        X_tr = tr_data[cols].values[sub_idx]
        X_te = te_data[cols].values

        gbr = GradientBoostingRegressor(n_estimators=40, max_depth=4, random_state=42)
        gbr.fit(X_tr, y_tr_sub)
        preds = np.maximum(gbr.predict(X_te), 0.0)

        mae = float(mean_absolute_error(y_te, preds))
        mse = float(mean_squared_error(y_te, preds))
        rmse = float(np.sqrt(mse))
        ablation_results[name] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "n_features": len(cols)}
        print(f"  Ablation {name:<18} ({len(cols)} features): MAE={mae:.4f}, RMSE={rmse:.4f}")

    return ablation_results


def main():
    start_total_time = time.perf_counter()
    root = Path(__file__).resolve().parent.parent
    raw_dir = root / "datasets" / "raw"
    results_dir = root / "results"
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 2 - GATE 3b: ROBUSTNESS FIXES FOR CPU RIGHT-SIZING")
    print("=" * 80)

    # 1. Load Data
    meta_df, usage_df = load_data(raw_dir)

    # 2. Windowing & Features
    full_df, censoring_stats = prepare_windowed_dataset(meta_df, usage_df, window_sec=1800)

    feature_cols = [
        "lag_1", "lag_2", "lag_3", "lag_4", "lag_5", "lag_6",
        "lag_7", "lag_8", "lag_9", "lag_10", "lag_11", "lag_12",
        "roll_mean_1h", "roll_std_1h", "roll_max_1h",
        "roll_mean_6h", "roll_std_6h", "roll_max_6h",
        "roll_mean_24h", "roll_std_24h", "roll_max_24h",
        "roll_p99_7d", "lag_48", "seasonal_naive", "ewma_demand",
        "hour_sin", "hour_cos", "req_cpu_cores"
    ]

    margin_grid = np.arange(1.00, 4.05, 0.05).round(2)

    # 3. Evaluate Splits (a) TIME, (b) APP, (c) APP+TIME
    res_time = evaluate_split_experiments(full_df, "TIME", feature_cols, margin_grid)
    res_app = evaluate_split_experiments(full_df, "APP", feature_cols, margin_grid)
    res_app_time = evaluate_split_experiments(full_df, "APP_TIME", feature_cols, margin_grid)

    # 4. Bootstrap Comparisons (500 resamples by app)
    boot_time = run_bootstrap_apps(res_time, "target_1pct", n_draws=500)
    boot_app = run_bootstrap_apps(res_app, "target_1pct", n_draws=500)
    boot_app_time = run_bootstrap_apps(res_app_time, "target_1pct", n_draws=500)

    # 5. Feature Importance of Best GBM
    best_gbm = res_time["best_gbm_model"]
    importances = best_gbm.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    feat_imp_dict = {f: round(float(imp), 4) for f, imp in feat_imp}

    print("\nTop 10 Feature Importances (GradientBoostingRegressor):")
    for f, imp in feat_imp[:10]:
        print(f"  {f:<18} : {imp:.4f}")

    # 6. Ablation Analysis
    ablation_res = run_ablation_analysis(full_df, feature_cols)

    # 7. Generate Tradeoff Figure (3 Subplots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_items = [
        ("Split (a): TIME", res_time["tradeoff_df"], axes[0]),
        ("Split (b): APP", res_app["tradeoff_df"], axes[1]),
        ("Split (c): APP+TIME", res_app_time["tradeoff_df"], axes[2]),
    ]

    for title, df_plot, ax in plot_items:
        for pol, grp in df_plot.groupby("policy"):
            g_sorted = grp.sort_values("violation_rate")
            ax.plot(
                g_sorted["violation_rate"] * 100.0,
                g_sorted["saved_pct"],
                marker="o",
                markersize=4,
                label=pol,
                alpha=0.85,
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Violation Rate (%)", fontsize=10)
        ax.set_ylabel("CPU Saved vs Request (%)", fontsize=10)
        ax.set_xlim(-0.5, 20.0)
        ax.grid(True, linestyle="--", alpha=0.5)

    axes[0].legend(loc="lower right", fontsize=7, ncol=2)
    plt.tight_layout()
    fig_path = fig_dir / "gate3b_tradeoff.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"\nSaved trade-off plot to {fig_path}")

    # 8. Save CSV and JSON
    all_tradeoffs = pd.concat([res_time["tradeoff_df"], res_app["tradeoff_df"], res_app_time["tradeoff_df"]], ignore_index=True)
    all_tradeoffs.to_csv(results_dir / "gate3b.csv", index=False)

    # Helper to strip non-serializable arrays
    def clean_target_dict(td):
        cleaned = {}
        for tgt, p_dict in td.items():
            cleaned[tgt] = {}
            for p, d in p_dict.items():
                d_c = d.copy()
                d_c.pop("reserve_vector", None)
                cleaned[tgt][p] = d_c
        return cleaned

    # Core 15 policies from pre-registration
    core_15_policies = [
        "Static", "RuleThreshold", "Naive", "RollingMax48", "RollingP99_7d",
        "SeasonalNaive", "EWMA", "PeakOfTraining", "Percentile95", "Percentile99",
        "LinearRegression", "DecisionTree", "RandomForest", "GradientBoosting",
        "GBR_quantile_95"
    ]
    all_targets = ["target_5pct", "target_1pct", "target_0_1pct"]

    # Hypothesis Verdicts
    # H1: ML saves more than best own-history rule at test violation <= 1% with non-overlapping app CIs
    h1_supported = bool(boot_time["ml_strictly_superior"] and not boot_time["cis_overlap"])

    # H2: Margin chosen on validation keeps test violation within 2x target across all 3 targets (135 checks)
    h2_summary: Dict[str, Any] = {
        "total_checks": 0,
        "total_exceeded": 0,
        "met_val_checks": 0,
        "true_generalization_failures": 0,
        "by_target": {},
        "by_split": {s: {"total": 0, "exceeded": 0, "met_val": 0, "true_failures": 0} for s in ["TIME", "APP", "APP_TIME"]},
        "exceeded_details": []
    }

    for tgt in all_targets:
        t_val = 0.05 if tgt == "target_5pct" else (0.01 if tgt == "target_1pct" else 0.001)
        h2_summary["by_target"][tgt] = {"target_val": t_val, "total": 0, "exceeded": 0, "met_val": 0, "true_failures": 0}

        for s_name, split_res in [("TIME", res_time), ("APP", res_app), ("APP_TIME", res_app_time)]:
            for pol in core_15_policies:
                if pol in split_res["target_results"][tgt]:
                    d = split_res["target_results"][tgt][pol]
                    h2_summary["total_checks"] += 1
                    h2_summary["by_target"][tgt]["total"] += 1
                    h2_summary["by_split"][s_name]["total"] += 1

                    if d["exceeds_2x_target"]:
                        h2_summary["total_exceeded"] += 1
                        h2_summary["by_target"][tgt]["exceeded"] += 1
                        h2_summary["by_split"][s_name]["exceeded"] += 1

                        h2_summary["exceeded_details"].append({
                            "split": s_name,
                            "target": tgt,
                            "target_val": t_val,
                            "policy": pol,
                            "margin": d["chosen_margin"],
                            "val_violation": d["val_violation"],
                            "test_violation": d["test_violation"],
                            "met_val_target": d["met_val_target"],
                            "true_gen_failure": d["true_generalization_failure"]
                        })

                    if d["met_val_target"]:
                        h2_summary["met_val_checks"] += 1
                        h2_summary["by_target"][tgt]["met_val"] += 1
                        h2_summary["by_split"][s_name]["met_val"] += 1

                        if d["exceeds_2x_target"]:
                            h2_summary["true_generalization_failures"] += 1
                            h2_summary["by_target"][tgt]["true_failures"] += 1
                            h2_summary["by_split"][s_name]["true_failures"] += 1

    # Overall H2 supported if true generalization failure rate among feasible policies < 20%
    h2_supported = bool(
        (h2_summary["true_generalization_failures"] / max(h2_summary["met_val_checks"], 1)) < 0.20
    )

    # H3: ML advantage persists on unseen apps
    h3_supported = bool(boot_app["ml_strictly_superior"] and boot_app_time["ml_strictly_superior"])

    final_json = {
        "hypotheses_verdicts": {
            "H1_ml_superiority_at_1pct_violation": {
                "verdict": "SUPPORTED" if h1_supported else "NOT SUPPORTED",
                "details": boot_time,
            },
            "H2_validation_margin_calibratability": {
                "verdict": "SUPPORTED" if h2_supported else "NOT SUPPORTED",
                "h2_summary_135_checks": h2_summary,
            },
            "H3_ml_generalization_unseen_apps": {
                "verdict": "SUPPORTED" if h3_supported else "NOT SUPPORTED",
                "split_app_details": boot_app,
                "split_app_time_details": boot_app_time,
            },
        },
        "bootstrap_results": {
            "TIME": boot_time,
            "APP": boot_app,
            "APP_TIME": boot_app_time,
        },
        "target_results": {
            "TIME": clean_target_dict(res_time["target_results"]),
            "APP": clean_target_dict(res_app["target_results"]),
            "APP_TIME": clean_target_dict(res_app_time["target_results"]),
        },
        "feature_importance_gbm": feat_imp_dict,
        "ablation_results": ablation_res,
        "memory_censoring_diagnostic": censoring_stats,
    }

    with open(results_dir / "gate3b.json", "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)

    total_time = time.perf_counter() - start_total_time
    print(f"Saved results to {results_dir / 'gate3b.csv'} and {results_dir / 'gate3b.json'}")
    print(f"Total pipeline runtime: {total_time:.2f}s")

    # Generate Markdown Tables by Script
    generate_final_tables(results_dir, final_json)

    # Print Summary Tables & Hypothesis Status
    print("\n" + "=" * 175)
    print("GATE 3b SUMMARY: APP-LEVEL BOOTSTRAP AT 1% VIOLATION (VALIDATION-BASED SELECTION, GRID UP TO 4.00)")
    print("=" * 175)
    for s_name, b in [("TIME", boot_time), ("APP", boot_app), ("APP_TIME", boot_app_time)]:
        print(f"\n--- Split: {s_name} ---")
        print(f"Best ML Policy: {b['best_ml_policy']} (Margin={b['best_ml_margin']:.2f}, Val Viol={b['best_ml_val_viol']*100:.2f}%, Test Viol={b['best_ml_test_viol']*100:.2f}%, Test Saved={b['best_ml_saved_pct']:.2f}% [{b['ml_ci_95'][0]}%, {b['ml_ci_95'][1]}%])")
        print(f"{'Policy':<20} | {'Status':<10} | {'Margin':<7} | {'Val Viol':<10} | {'Test Viol':<10} | {'Test Saved [95% CI]':<26} | {'Diff vs ML [95% CI]':<26} | {'Spans 0?'}")
        print("-" * 130)
        for p_label, p_res in b["policy_results"].items():
            pol_name = p_res["policy"]
            if p_res["feasible"]:
                m_str = f"{p_res['chosen_margin']:.2f}"
                vv_str = f"{p_res['val_violation']*100:.2f}%"
                tv_str = f"{p_res['test_violation']*100:.2f}%"
                ts_str = f"{p_res['test_saved_pct']:.2f}% [{p_res['saved_ci_95'][0]}%, {p_res['saved_ci_95'][1]}%]"
                diff_str = f"{p_res['diff_mean']:+.2f}% [{p_res['diff_ci_95'][0]:+.2f}%, {p_res['diff_ci_95'][1]:+.2f}%]"
                spans_str = str(p_res["diff_spans_zero"])
                print(f"{pol_name:<20} | {'Feasible':<10} | {m_str:<7} | {vv_str:<10} | {tv_str:<10} | {ts_str:<26} | {diff_str:<26} | {spans_str}")
            else:
                m_str = f"{p_res['chosen_margin']:.2f}"
                vv_str = f"{p_res['val_violation']*100:.2f}%"
                tv_str = f"{p_res['test_violation']*100:.2f}%"
                reason_short = p_res['drop_reason'][:40] + "..." if len(p_res['drop_reason']) > 40 else p_res['drop_reason']
                print(f"{pol_name:<20} | {'DROPPED':<10} | {m_str:<7} | {vv_str:<10} | {tv_str:<10} | {reason_short:<26} | {'N/A':<26} | N/A")

    print("\n" + "=" * 110)
    print("PRE-REGISTERED HYPOTHESES VERDICTS")
    print("=" * 110)
    print(f"H1 (ML > Own-History Rule at 1% Violation): {'SUPPORTED' if h1_supported else 'NOT SUPPORTED'}")
    print(f"   Details: TIME split ML ({boot_time['best_ml_policy']}) saved {boot_time['best_ml_saved_pct']}% vs Rule ({boot_time['best_rule_policy']}) saved {boot_time['best_rule_saved_pct']}%. Paired diff = {boot_time['paired_diff_mean']}% (CI: {boot_time['paired_diff_ci_95']}%). Overlap = {boot_time['cis_overlap']}.")

    print(f"H2 (Validation Margin Calibratability across 135 checks): {'SUPPORTED' if h2_supported else 'NOT SUPPORTED'}")
    print(f"   Total checks: {h2_summary['total_checks']} (15 core policies x 3 targets x 3 splits)")
    print(f"   Total exceeding 2x test limit: {h2_summary['total_exceeded']}/{h2_summary['total_checks']} ({h2_summary['total_exceeded']/h2_summary['total_checks']*100:.1f}%)")
    print(f"   Policies meeting validation target: {h2_summary['met_val_checks']}/{h2_summary['total_checks']}")
    print(f"   True generalization failures (met val <= target, but test > 2x target): {h2_summary['true_generalization_failures']}/{h2_summary['met_val_checks']} ({h2_summary['true_generalization_failures']/max(h2_summary['met_val_checks'],1)*100:.1f}%)")

    print(f"H3 (ML Advantage Persists on Unseen Apps): {'SUPPORTED' if h3_supported else 'NOT SUPPORTED'}")
    print(f"   Details: APP split paired diff = {boot_app['paired_diff_mean']}% (CI: {boot_app['paired_diff_ci_95']}%); APP+TIME paired diff = {boot_app_time['paired_diff_mean']}% (CI: {boot_app_time['paired_diff_ci_95']}%).")
    print("=" * 110 + "\n")


def generate_final_tables(results_dir: Path, data: Dict[str, Any]) -> None:
    """Generate Markdown tables directly from gate3b.json and save to results/final_tables.md."""
    lines: List[str] = [
        "# Phase 2 — Gate 3b Final Empirical Results & Tables",
        "",
        "Generated automatically by `scripts/gate3b.py` from `results/gate3b.json`. Zero hand-typed numbers.",
        "",
        "---",
        "",
        "## Table 1: Policy Feasibility & Validation Selection at 1.0% Target (Margin Grid up to 4.00)",
        "",
        "| Split | Policy | Feasible on Val? | Margin ($m$) | Val Viol (%) | Val CPU Saved (%) | Test Viol (%) | Test CPU Saved (%) | Exceeds 2x Test? | True Gen Failure? |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    target_res = data["target_results"]
    for s_name in ["TIME", "APP", "APP_TIME"]:
        s_data = target_res[s_name]["target_1pct"]
        for pol, p_info in s_data.items():
            feas_str = "**Yes**" if p_info.get("met_val_target") else "No (Dropped)"
            m_str = f"{p_info['chosen_margin']:.2f}"
            vv_str = f"{p_info['val_violation']*100:.2f}%"
            vs_str = f"{p_info.get('val_saved_pct', 0.0):.2f}%"
            tv_str = f"{p_info['test_violation']*100:.2f}%"
            ts_str = f"{p_info['test_saved_pct']:.2f}%"
            exc_str = "Yes" if p_info.get("exceeds_2x_target") else "No"
            tgf_str = "**Yes**" if p_info.get("true_generalization_failure") else "No"
            lines.append(f"| {s_name} | `{pol}` | {feas_str} | {m_str} | {vv_str} | {vs_str} | {tv_str} | {ts_str} | {exc_str} | {tgf_str} |")

    lines.extend([
        "",
        "---",
        "",
        "## Table 2: Paired 500-Draw App-Level Bootstrap Comparison at 1.0% Target",
        "",
        "| Split | Policy | Status | Margin | Test Viol (%) | Test CPU Saved [95% CI] | Diff vs Best ML [95% CI] | Diff Spans 0? |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    boot_res = data["bootstrap_results"]
    for s_name in ["TIME", "APP", "APP_TIME"]:
        b = boot_res[s_name]
        ml_pol = b["best_ml_policy"]
        ml_line = f"| **{s_name}** | **Best ML: `{ml_pol}`** | **Selected** | **{b['best_ml_margin']:.2f}** | **{b['best_ml_test_viol']*100:.2f}%** | **{b['best_ml_saved_pct']:.2f}% [{b['ml_ci_95'][0]}%, {b['ml_ci_95'][1]}%]** | Baseline Reference | — |"
        lines.append(ml_line)

        for p_label, p_res in b["policy_results"].items():
            pol_name = p_res["policy"]
            if p_res["feasible"]:
                m_str = f"{p_res['chosen_margin']:.2f}"
                tv_str = f"{p_res['test_violation']*100:.2f}%"
                ts_str = f"{p_res['test_saved_pct']:.2f}% [{p_res['saved_ci_95'][0]}%, {p_res['saved_ci_95'][1]}%]"
                diff_str = f"{p_res['diff_mean']:+.2f}% [{p_res['diff_ci_95'][0]:+.2f}%, {p_res['diff_ci_95'][1]:+.2f}%]"
                spans_str = "Yes" if p_res["diff_spans_zero"] else "**No**"
                lines.append(f"| {s_name} | `{pol_name}` | Feasible | {m_str} | {tv_str} | {ts_str} | {diff_str} | {spans_str} |")
            else:
                m_str = f"{p_res['chosen_margin']:.2f}"
                tv_str = f"{p_res['test_violation']*100:.2f}%"
                reason_clean = p_res["drop_reason"].replace("|", "/")
                lines.append(f"| {s_name} | `{pol_name}` | **DROPPED** | {m_str} | {tv_str} | {reason_clean} | N/A | N/A |")

    lines.extend([
        "",
        "---",
        "",
        "## Table 3: Hypothesis 2 (H2) Generalization Breakdown (135 Core Checks)",
        "",
        "### A. Breakdown by Target",
        "",
        "| Target | 2x Test Violation Limit | Total Checks | Exceeded 2x Test Limit | Met Validation Target | True Generalization Failures | True Failure Rate among Feasible |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    h2_sum = data["hypotheses_verdicts"]["H2_validation_margin_calibratability"]["h2_summary_135_checks"]
    for tgt_key, t_info in h2_sum["by_target"].items():
        t_label = "5.0%" if tgt_key == "target_5pct" else ("1.0%" if tgt_key == "target_1pct" else "0.1%")
        limit_str = f"<= {t_info['target_val']*2*100:.1f}%" if t_info['target_val'] >= 0.005 else f"<= {t_info['target_val']*2*100:.2f}%"
        rate_str = f"{t_info['true_failures']/max(t_info['met_val'],1)*100:.1f}%" if t_info['met_val'] > 0 else "N/A"
        lines.append(f"| **{t_label}** | {limit_str} | {t_info['total']} | {t_info['exceeded']} | {t_info['met_val']} | {t_info['true_failures']} | {rate_str} |")

    lines.extend([
        "",
        "### B. Breakdown by Split",
        "",
        "| Split | Total Checks | Exceeded 2x Test Limit | Met Validation Target | True Generalization Failures | True Failure Rate among Feasible |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ])

    for s_name, s_info in h2_sum["by_split"].items():
        rate_str = f"{s_info['true_failures']/max(s_info['met_val'],1)*100:.1f}%" if s_info['met_val'] > 0 else "N/A"
        lines.append(f"| **{s_name}** | {s_info['total']} | {s_info['exceeded']} | {s_info['met_val']} | {s_info['true_failures']} | {rate_str} |")

    lines.extend([
        "",
        "### C. Overall Generalization Summary",
        "",
        f"- **Total Checks**: {h2_sum['total_checks']} (15 core policies x 3 targets x 3 splits)",
        f"- **Raw Test Exceedances (> 2x target)**: {h2_sum['total_exceeded']} / {h2_sum['total_checks']} ({h2_sum['total_exceeded']/h2_sum['total_checks']*100:.1f}%)",
        f"- **Policies Meeting Validation Target**: {h2_sum['met_val_checks']} / {h2_sum['total_checks']} ({h2_sum['met_val_checks']/h2_sum['total_checks']*100:.1f}%)",
        f"- **True Generalization Failures (met val <= target, but test > 2x target)**: **{h2_sum['true_generalization_failures']} / {h2_sum['met_val_checks']}** ({h2_sum['true_generalization_failures']/max(h2_sum['met_val_checks'],1)*100:.1f}%)",
        "",
        "---",
        "",
        "## Table 4: Exceeded Configurations Breakdown",
        "",
        "| Split | Target | Policy | Selected Margin | Val Viol (%) | Test Viol (%) | Exceed Factor | Failure Classification |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |"
    ])

    for exc in h2_sum["exceeded_details"]:
        t_label = "5.0%" if exc["target"] == "target_5pct" else ("1.0%" if exc["target"] == "target_1pct" else "0.1%")
        factor = exc["test_violation"] / exc["target_val"]
        class_str = "**True Generalization Failure** (Met Val Target)" if exc["true_gen_failure"] else "Unmet Validation Target (Failed on Val)"
        lines.append(f"| {exc['split']} | {t_label} | `{exc['policy']}` | {exc['margin']:.2f} | {exc['val_violation']*100:.2f}% | {exc['test_violation']*100:.2f}% | {factor:.2f}x | {class_str} |")

    out_file = results_dir / "final_tables.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated final report tables to {out_file}")


if __name__ == "__main__":
    main()
