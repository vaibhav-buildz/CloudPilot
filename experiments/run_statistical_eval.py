"""Comprehensive statistical evaluation without data leakage across load levels and cluster scales."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from allocation.cluster import Cluster
from allocation.config import build_heterogeneous_specs
from allocation.heuristics import Allocator, BestFit, FirstFit, RoundRobin, WorstFit
from allocation.simulator import SimulationResult, TraceSimulator
from allocation.workload import Workload
from datasets.trace_loader import generate_alibaba_style_trace
from ml.features import bucketize_trace
from ml.optimizer import PredictiveAllocator, UCBBanditAllocator
from ml.predictor import EWMAForecaster, LinearARForecaster

# Student's t critical value for df = 29 at two-tailed alpha = 0.05
T_CRIT_29 = 2.04523

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class ForcedBurstPredictive(PredictiveAllocator):
    """Ablation: forces burst multiplier to constant 1.0 (no ML forecast modulation)."""
    def _compute_burst_factor(self) -> float:
        return 1.0


def compute_student_t_p_value(t_stat: float, df: int = 29) -> float:
    """Calculates two-tailed p-value for Student's t distribution via numerical integration."""
    t = abs(t_stat)
    l_num = math.lgamma((df + 1) / 2)
    l_den = 0.5 * math.log(df * math.pi) + math.lgamma(df / 2)
    coeff = math.exp(l_num - l_den)

    def pdf(x: float) -> float:
        return coeff * (1.0 + (x * x) / df) ** (-(df + 1) / 2)

    step = 0.002
    x = t
    integral = 0.0
    while x < 40.0:
        integral += pdf(x) * step
        x += step
    return min(1.0, max(0.0, 2.0 * integral))


def train_leak_free_linear_ar() -> LinearARForecaster:
    """Pre-trains LinearARForecaster STRICTLY on independent training seeds (901..903)."""
    train_trace: List[Workload] = []
    for s in (901, 902, 903):
        train_trace.extend(generate_alibaba_style_trace(n=400, seed=s, horizon_seconds=1200.0))
    train_buckets = bucketize_trace(train_trace, window_size=30.0)
    model = LinearARForecaster(lags=4, ridge_lambda=2.0)
    model.fit(train_buckets)
    return model


def get_allocator_registry(trained_linear_ar: LinearARForecaster) -> Dict[str, Callable[[], Allocator]]:
    """Returns factory callables for all 7 primary policies and 4 ablation baselines."""
    return {
        # Core Baselines
        "FirstFit": lambda: FirstFit(),
        "BestFit": lambda: BestFit(),
        "WorstFit": lambda: WorstFit(),
        "RoundRobin": lambda: RoundRobin(),
        # ML / RL Policies
        "Predictive (EWMA)": lambda: PredictiveAllocator(predictor=EWMAForecaster(alpha=0.35, beta=0.1)),
        "Predictive (LinearAR)": lambda: PredictiveAllocator(predictor=trained_linear_ar),
        "UCB-Bandit (RL)": lambda: UCBBanditAllocator(),
        # Ablations
        "Ablation: No Forecast": lambda: ForcedBurstPredictive(predictor=EWMAForecaster(alpha=0.35, beta=0.1)),
        "Ablation: No Tier Pen": lambda: PredictiveAllocator(
            predictor=EWMAForecaster(alpha=0.35, beta=0.1), tier_penalty_weight=0.0
        ),
        "Ablation: No Skew": lambda: PredictiveAllocator(
            predictor=EWMAForecaster(alpha=0.35, beta=0.1), skew_weight=0.0
        ),
        "Ablation: No Sliver Pen": lambda: PredictiveAllocator(
            predictor=EWMAForecaster(alpha=0.35, beta=0.1), sliver_penalty=0.0
        ),
    }


def run_statistical_evaluation() -> Tuple[List[Dict[str, Any]], str]:
    """Runs rigorous 30-seed statistical evaluation across server scales and offered loads."""
    print("=" * 80)
    print(" CloudPilot Step 8: Production Statistical Evaluation (Zero Leakage)")
    print("=" * 80)

    trained_linear_ar = train_leak_free_linear_ar()
    factories = get_allocator_registry(trained_linear_ar)

    # Matrix: Server Scale x Offered Load
    sweeps = [
        # 20 Nodes Sweep (Loads 0.6, 1.0, 1.5)
        (20, 0.6, 600),
        (20, 1.0, 1000),
        (20, 1.5, 1500),
        # Scalability Sweep (Loads 0.6, 1.0, 1.5)
        (50, 0.6, 1500),
        (50, 1.0, 2500),
        (50, 1.5, 3750),
        (100, 0.6, 3000),
        (100, 1.0, 5000),
        (100, 1.5, 7500),
    ]

    all_records: List[Dict[str, Any]] = []
    report_lines: List[str] = [
        "# CloudPilot Step 8: Statistical Evaluation & Robustness Report",
        "",
        "## Executive Summary",
        "",
        "- **Zero Data Leakage**: `LinearARForecaster` was pre-trained solely on disjoint training traces (seeds 901–903) and evaluated strictly out-of-sample.",
        "- **30-Seed Statistical Rigor**: Evaluated across 30 independent pseudo-random evaluation seeds (1..30).",
        "- **Load & Scale Coverage**: 9 total configurations spanning 20, 50, and 100 heterogeneous servers under low (0.6x), standard (1.0x), and surge (1.5x) offered loads.",
        "",
    ]

    for servers, load_mult, n_jobs in sweeps:
        config_name = f"{servers}_nodes_load_{load_mult}x"
        print(f"\n---> Evaluating {config_name} ({servers} servers, {n_jobs} jobs, load {load_mult}x) across 30 seeds...")
        specs = build_heterogeneous_specs(servers)

        # Map: policy_name -> list of 30 placed_counts
        policy_placements: Dict[str, List[int]] = {name: [] for name in factories}

        t_start = time.perf_counter()
        for seed in range(1, 31):
            trace = generate_alibaba_style_trace(n=n_jobs, seed=seed, horizon_seconds=3600.0)
            for p_name, factory in factories.items():
                alloc = factory()
                cluster = Cluster(specs)
                sim = TraceSimulator(cluster=cluster, allocator=alloc)
                res = sim.run(trace, reset_cluster=True)
                policy_placements[p_name].append(res.placed_count)

        elapsed = time.perf_counter() - t_start
        print(f"     Finished 30 seeds in {elapsed:.2f}s ({elapsed/330*1000:.1f} ms/run)")

        ff_counts = policy_placements["FirstFit"]
        n_seeds = len(ff_counts)
        sqrt_n = math.sqrt(n_seeds)

        report_lines.append(f"### Configuration: `{config_name}` ({servers} Servers, {n_jobs} Jobs)")
        report_lines.append("")
        report_lines.append(
            "| Policy | Mean Placed | Std Dev | 95% Confidence Interval | Mean Diff vs FirstFit | Paired t-stat | Significant? (p < 0.05) |"
        )
        report_lines.append(
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
        )

        for p_name, counts in policy_placements.items():
            mean_c = statistics.mean(counts)
            std_c = statistics.stdev(counts) if len(counts) > 1 else 0.0
            se_c = std_c / sqrt_n
            ci_half = T_CRIT_29 * se_c
            ci_str = f"[{mean_c - ci_half:.1f}, {mean_c + ci_half:.1f}]"

            if p_name == "FirstFit":
                diff_mean = 0.0
                t_stat = 0.0
                p_val = 1.0
                sig_str = "Baseline"
            else:
                diffs = [counts[i] - ff_counts[i] for i in range(n_seeds)]
                diff_mean = statistics.mean(diffs)
                diff_std = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
                diff_se = diff_std / sqrt_n
                t_stat = diff_mean / diff_se if diff_se > 1e-9 else 0.0
                p_val = compute_student_t_p_value(t_stat, df=n_seeds - 1)
                sig_str = f"**YES** (p={p_val:.4f})" if abs(t_stat) >= T_CRIT_29 else f"No (p={p_val:.4f})"

            rec = {
                "config": config_name,
                "servers": servers,
                "offered_load": load_mult,
                "total_workloads": n_jobs,
                "policy": p_name,
                "mean_placed": round(mean_c, 2),
                "std_placed": round(std_c, 2),
                "ci_lower": round(mean_c - ci_half, 2),
                "ci_upper": round(mean_c + ci_half, 2),
                "diff_vs_firstfit": round(diff_mean, 2),
                "paired_t_stat": round(t_stat, 3),
                "p_value": round(p_val, 4),
                "statistically_significant": abs(t_stat) >= T_CRIT_29,
            }
            all_records.append(rec)

            diff_str = f"{diff_mean:+.2f}" if p_name != "FirstFit" else "(baseline)"
            t_str = f"{t_stat:+.3f}" if p_name != "FirstFit" else "-"
            report_lines.append(
                f"| **{p_name}** | {mean_c:.1f} | {std_c:.2f} | {ci_str} | {diff_str} | {t_str} | {sig_str} |"
            )
        report_lines.append("")

    # Export structured JSON
    json_path = RESULTS_DIR / "statistical_eval.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    print(f"\n[Artifact Saved] JSON metrics -> {json_path}")

    # Export structured CSV
    csv_path = RESULTS_DIR / "statistical_eval.csv"
    if all_records:
        keys = list(all_records[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_records)
    print(f"[Artifact Saved] CSV metrics  -> {csv_path}")

    # Export Markdown Report
    report_path = RESULTS_DIR / "statistical_eval_report.md"
    report_text = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[Artifact Saved] Markdown Report -> {report_path}")

    return all_records, report_text


if __name__ == "__main__":
    run_statistical_evaluation()
