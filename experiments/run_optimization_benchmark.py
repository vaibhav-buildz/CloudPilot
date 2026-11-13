"""Comprehensive benchmark comparing Intelligent ML Optimizers against Baseline Heuristics."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is on sys.path when executed directly as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from allocation.cluster import Cluster
from allocation.heuristics import BestFit, FirstFit, RoundRobin, WorstFit
from allocation.simulator import SimulationResult, TraceSimulator
from datasets.trace_loader import (
    generate_alibaba_style_trace,
    load_csv_trace,
    save_csv_trace,
)
from ml.features import bucketize_trace
from ml.optimizer import PredictiveAllocator, UCBBanditAllocator
from ml.predictor import EWMAForecaster, LinearARForecaster

SAMPLE_TRACE_PATH = PROJECT_ROOT / "datasets" / "sample_alibaba_trace.csv"


def build_heterogeneous_cluster_specs() -> List[Tuple[str, int, int]]:
    """Build specifications for 20 servers: 12 small (8 CPU, 16 GB) + 8 large (32 CPU, 64 GB)."""
    specs: List[Tuple[str, int, int]] = []
    for i in range(12):
        specs.append((f"srv-s-{i+1:02d}", 8, 16))
    for i in range(8):
        specs.append((f"srv-l-{i+1:02d}", 32, 64))
    return specs


def run_optimization_benchmark() -> List[SimulationResult]:
    """Execute dynamic trace simulations comparing all baseline and ML/RL algorithms."""
    if not SAMPLE_TRACE_PATH.exists():
        print(f"Generating synthetic Alibaba-style trace at {SAMPLE_TRACE_PATH}...")
        raw_workloads = generate_alibaba_style_trace(n=1000, seed=42)
        save_csv_trace(raw_workloads, SAMPLE_TRACE_PATH)

    workloads = load_csv_trace(SAMPLE_TRACE_PATH)
    specs = build_heterogeneous_cluster_specs()
    cluster = Cluster(specs)

    # Pre-train LinearARForecaster on historical warm-up buckets
    buckets = bucketize_trace(workloads, window_size=30.0)
    warmup_buckets = buckets[: max(int(len(buckets) * 0.4), 10)]
    fitted_linear_ar = LinearARForecaster(lags=4, ridge_lambda=2.0).fit(warmup_buckets)

    algorithms = [
        ("FirstFit", FirstFit()),
        ("BestFit", BestFit()),
        ("WorstFit", WorstFit()),
        ("RoundRobin", RoundRobin()),
        (
            "Predictive (EWMA)",
            PredictiveAllocator(predictor=EWMAForecaster(alpha=0.35, beta=0.1)),
        ),
        (
            "Predictive (LinearAR)",
            PredictiveAllocator(predictor=fitted_linear_ar),
        ),
        ("UCB-Bandit (RL)", UCBBanditAllocator()),
    ]

    results: List[SimulationResult] = []
    for display_name, allocator in algorithms:
        simulator = TraceSimulator(cluster, allocator)
        raw_res = simulator.run(workloads)
        # Use display_name for clean reporting
        clean_res = SimulationResult(
            algorithm=display_name,
            total_workloads=raw_res.total_workloads,
            placed_count=raw_res.placed_count,
            rejected_count=raw_res.rejected_count,
            rejection_rate_pct=raw_res.rejection_rate_pct,
            avg_cpu_utilization_pct=raw_res.avg_cpu_utilization_pct,
            avg_ram_utilization_pct=raw_res.avg_ram_utilization_pct,
            peak_cpu_utilization_pct=raw_res.peak_cpu_utilization_pct,
            peak_ram_utilization_pct=raw_res.peak_ram_utilization_pct,
            avg_load_imbalance=raw_res.avg_load_imbalance,
            peak_active_servers=raw_res.peak_active_servers,
            allocated_cpu_hours=raw_res.allocated_cpu_hours,
            allocated_ram_gb_hours=raw_res.allocated_ram_gb_hours,
            simulation_duration_sec=raw_res.simulation_duration_sec,
        )
        results.append(clean_res)

    return results


def display_results_table(results: List[SimulationResult]) -> None:
    """Print an aligned ASCII comparison table for the optimization benchmarks."""
    header = (
        f"{'Algorithm / Policy':<23} | {'Placed':>6} | {'Rejected':>8} | {'Reject %':>8} | "
        f"{'Avg CPU':>7} | {'Avg RAM':>7} | {'Peak CPU':>8} | {'Peak RAM':>8} | "
        f"{'Avg Imb':>7} | {'Peak Srv':>8} | {'CPU Hrs':>7} | {'RAM GB-H':>8}"
    )
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print(" CloudPilot Intelligent Resource Allocation & ML Optimization Benchmark")
    print("=" * len(header))
    print(header)
    print(separator)

    for r in results:
        print(
            f"{r.algorithm:<23} | "
            f"{r.placed_count:>6} | "
            f"{r.rejected_count:>8} | "
            f"{r.rejection_rate_pct:>7.2f}% | "
            f"{r.avg_cpu_utilization_pct:>6.2f}% | "
            f"{r.avg_ram_utilization_pct:>6.2f}% | "
            f"{r.peak_cpu_utilization_pct:>7.2f}% | "
            f"{r.peak_ram_utilization_pct:>7.2f}% | "
            f"{r.avg_load_imbalance:>7.2f} | "
            f"{r.peak_active_servers:>8} | "
            f"{r.allocated_cpu_hours:>7.2f} | "
            f"{r.allocated_ram_gb_hours:>8.2f}"
        )
    print(separator + "\n")


def save_results(results: List[SimulationResult], results_dir: str = "results") -> None:
    """Export benchmark telemetry to JSON and CSV formats."""
    out_dir = PROJECT_ROOT / results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "optimization_metrics.json"
    dict_results = [asdict(r) for r in results]
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(dict_results, f, indent=2)

    csv_path = out_dir / "optimization_metrics.csv"
    if dict_results:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict_results[0].keys()))
            writer.writeheader()
            writer.writerows(dict_results)

    print(f"Optimization metrics successfully saved to:\n  - {json_path}\n  - {csv_path}")


def main() -> None:
    results = run_optimization_benchmark()
    display_results_table(results)
    save_results(results)


if __name__ == "__main__":
    main()

