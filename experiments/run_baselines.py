"""Benchmark traditional resource allocation baselines against synthetic workloads."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path when executed directly as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from allocation.cluster import Cluster
from allocation.heuristics import BestFit, FirstFit, RoundRobin, WorstFit
from allocation.workload import Workload
from datasets.synthetic import generate_workloads

NUM_WORKLOADS = 500
RANDOM_SEED = 42


def build_mixed_cluster_specs() -> List[Tuple[str, int, int]]:
    """Build specs for 20 servers: 12 small (8 CPU, 16 GB) and 8 large (32 CPU, 64 GB)."""
    specs: List[Tuple[str, int, int]] = []
    # 12 Small servers
    for i in range(12):
        specs.append((f"srv-small-{i+1:02d}", 8, 16))
    # 8 Large servers
    for i in range(8):
        specs.append((f"srv-large-{i+1:02d}", 32, 64))
    return specs


def run_benchmark() -> List[Dict[str, Any]]:
    """Execute baseline allocation algorithms and collect performance metrics."""
    specs = build_mixed_cluster_specs()
    raw_workloads = generate_workloads(NUM_WORKLOADS, seed=RANDOM_SEED)

    # Sort largest-first for Decreasing heuristics (primary by CPU, secondary by RAM)
    sorted_workloads = sorted(
        raw_workloads,
        key=lambda w: (w.cpu, w.ram),
        reverse=True,
    )

    experiments = [
        ("First-Fit", FirstFit(), raw_workloads),
        ("Best-Fit", BestFit(), raw_workloads),
        ("Worst-Fit", WorstFit(), raw_workloads),
        ("Round-Robin", RoundRobin(), raw_workloads),
        ("First-Fit Decreasing", FirstFit(), sorted_workloads),
        ("Best-Fit Decreasing", BestFit(), sorted_workloads),
    ]

    metrics: List[Dict[str, Any]] = []

    for name, allocator, workload_list in experiments:
        cluster = Cluster(specs)
        placed = 0
        rejected = 0

        for wl in workload_list:
            server = allocator.place(wl, cluster)
            if server is not None:
                placed += 1
            else:
                rejected += 1

        total = placed + rejected
        rejection_rate = (rejected / total * 100.0) if total > 0 else 0.0

        metrics.append(
            {
                "algorithm": name,
                "placed": placed,
                "rejected": rejected,
                "rejection_rate_pct": round(rejection_rate, 2),
                "cpu_util_pct": round(cluster.cpu_utilization, 2),
                "ram_util_pct": round(cluster.ram_utilization, 2),
                "servers_in_use": cluster.servers_in_use,
                "load_imbalance": round(cluster.load_imbalance, 2),
            }
        )

    return metrics


def display_metrics_table(metrics: List[Dict[str, Any]]) -> None:
    """Print an ASCII comparison table of baseline algorithm metrics."""
    header = (
        f"{'Algorithm':<22} | {'Placed':>6} | {'Rejected':>8} | "
        f"{'Reject %':>8} | {'CPU %':>6} | {'RAM %':>6} | {'Active':>6} | {'Imbalance':>9}"
    )
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print(" CloudPilot Baseline Algorithm Benchmark Results")
    print("=" * len(header))
    print(header)
    print(separator)

    for m in metrics:
        print(
            f"{m['algorithm']:<22} | "
            f"{m['placed']:>6} | "
            f"{m['rejected']:>8} | "
            f"{m['rejection_rate_pct']:>7.2f}% | "
            f"{m['cpu_util_pct']:>5.2f}% | "
            f"{m['ram_util_pct']:>5.2f}% | "
            f"{m['servers_in_use']:>6} | "
            f"{m['load_imbalance']:>9.2f}"
        )
    print(separator + "\n")


def save_results(metrics: List[Dict[str, Any]], results_dir: str = "results") -> None:
    """Save metrics to both JSON and CSV files."""
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "baseline_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    csv_path = os.path.join(results_dir, "baseline_metrics.csv")
    fieldnames = [
        "algorithm",
        "placed",
        "rejected",
        "rejection_rate_pct",
        "cpu_util_pct",
        "ram_util_pct",
        "servers_in_use",
        "load_imbalance",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    print(f"Metrics successfully saved to:\n  - {json_path}\n  - {csv_path}")


def main() -> None:
    metrics = run_benchmark()
    display_metrics_table(metrics)
    save_results(metrics)


if __name__ == "__main__":
    main()
