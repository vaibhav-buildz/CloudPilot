"""Benchmark dynamic workload trace replay on a heterogeneous server cluster."""

from __future__ import annotations

import csv
import json
import os
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

SAMPLE_TRACE_PATH = PROJECT_ROOT / "datasets" / "sample_alibaba_trace.csv"


def build_heterogeneous_cluster_specs() -> List[Tuple[str, int, int]]:
    """Build specifications for 20 servers: 12 small (8 CPU, 16 GB) + 8 large (32 CPU, 64 GB)."""
    specs: List[Tuple[str, int, int]] = []
    for i in range(12):
        specs.append((f"srv-s-{i+1:02d}", 8, 16))
    for i in range(8):
        specs.append((f"srv-l-{i+1:02d}", 32, 64))
    return specs


def run_trace_benchmark() -> List[SimulationResult]:
    """Execute dynamic trace simulations across placement heuristics."""
    if not SAMPLE_TRACE_PATH.exists():
        print(f"Generating synthetic Alibaba-style trace at {SAMPLE_TRACE_PATH}...")
        raw_workloads = generate_alibaba_style_trace(n=1000, seed=42)
        save_csv_trace(raw_workloads, SAMPLE_TRACE_PATH)

    workloads = load_csv_trace(SAMPLE_TRACE_PATH)
    specs = build_heterogeneous_cluster_specs()
    cluster = Cluster(specs)

    algorithms = [
        FirstFit(),
        BestFit(),
        WorstFit(),
        RoundRobin(),
    ]

    results: List[SimulationResult] = []
    for allocator in algorithms:
        simulator = TraceSimulator(cluster, allocator)
        res = simulator.run(workloads)
        results.append(res)

    return results


def display_results_table(results: List[SimulationResult]) -> None:
    """Print a clean ASCII comparison table of simulation telemetry."""
    header = (
        f"{'Algorithm':<13} | {'Placed':>6} | {'Rejected':>8} | {'Reject %':>8} | "
        f"{'Avg CPU':>7} | {'Avg RAM':>7} | {'Peak CPU':>8} | {'Peak RAM':>8} | "
        f"{'Avg Imb':>7} | {'Peak Srv':>8} | {'CPU Hrs':>7} | {'RAM GB-H':>8}"
    )
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print(" CloudPilot Dynamic Trace Simulation Benchmark Results (Alibaba Trace)")
    print("=" * len(header))
    print(header)
    print(separator)

    for r in results:
        print(
            f"{r.algorithm:<13} | "
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


def save_results(results: List[SimulationResult], results_dir: Union[str, Path] = "results") -> None:
    """Export simulation results to JSON and CSV formats."""
    out_dir = Path(results_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "trace_metrics.json"
    dict_results = [asdict(r) for r in results]
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(dict_results, f, indent=2)

    csv_path = out_dir / "trace_metrics.csv"
    if dict_results:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict_results[0].keys()))
            writer.writeheader()
            writer.writerows(dict_results)

    print(f"Trace metrics successfully saved to:\n  - {json_path}\n  - {csv_path}")


def main() -> None:
    results = run_trace_benchmark()
    display_results_table(results)
    save_results(results)


if __name__ == "__main__":
    main()

