"""Comprehensive multi-scenario and multi-scale performance benchmarking engine."""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from allocation.cluster import Cluster
from allocation.heuristics import Allocator, BestFit, FirstFit, RoundRobin, WorstFit
from allocation.simulator import SimulationResult, TraceSimulator
from allocation.workload import Workload
from datasets.trace_loader import generate_alibaba_style_trace
from ml.features import bucketize_trace
from ml.optimizer import PredictiveAllocator, UCBBanditAllocator
from ml.predictor import EWMAForecaster, LinearARForecaster


@dataclass(frozen=True)
class BenchmarkRecord:
    """Extended telemetry record combining simulation metrics and scheduler runtime profiling."""

    scenario: str
    cluster_size: int
    algorithm: str
    total_workloads: int
    placed_count: int
    rejected_count: int
    rejection_rate_pct: float
    avg_cpu_utilization_pct: float
    avg_ram_utilization_pct: float
    peak_cpu_utilization_pct: float
    peak_ram_utilization_pct: float
    avg_load_imbalance: float
    peak_active_servers: int
    allocated_cpu_hours: float
    allocated_ram_gb_hours: float
    total_runtime_ms: float
    us_per_workload: float
    sla_efficiency_score: float


def build_heterogeneous_specs(num_servers: int = 20) -> List[Tuple[str, float, float]]:
    """Builds a 60% small (8 CPU, 16 GB) / 40% large (32 CPU, 64 GB) heterogeneous cluster."""
    num_small = int(round(num_servers * 0.6))
    num_large = max(0, num_servers - num_small)
    specs: List[Tuple[str, float, float]] = []
    for i in range(num_small):
        specs.append((f"small-{i:03d}", 8.0, 16.0))
    for i in range(num_large):
        specs.append((f"large-{i:03d}", 32.0, 64.0))
    return specs


def compute_sla_efficiency_score(sim: SimulationResult) -> float:
    """Computes a composite [0, 100] research score balancing SLA acceptance (60%),
    RAM utilization (25%), and load balance (15%).
    """
    acceptance_rate = 100.0 - sim.rejection_rate_pct
    balance_score = max(0.0, 100.0 - sim.avg_load_imbalance * 2.5)
    score = (
        0.60 * acceptance_rate
        + 0.25 * sim.avg_ram_utilization_pct
        + 0.15 * balance_score
    )
    return round(score, 2)


def build_allocator_factories(
    warmup_workloads: List[Workload],
) -> List[Tuple[str, Callable[[], Allocator]]]:
    """Creates fresh allocator factory callables, pre-fitting LinearAR on warmup buckets."""
    buckets = bucketize_trace(warmup_workloads, window_size=30.0)
    ar_model = LinearARForecaster(lags=4, ridge_lambda=2.0)
    if len(buckets) > 6:
        ar_model.fit(buckets)

    return [
        ("FirstFit", lambda: FirstFit()),
        ("BestFit", lambda: BestFit()),
        ("WorstFit", lambda: WorstFit()),
        ("RoundRobin", lambda: RoundRobin()),
        ("Predictive (EWMA)", lambda: PredictiveAllocator(predictor=EWMAForecaster(alpha=0.35, beta=0.1))),
        ("Predictive (LinearAR)", lambda: PredictiveAllocator(predictor=ar_model)),
        ("UCB-Bandit (RL)", lambda: UCBBanditAllocator()),
    ]


def evaluate_scenario(
    scenario_name: str,
    workloads: List[Workload],
    cluster_size: int = 20,
) -> List[BenchmarkRecord]:
    """Runs all 7 scheduling policies on a scenario and profiles execution latency."""
    specs = build_heterogeneous_specs(cluster_size)
    factories = build_allocator_factories(workloads)
    records: List[BenchmarkRecord] = []

    for alg_name, factory in factories:
        cluster = Cluster(specs)
        allocator = factory()
        sim = TraceSimulator(cluster=cluster, allocator=allocator)

        t0 = time.perf_counter_ns()
        res = sim.run(workloads, reset_cluster=True)
        elapsed_ns = time.perf_counter_ns() - t0

        runtime_ms = round(elapsed_ns / 1_000_000.0, 3)
        us_per_job = round((elapsed_ns / 1_000.0) / max(len(workloads), 1), 3)
        eff_score = compute_sla_efficiency_score(res)

        records.append(
            BenchmarkRecord(
                scenario=scenario_name,
                cluster_size=cluster_size,
                algorithm=alg_name,
                total_workloads=res.total_workloads,
                placed_count=res.placed_count,
                rejected_count=res.rejected_count,
                rejection_rate_pct=res.rejection_rate_pct,
                avg_cpu_utilization_pct=res.avg_cpu_utilization_pct,
                avg_ram_utilization_pct=res.avg_ram_utilization_pct,
                peak_cpu_utilization_pct=res.peak_cpu_utilization_pct,
                peak_ram_utilization_pct=res.peak_ram_utilization_pct,
                avg_load_imbalance=res.avg_load_imbalance,
                peak_active_servers=res.peak_active_servers,
                allocated_cpu_hours=res.allocated_cpu_hours,
                allocated_ram_gb_hours=res.allocated_ram_gb_hours,
                total_runtime_ms=runtime_ms,
                us_per_workload=us_per_job,
                sla_efficiency_score=eff_score,
            )
        )
    return records

