"""Unit tests for the multi-scenario and multi-scale benchmark suite."""

import unittest
from pathlib import Path

from allocation.cluster import Cluster
from allocation.heuristics import FirstFit
from allocation.simulator import TraceSimulator
from datasets.trace_loader import (
    generate_alibaba_style_trace,
    load_csv_trace,
    save_csv_trace,
)
from experiments.benchmark_suite import (
    BenchmarkRecord,
    build_allocator_factories,
    build_heterogeneous_specs,
    compute_sla_efficiency_score,
    evaluate_scenario,
)
from ml.optimizer import PredictiveAllocator
from ml.predictor import EWMAForecaster, LinearARForecaster

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_TRACE_PATH = PROJECT_ROOT / "datasets" / "sample_alibaba_trace.csv"


class TestBenchmarkSuite(unittest.TestCase):
    """Test suite for benchmark suite generator, scoring, and telemetry invariants."""

    def test_build_heterogeneous_specs(self) -> None:
        """build_heterogeneous_specs creates 60% small and 40% large server specifications."""
        specs_20 = build_heterogeneous_specs(20)
        self.assertEqual(len(specs_20), 20)
        num_small = sum(1 for name, cpu, ram in specs_20 if cpu <= 8.0)
        num_large = sum(1 for name, cpu, ram in specs_20 if cpu >= 32.0)
        self.assertEqual(num_small, 12)
        self.assertEqual(num_large, 8)

        # Scale to 50
        specs_50 = build_heterogeneous_specs(50)
        self.assertEqual(len(specs_50), 50)
        self.assertEqual(sum(1 for _, cpu, _ in specs_50 if cpu <= 8.0), 30)
        self.assertEqual(sum(1 for _, cpu, _ in specs_50 if cpu >= 32.0), 20)

    def test_sla_efficiency_score_bounds(self) -> None:
        """compute_sla_efficiency_score is bounded in [0.0, 100.0]."""
        specs = build_heterogeneous_specs(10)
        cluster = Cluster(specs)
        sim = TraceSimulator(cluster, FirstFit())
        workloads = generate_alibaba_style_trace(n=10, seed=99)
        res = sim.run(workloads)

        score = compute_sla_efficiency_score(res)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_evaluate_scenario_returns_full_telemetry(self) -> None:
        """evaluate_scenario evaluates all 7 policies and records positive timing profiles."""
        workloads = generate_alibaba_style_trace(n=15, seed=77)
        records = evaluate_scenario("test_mini_scenario", workloads, cluster_size=10)

        self.assertEqual(len(records), 7)
        for r in records:
            self.assertIsInstance(r, BenchmarkRecord)
            self.assertEqual(r.scenario, "test_mini_scenario")
            self.assertEqual(r.cluster_size, 10)
            self.assertEqual(r.total_workloads, 15)
            self.assertGreater(r.total_runtime_ms, 0.0)
            self.assertGreater(r.us_per_workload, 0.0)
            self.assertGreaterEqual(r.sla_efficiency_score, 0.0)
            self.assertLessEqual(r.sla_efficiency_score, 100.0)

    def test_predictive_beats_or_matches_firstfit_on_alibaba_trace(self) -> None:
        """PredictiveAllocator achieves placed_count >= FirstFit (>= 937) on the standard Alibaba trace."""
        if not SAMPLE_TRACE_PATH.exists():
            raw_alibaba = generate_alibaba_style_trace(n=1000, seed=42)
            save_csv_trace(raw_alibaba, SAMPLE_TRACE_PATH)

        workloads = load_csv_trace(SAMPLE_TRACE_PATH)
        specs = build_heterogeneous_specs(20)

        # Baseline: FirstFit
        res_ff = TraceSimulator(Cluster(specs), FirstFit()).run(workloads)
        self.assertEqual(res_ff.placed_count, 937)

        # Predictive EWMA
        alloc_ewma = PredictiveAllocator(predictor=EWMAForecaster(alpha=0.35, beta=0.1))
        res_ewma = TraceSimulator(Cluster(specs), alloc_ewma).run(workloads)
        self.assertGreaterEqual(res_ewma.placed_count, res_ff.placed_count)
        self.assertEqual(res_ewma.placed_count, 939)

        # Predictive LinearAR
        alloc_ar = PredictiveAllocator()
        res_ar = TraceSimulator(Cluster(specs), alloc_ar).run(workloads)
        self.assertGreaterEqual(res_ar.placed_count, res_ff.placed_count)
        self.assertEqual(res_ar.placed_count, 939)


if __name__ == "__main__":
    unittest.main()

