"""Unit tests for ML-guided PredictiveAllocator and UCBBanditAllocator."""

import unittest

from allocation.cluster import Cluster
from allocation.heuristics import BestFit, FirstFit
from allocation.simulator import TraceSimulator
from allocation.workload import Workload
from datasets.trace_loader import generate_alibaba_style_trace
from ml.optimizer import PredictiveAllocator, UCBBanditAllocator
from ml.predictor import EWMAForecaster, LinearARForecaster


class TestPredictiveAllocator(unittest.TestCase):
    """Test suite for PredictiveAllocator features (tier protection, skew alignment, streaming updates)."""

    def test_tier_aware_whale_protection(self) -> None:
        """Small jobs are routed to small nodes, preserving large nodes for subsequent large jobs."""
        # Large server listed first, small server listed second
        specs = [("srv-large", 32, 64), ("srv-small", 8, 16)]
        cluster = Cluster(specs)

        # Allocate small job (2 CPU, 4 GB)
        allocator = PredictiveAllocator(small_cpu_threshold=8.0, small_ram_threshold=16.0)
        small_job = Workload("w-small", cpu_requirement=2, ram_requirement=4, arrival_time=0.0)

        chosen_server = allocator.place(small_job, cluster)
        self.assertIsNotNone(chosen_server)
        # Must choose small server despite large server being listed first and idle
        self.assertEqual(chosen_server.server_id, "srv-small")

        # Now allocate a large job that can only fit on the large server
        large_job = Workload("w-large", cpu_requirement=16, ram_requirement=32, arrival_time=1.0)
        chosen_large = allocator.place(large_job, cluster)
        self.assertIsNotNone(chosen_large)
        self.assertEqual(chosen_large.server_id, "srv-large")

        # Both placed successfully
        self.assertEqual(cluster.servers_in_use, 2)

    def test_cross_dimensional_skew_alignment(self) -> None:
        """RAM-heavy workload is placed on the RAM-rich server to minimize residual dimension skew."""
        # Two equal-capacity servers (16 CPU, 32 GB)
        # Server A has used (4 CPU, 22 GB) -> remaining: 12 CPU, 10 GB
        # Server B has used (10 CPU, 16 GB) -> remaining: 6 CPU, 16 GB
        cluster = Cluster([("srv-A", 16, 32), ("srv-B", 16, 32)])
        cluster[0].allocate(4, 22)
        cluster[1].allocate(10, 16)

        allocator = PredictiveAllocator(skew_weight=2.0, packing_weight=0.1, idle_wakeup_penalty=0.0)
        # RAM-heavy job: 2 CPU, 8 GB
        ram_heavy_job = Workload("w-ram-heavy", cpu_requirement=2, ram_requirement=8, arrival_time=0.0)

        # On Server A: rem_cpu = 10/16 = 0.625, rem_ram = 2/32 = 0.0625 -> skew = 0.5625
        # On Server B: rem_cpu = 4/16 = 0.25, rem_ram = 8/32 = 0.25 -> skew = 0.0 (perfect balance)
        chosen = allocator.place(ram_heavy_job, cluster)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.server_id, "srv-B")

    def test_online_streaming_forecast_update(self) -> None:
        """PredictiveAllocator rolls windows and updates demand forecasts as time advances."""
        allocator = PredictiveAllocator(window_size=10.0)
        cluster = Cluster([("srv-1", 64, 128)])

        # Arrivals at t=0, 5, 12, 25
        w1 = Workload("w1", 4, 8, arrival_time=0.0)
        w2 = Workload("w2", 4, 8, arrival_time=5.0)
        w3 = Workload("w3", 4, 8, arrival_time=12.0)
        w4 = Workload("w4", 4, 8, arrival_time=25.0)

        allocator.place(w1, cluster)
        allocator.place(w2, cluster)
        self.assertEqual(len(allocator._completed_buckets), 0)

        allocator.place(w3, cluster)
        # Crossed t=10.0 window boundary -> bucket 0 finalized
        self.assertEqual(len(allocator._completed_buckets), 1)
        self.assertEqual(allocator._completed_buckets[0].arrival_count, 2)

        allocator.place(w4, cluster)
        # Crossed t=20.0 window boundary -> bucket 1 finalized
        self.assertEqual(len(allocator._completed_buckets), 2)


class TestUCBBanditAllocator(unittest.TestCase):
    """Test suite for Contextual UCB Multi-Armed Bandit allocator."""

    def test_bandit_initialization_and_arm_execution(self) -> None:
        """Bandit initializes with candidate policy arms and allocates feasible workloads."""
        bandit = UCBBanditAllocator(exploration_c=0.5)
        cluster = Cluster([("srv-1", 16, 32), ("srv-2", 32, 64)])

        w = Workload("w-1", 4, 8, arrival_time=0.0)
        server = bandit.place(w, cluster)
        self.assertIsNotNone(server)
        self.assertIn(server.server_id, ["srv-1", "srv-2"])
        self.assertEqual(cluster.servers_in_use, 1)

    def test_bandit_rejects_when_impossible(self) -> None:
        """Bandit returns None and penalizes arms when a workload exceeds all cluster capacity."""
        bandit = UCBBanditAllocator()
        cluster = Cluster([("srv-small", 4, 8)])
        oversized = Workload("w-huge", 32, 64)

        result = bandit.place(oversized, cluster)
        self.assertIsNone(result)
        self.assertEqual(cluster.servers_in_use, 0)


class TestOptimizationIntegration(unittest.TestCase):
    """Integration test verifying trace simulation execution and capacity invariant enforcement."""

    def test_trace_simulation_invariants_and_placement(self) -> None:
        """PredictiveAllocator and UCBBanditAllocator maintain capacity invariants on dynamic traces."""
        specs = [("srv-s", 8, 16), ("srv-l", 32, 64)]
        cluster = Cluster(specs)

        # 20 workloads from synthetic Alibaba trace
        workloads = generate_alibaba_style_trace(n=20, seed=42)

        for allocator in [PredictiveAllocator(), UCBBanditAllocator()]:
            with self.subTest(allocator=allocator.__class__.__name__):
                simulator = TraceSimulator(cluster, allocator)
                result = simulator.run(workloads)

                self.assertEqual(result.total_workloads, 20)
                self.assertGreater(result.placed_count, 0)
                # Invariant: used <= total for every server at all times
                for s in cluster.servers:
                    self.assertLessEqual(s.used_cpu, s.total_cpu + 1e-9)
                    self.assertLessEqual(s.used_ram, s.total_ram + 1e-9)


if __name__ == "__main__":
    unittest.main()

