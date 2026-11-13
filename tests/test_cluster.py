"""Unit tests for the Cluster model."""

import statistics
import unittest

from allocation.cluster import Cluster
from allocation.workload import Workload


class TestCluster(unittest.TestCase):
    """Test suite for Cluster aggregate metrics and lifecycle."""

    def setUp(self) -> None:
        self.specs = [
            ("node-1", 8, 16),
            ("node-2", 16, 32),
            ("node-3", 32, 64),
        ]
        self.cluster = Cluster(self.specs)

    def test_cluster_initialization(self) -> None:
        """Cluster initializes with expected idle capacities and server count."""
        self.assertEqual(len(self.cluster), 3)
        self.assertEqual(self.cluster.total_cpu, 56)
        self.assertEqual(self.cluster.total_ram, 112)
        self.assertEqual(self.cluster.used_cpu, 0)
        self.assertEqual(self.cluster.used_ram, 0)
        self.assertAlmostEqual(self.cluster.cpu_utilization, 0.0)
        self.assertAlmostEqual(self.cluster.ram_utilization, 0.0)
        self.assertEqual(self.cluster.servers_in_use, 0)
        self.assertAlmostEqual(self.cluster.load_imbalance, 0.0)

    def test_cluster_utilization_metrics(self) -> None:
        """Cluster-wide CPU and RAM utilization percentages update accurately."""
        # Allocate to node-1: 4/8 CPU (50%), 8/16 RAM (50%) -> avg 50%
        self.cluster[0].allocate(4, 8)
        # Allocate to node-2: 8/16 CPU (50%), 16/32 RAM (50%) -> avg 50%
        self.cluster[1].allocate(8, 16)

        self.assertEqual(self.cluster.used_cpu, 12)
        self.assertEqual(self.cluster.used_ram, 24)
        self.assertAlmostEqual(self.cluster.cpu_utilization, (12 / 56) * 100.0)
        self.assertAlmostEqual(self.cluster.ram_utilization, (24 / 112) * 100.0)
        self.assertEqual(self.cluster.servers_in_use, 2)
        # Both active servers have identical 50% utilization -> imbalance is 0.0
        self.assertAlmostEqual(self.cluster.load_imbalance, 0.0)

    def test_load_imbalance_calculation(self) -> None:
        """load_imbalance matches population standard deviation across active servers."""
        # node-1: 100% CPU, 100% RAM -> avg = 100.0%
        self.cluster[0].allocate(8, 16)
        # node-2: 0% CPU, 50% RAM -> avg = 25.0%
        self.cluster[1].allocate(Workload("w-ram", 0.001, 16))

        # Expected population std-dev of [100.0, 25.0] (ignoring idle node-3)
        expected_imbalance = statistics.pstdev([100.0, (self.cluster[1].cpu_utilization + self.cluster[1].ram_utilization) / 2.0])
        self.assertAlmostEqual(self.cluster.load_imbalance, expected_imbalance, places=4)

    def test_cluster_reset_restores_empty_state(self) -> None:
        """Cluster.reset() restores an empty state across all nodes."""
        # Saturate nodes
        self.cluster[0].allocate(8, 16)
        self.cluster[1].allocate(16, 32)
        self.cluster[2].allocate(32, 64)

        self.assertEqual(self.cluster.servers_in_use, 3)
        self.assertAlmostEqual(self.cluster.cpu_utilization, 100.0)

        # Reset cluster
        self.cluster.reset()

        self.assertEqual(len(self.cluster), 3)
        self.assertEqual(self.cluster.servers_in_use, 0)
        self.assertEqual(self.cluster.used_cpu, 0)
        self.assertEqual(self.cluster.used_ram, 0)
        self.assertAlmostEqual(self.cluster.cpu_utilization, 0.0)
        self.assertAlmostEqual(self.cluster.ram_utilization, 0.0)
        self.assertAlmostEqual(self.cluster.load_imbalance, 0.0)

        for server in self.cluster:
            self.assertEqual(server.used_cpu, 0)
            self.assertEqual(server.used_ram, 0)
            self.assertEqual(server.available_cpu(), server.total_cpu)
            self.assertEqual(server.available_ram(), server.total_ram)


if __name__ == "__main__":
    unittest.main()

