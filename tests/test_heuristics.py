"""Unit tests for heuristic placement algorithms."""

import unittest

from allocation.cluster import Cluster
from allocation.heuristics import BestFit, FirstFit, RoundRobin, WorstFit
from allocation.workload import Workload
from datasets.synthetic import generate_workloads


class TestHeuristics(unittest.TestCase):
    """Test suite for FirstFit, BestFit, WorstFit, and RoundRobin allocators."""

    def test_bestfit_chooses_tighter_server_than_worstfit_crafted_example(self) -> None:
        """BestFit chooses the tighter server than WorstFit on a crafted 2-server example."""
        # Server 1: total_cpu=4, total_ram=8
        # Server 2: total_cpu=16, total_ram=32
        specs = [("server-tight", 4, 8), ("server-loose", 16, 32)]
        workload = Workload("wl-crafted", cpu_requirement=2, ram_requirement=4)

        # On server-tight:
        # leftover = (4-2)/4 + (8-4)/8 = 0.5 + 0.5 = 1.0
        # On server-loose:
        # leftover = (16-2)/16 + (32-4)/32 = 0.875 + 0.875 = 1.75

        # BestFit should choose the tighter server (min leftover -> server-tight)
        cluster_bf = Cluster(specs)
        bf = BestFit()
        placed_bf = bf.place(workload, cluster_bf)
        self.assertIsNotNone(placed_bf)
        self.assertEqual(placed_bf.server_id, "server-tight")

        # WorstFit should choose the looser server (max leftover -> server-loose)
        cluster_wf = Cluster(specs)
        wf = WorstFit()
        placed_wf = wf.place(workload, cluster_wf)
        self.assertIsNotNone(placed_wf)
        self.assertEqual(placed_wf.server_id, "server-loose")

    def test_rejects_when_nothing_fits(self) -> None:
        """Allocators reject (return None) when workload cannot fit on any server."""
        specs = [("s1", 4, 8), ("s2", 8, 16)]
        oversized = Workload("wl-huge", cpu_requirement=16, ram_requirement=32)

        for AllocatorClass in [FirstFit, BestFit, WorstFit, RoundRobin]:
            with self.subTest(allocator=AllocatorClass.__name__):
                cluster = Cluster(specs)
                allocator = AllocatorClass()
                result = allocator.place(oversized, cluster)
                self.assertIsNone(result, f"{AllocatorClass.__name__} should return None on SLA violation")
                # Cluster remains completely unallocated
                self.assertEqual(cluster.used_cpu, 0)
                self.assertEqual(cluster.used_ram, 0)

    def test_never_over_allocates_under_load(self) -> None:
        """Under a heavy synthetic workload stream, servers never exceed capacity."""
        specs = [
            ("s1", 8, 16),
            ("s2", 16, 32),
            ("s3", 8, 16),
            ("s4", 32, 64),
        ]
        workloads = generate_workloads(100, seed=99)

        for AllocatorClass in [FirstFit, BestFit, WorstFit, RoundRobin]:
            with self.subTest(allocator=AllocatorClass.__name__):
                cluster = Cluster(specs)
                allocator = AllocatorClass()

                for wl in workloads:
                    allocator.place(wl, cluster)
                    # Invariant: used <= total for every server at all times
                    for server in cluster.servers:
                        self.assertLessEqual(
                            server.used_cpu,
                            server.total_cpu + 1e-9,
                            f"{AllocatorClass.__name__} exceeded CPU on {server.server_id}",
                        )
                        self.assertLessEqual(
                            server.used_ram,
                            server.total_ram + 1e-9,
                            f"{AllocatorClass.__name__} exceeded RAM on {server.server_id}",
                        )

    def test_firstfit_order_preference(self) -> None:
        """FirstFit always picks the first feasible server according to server index."""
        specs = [("s-first", 8, 16), ("s-second", 8, 16)]
        cluster = Cluster(specs)
        ff = FirstFit()
        wl = Workload("wl-1", 2, 4)

        server = ff.place(wl, cluster)
        self.assertIsNotNone(server)
        self.assertEqual(server.server_id, "s-first")

    def test_round_robin_cycles_servers(self) -> None:
        """RoundRobin distributes workloads sequentially across available servers."""
        specs = [("n1", 16, 32), ("n2", 16, 32), ("n3", 16, 32)]
        cluster = Cluster(specs)
        rr = RoundRobin()

        w1 = Workload("w1", 2, 4)
        w2 = Workload("w2", 2, 4)
        w3 = Workload("w3", 2, 4)

        s1 = rr.place(w1, cluster)
        s2 = rr.place(w2, cluster)
        s3 = rr.place(w3, cluster)

        self.assertIsNotNone(s1)
        self.assertIsNotNone(s2)
        self.assertIsNotNone(s3)
        self.assertEqual(s1.server_id, "n1")
        self.assertEqual(s2.server_id, "n2")
        self.assertEqual(s3.server_id, "n3")


if __name__ == "__main__":
    unittest.main()

