"""Unit tests for workload temporal dynamics, trace loader, and discrete-event TraceSimulator."""

import tempfile
import unittest
from pathlib import Path

from allocation.cluster import Cluster
from allocation.exceptions import InvalidResourceError
from allocation.heuristics import BestFit, FirstFit, RoundRobin, WorstFit
from allocation.simulator import SimulationResult, TraceSimulator
from allocation.workload import Workload
from datasets.trace_loader import (
    generate_alibaba_style_trace,
    load_csv_trace,
    save_csv_trace,
)


class TestWorkloadTemporalAttributes(unittest.TestCase):
    """Test suite for Workload temporal validation and properties."""

    def test_valid_temporal_initialization(self) -> None:
        """Workload properly stores arrival_time, duration, and computes completion_time."""
        w = Workload("w1", cpu_requirement=2, ram_requirement=4, arrival_time=15.5, duration=30.0)
        self.assertEqual(w.arrival_time, 15.5)
        self.assertEqual(w.duration, 30.0)
        self.assertEqual(w.completion_time, 45.5)

    def test_default_temporal_values(self) -> None:
        """Default temporal parameters remain backward-compatible (arrival=0.0, duration=None)."""
        w = Workload("w-default", 4, 8)
        self.assertEqual(w.arrival_time, 0.0)
        self.assertIsNone(w.duration)
        self.assertIsNone(w.completion_time)

    def test_invalid_arrival_time(self) -> None:
        """Negative values, booleans, and non-numeric arrival times are rejected."""
        invalid_arrivals = [-0.01, -10.0, True, False, "10", None, float("nan"), float("inf")]
        for bad_arrival in invalid_arrivals:
            with self.subTest(bad_arrival=bad_arrival):
                with self.assertRaises(InvalidResourceError):
                    Workload("w-bad", 2, 4, arrival_time=bad_arrival)  # type: ignore

    def test_invalid_duration(self) -> None:
        """Zero, negative, boolean, and non-numeric durations are rejected."""
        invalid_durations = [0.0, -1.0, -0.5, True, False, "30", float("nan"), float("inf")]
        for bad_dur in invalid_durations:
            with self.subTest(bad_dur=bad_dur):
                with self.assertRaises(InvalidResourceError):
                    Workload("w-bad", 2, 4, duration=bad_dur)  # type: ignore


class TestTraceSimulator(unittest.TestCase):
    """Test suite for discrete-event simulation of dynamic workload lifecycles."""

    def test_sequential_jobs_on_single_server_reuse_capacity(self) -> None:
        """Jobs arriving consecutively at the exact completion time reuse released resources.

        Single server: 8 CPU, 16 GB.
        Job 1: 8 CPU, 16 GB at t=0 for 10s (departs at t=10).
        Job 2: 8 CPU, 16 GB at t=10 for 10s (arrives at t=10, departs at t=20).
        Both must be placed successfully, and cluster must be idle at end.
        """
        cluster = Cluster([("srv-1", 8, 16)])
        allocator = FirstFit()
        simulator = TraceSimulator(cluster, allocator)

        w1 = Workload("job-1", cpu_requirement=8, ram_requirement=16, arrival_time=0.0, duration=10.0)
        w2 = Workload("job-2", cpu_requirement=8, ram_requirement=16, arrival_time=10.0, duration=10.0)

        result = simulator.run([w1, w2])

        self.assertEqual(result.total_workloads, 2)
        self.assertEqual(result.placed_count, 2)
        self.assertEqual(result.rejected_count, 0)
        self.assertAlmostEqual(result.rejection_rate_pct, 0.0)
        self.assertEqual(result.peak_active_servers, 1)
        self.assertAlmostEqual(result.simulation_duration_sec, 20.0)

        # Confirm cluster is completely idle after simulation completes
        self.assertEqual(cluster.used_cpu, 0)
        self.assertEqual(cluster.used_ram, 0)
        self.assertEqual(cluster.servers_in_use, 0)

    def test_overlapping_jobs_exceeding_capacity_are_rejected(self) -> None:
        """When an incoming job overlaps with an active full-capacity job, it is rejected."""
        cluster = Cluster([("srv-1", 8, 16)])
        simulator = TraceSimulator(cluster, BestFit())

        w1 = Workload("job-1", cpu_requirement=8, ram_requirement=16, arrival_time=0.0, duration=10.0)
        w2 = Workload("job-2", cpu_requirement=8, ram_requirement=16, arrival_time=5.0, duration=10.0)

        result = simulator.run([w1, w2])

        self.assertEqual(result.total_workloads, 2)
        self.assertEqual(result.placed_count, 1)
        self.assertEqual(result.rejected_count, 1)
        self.assertAlmostEqual(result.rejection_rate_pct, 50.0)

        # After t=10, job 1 departs, leaving cluster empty
        self.assertEqual(cluster.used_cpu, 0)
        self.assertEqual(cluster.used_ram, 0)

    def test_empty_workload_simulation(self) -> None:
        """Simulating an empty workload list returns a zeroed SimulationResult."""
        cluster = Cluster([("srv-1", 8, 16)])
        simulator = TraceSimulator(cluster, WorstFit())
        result = simulator.run([])

        self.assertEqual(result.total_workloads, 0)
        self.assertEqual(result.placed_count, 0)
        self.assertEqual(result.rejected_count, 0)
        self.assertEqual(result.simulation_duration_sec, 0.0)


class TestTraceLoader(unittest.TestCase):
    """Test suite for CSV trace loading, exporting, and schema scaling."""

    def test_csv_save_and_load_round_trip(self) -> None:
        """Exporting workloads and reloading them preserves all properties."""
        workloads = [
            Workload("w-01", 2.0, 4.0, arrival_time=0.0, duration=100.0),
            Workload("w-02", 4.0, 8.0, arrival_time=25.5, duration=200.0),
            Workload("w-03", 1.0, 2.0, arrival_time=50.0, duration=None),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_file = Path(tmp_dir) / "test_trace.csv"
            save_csv_trace(workloads, csv_file)
            self.assertTrue(csv_file.exists())

            loaded = load_csv_trace(csv_file)
            self.assertEqual(len(loaded), 3)

            self.assertEqual(loaded[0].workload_id, "w-01")
            self.assertAlmostEqual(loaded[0].cpu, 2.0)
            self.assertAlmostEqual(loaded[0].ram, 4.0)
            self.assertAlmostEqual(loaded[0].arrival_time, 0.0)
            self.assertAlmostEqual(loaded[0].duration, 100.0)

            self.assertEqual(loaded[1].workload_id, "w-02")
            self.assertAlmostEqual(loaded[1].arrival_time, 25.5)

            self.assertEqual(loaded[2].workload_id, "w-03")
            self.assertIsNone(loaded[2].duration)

    def test_google_borg_scaling(self) -> None:
        """Borg fractional cpu_request and mem_request scale correctly to cores and GB."""
        borg_csv = (
            "job_id,timestamp,runtime,cpu_request,mem_request\n"
            "borg-task-1,100,600,0.25,0.5\n"
            "borg-task-2,150,300,0.50,0.25\n"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "borg_trace.csv"
            csv_path.write_text(borg_csv, encoding="utf-8")

            # Scale fractional requests by 32 CPU cores and 64 GB RAM
            workloads = load_csv_trace(csv_path, cpu_scale=32.0, ram_scale=64.0)

            self.assertEqual(len(workloads), 2)
            self.assertEqual(workloads[0].workload_id, "borg-task-1")
            self.assertAlmostEqual(workloads[0].cpu, 8.0)  # 0.25 * 32
            self.assertAlmostEqual(workloads[0].ram, 32.0)  # 0.5 * 64
            self.assertAlmostEqual(workloads[0].arrival_time, 100.0)
            self.assertAlmostEqual(workloads[0].duration, 600.0)

            self.assertEqual(workloads[1].workload_id, "borg-task-2")
            self.assertAlmostEqual(workloads[1].cpu, 16.0)  # 0.5 * 32
            self.assertAlmostEqual(workloads[1].ram, 16.0)  # 0.25 * 64

    def test_file_not_found_error(self) -> None:
        """Non-existent trace path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_csv_trace("non_existent_trace_file_xyz.csv")

    def test_missing_resource_columns_raises_error(self) -> None:
        """CSV without CPU or RAM columns raises InvalidResourceError."""
        bad_csv = "job_id,arrival_time\nw1,10.0\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "bad_trace.csv"
            csv_path.write_text(bad_csv, encoding="utf-8")

            with self.assertRaises(InvalidResourceError):
                load_csv_trace(csv_path)

    def test_generate_alibaba_style_trace_reproducibility(self) -> None:
        """generate_alibaba_style_trace is deterministic given the same seed."""
        trace1 = generate_alibaba_style_trace(n=50, seed=123)
        trace2 = generate_alibaba_style_trace(n=50, seed=123)
        self.assertEqual(len(trace1), 50)
        self.assertEqual(trace1, trace2)


if __name__ == "__main__":
    unittest.main()

