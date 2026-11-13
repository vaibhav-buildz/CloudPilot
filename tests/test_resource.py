"""Unit tests for CloudServer resource management and validation."""

import math
import unittest

from allocation.exceptions import (
    AllocationError,
    InsufficientCapacityError,
    InsufficientCPUError,
    InsufficientRAMError,
    InvalidResourceError,
    ReleaseError,
)
from allocation.resource import CloudServer
from allocation.workload import Workload


class TestCloudServer(unittest.TestCase):
    """Test suite for CloudServer initialization, allocations, and constraints."""

    # -------------------------------------------------------------------------
    # 1. Server initialization
    # -------------------------------------------------------------------------

    def test_server_initialization_success(self) -> None:
        """CloudServer initializes with expected idle state and attributes."""
        server = CloudServer(server_id="server-alpha", total_cpu=16, total_ram=64)

        self.assertEqual(server.server_id, "server-alpha")
        self.assertEqual(server.total_cpu, 16)
        self.assertEqual(server.total_ram, 64)
        self.assertEqual(server.used_cpu, 0)
        self.assertEqual(server.used_ram, 0)
        self.assertEqual(server.available_cpu(), 16)
        self.assertEqual(server.available_ram(), 64)
        self.assertAlmostEqual(server.cpu_utilization, 0.0)
        self.assertAlmostEqual(server.ram_utilization, 0.0)
        self.assertEqual(server.allocated_workloads, [])

    def test_server_initialization_float_capacity(self) -> None:
        """CloudServer supports floating point total capacity values."""
        server = CloudServer(server_id="server-small", total_cpu=2.5, total_ram=4.5)
        self.assertEqual(server.total_cpu, 2.5)
        self.assertEqual(server.total_ram, 4.5)
        self.assertEqual(server.available_cpu(), 2.5)
        self.assertEqual(server.available_ram(), 4.5)

    def test_server_initialization_invalid_id(self) -> None:
        """Server initialization rejects empty, non-string, or whitespace-only IDs."""
        for bad_id in ["", "   ", None, 123, []]:
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(InvalidResourceError):
                    CloudServer(server_id=bad_id, total_cpu=8, total_ram=16)  # type: ignore

    def test_server_initialization_invalid_capacity(self) -> None:
        """Server initialization rejects non-positive or non-numeric total capacity."""
        invalid_capacities = [0, -1, -0.5, "8", None, True, False, float("nan"), float("inf")]
        for bad_cap in invalid_capacities:
            with self.subTest(bad_cap=bad_cap):
                with self.assertRaises(InvalidResourceError):
                    CloudServer(server_id="s1", total_cpu=bad_cap, total_ram=16)  # type: ignore
                with self.assertRaises(InvalidResourceError):
                    CloudServer(server_id="s1", total_cpu=8, total_ram=bad_cap)  # type: ignore

    # -------------------------------------------------------------------------
    # 2. Successful allocation
    # -------------------------------------------------------------------------

    def test_successful_numeric_allocation(self) -> None:
        """Allocating valid numeric CPU and RAM updates server state accurately."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)
        result = server.allocate(cpu_or_workload=2, ram=4)

        self.assertTrue(result)
        self.assertEqual(server.used_cpu, 2)
        self.assertEqual(server.used_ram, 4)
        self.assertEqual(server.available_cpu(), 6)
        self.assertEqual(server.available_ram(), 12)
        self.assertAlmostEqual(server.cpu_utilization, 25.0)
        self.assertAlmostEqual(server.ram_utilization, 25.0)

    def test_successful_workload_allocation(self) -> None:
        """Allocating a Workload instance correctly tracks resources and the workload."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)
        wl = Workload(workload_id="job-1", cpu_requirement=4, ram_requirement=8)

        result = server.allocate(wl)
        self.assertTrue(result)
        self.assertEqual(server.used_cpu, 4)
        self.assertEqual(server.used_ram, 8)
        self.assertEqual(server.available_cpu(), 4)
        self.assertEqual(server.available_ram(), 8)
        self.assertIn(wl, server.allocated_workloads)

    def test_allocate_workload_convenience_method(self) -> None:
        """allocate_workload convenience helper works identically."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)
        wl = Workload("wl-x", 3, 6)
        server.allocate_workload(wl)
        self.assertEqual(server.used_cpu, 3)
        self.assertEqual(server.used_ram, 6)

    def test_multiple_sequential_allocations(self) -> None:
        """Multiple sequential allocations accumulate correctly up to 100% capacity."""
        server = CloudServer(server_id="s1", total_cpu=10, total_ram=20)

        server.allocate(3, 5)
        server.allocate(4, 10)
        server.allocate(3, 5)

        self.assertEqual(server.used_cpu, 10)
        self.assertEqual(server.used_ram, 20)
        self.assertEqual(server.available_cpu(), 0)
        self.assertEqual(server.available_ram(), 0)
        self.assertAlmostEqual(server.cpu_utilization, 100.0)
        self.assertAlmostEqual(server.ram_utilization, 100.0)

    # -------------------------------------------------------------------------
    # 3. Successful resource release
    # -------------------------------------------------------------------------

    def test_successful_numeric_release(self) -> None:
        """Releasing allocated numeric resources restores available capacity."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)
        server.allocate(4, 8)

        result = server.release(2, 4)
        self.assertTrue(result)
        self.assertEqual(server.used_cpu, 2)
        self.assertEqual(server.used_ram, 4)
        self.assertEqual(server.available_cpu(), 6)
        self.assertEqual(server.available_ram(), 12)

    def test_successful_workload_release(self) -> None:
        """Releasing a Workload instance restores capacity and untracks the workload."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)
        wl = Workload("job-1", 4, 8)
        server.allocate(wl)

        result = server.release(wl)
        self.assertTrue(result)
        self.assertEqual(server.used_cpu, 0)
        self.assertEqual(server.used_ram, 0)
        self.assertEqual(server.available_cpu(), 8)
        self.assertEqual(server.available_ram(), 16)
        self.assertNotIn(wl, server.allocated_workloads)

    def test_release_workload_convenience_method(self) -> None:
        """release_workload helper releases the specified workload."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)
        wl = Workload("job-a", 2, 4)
        server.allocate_workload(wl)
        server.release_workload(wl)
        self.assertEqual(server.used_cpu, 0)
        self.assertEqual(server.used_ram, 0)

    def test_full_release_to_idle(self) -> None:
        """Releasing all allocated resources restores clean idle state."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)
        server.allocate(8, 16)
        server.release(8, 16)

        self.assertEqual(server.used_cpu, 0)
        self.assertEqual(server.used_ram, 0)
        self.assertEqual(server.available_cpu(), 8)
        self.assertEqual(server.available_ram(), 16)
        self.assertAlmostEqual(server.cpu_utilization, 0.0)
        self.assertAlmostEqual(server.ram_utilization, 0.0)

    # -------------------------------------------------------------------------
    # 4. Insufficient CPU
    # -------------------------------------------------------------------------

    def test_insufficient_cpu_raises_error(self) -> None:
        """Allocating more CPU than available raises InsufficientCPUError and preserves state."""
        server = CloudServer(server_id="s1", total_cpu=4, total_ram=16)
        server.allocate(2, 4)

        # Available CPU is 2, requesting 3 (RAM has 12 available, requesting 4)
        with self.assertRaises(InsufficientCPUError) as ctx:
            server.allocate(cpu_or_workload=3, ram=4)

        self.assertIn("Insufficient CPU", str(ctx.exception))
        # Ensure it also inherits from InsufficientCapacityError, AllocationError, and ValueError
        self.assertIsInstance(ctx.exception, InsufficientCapacityError)
        self.assertIsInstance(ctx.exception, AllocationError)
        self.assertIsInstance(ctx.exception, ValueError)

        # State must remain unmodified
        self.assertEqual(server.used_cpu, 2)
        self.assertEqual(server.used_ram, 4)

    # -------------------------------------------------------------------------
    # 5. Insufficient RAM
    # -------------------------------------------------------------------------

    def test_insufficient_ram_raises_error(self) -> None:
        """Allocating more RAM than available raises InsufficientRAMError and preserves state."""
        server = CloudServer(server_id="s1", total_cpu=16, total_ram=8)
        server.allocate(4, 4)

        # Available RAM is 4, requesting 8 (CPU has 12 available, requesting 2)
        with self.assertRaises(InsufficientRAMError) as ctx:
            server.allocate(cpu_or_workload=2, ram=8)

        self.assertIn("Insufficient RAM", str(ctx.exception))
        self.assertIsInstance(ctx.exception, InsufficientCapacityError)
        self.assertIsInstance(ctx.exception, AllocationError)
        self.assertIsInstance(ctx.exception, ValueError)

        # State must remain unmodified
        self.assertEqual(server.used_cpu, 4)
        self.assertEqual(server.used_ram, 4)

    # -------------------------------------------------------------------------
    # 6. Invalid resource values
    # -------------------------------------------------------------------------

    def test_allocate_rejects_negative_or_zero_values(self) -> None:
        """allocate rejects negative or zero CPU/RAM values."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)

        for bad_cpu in [0, -1, -0.1]:
            with self.subTest(bad_cpu=bad_cpu):
                with self.assertRaises(InvalidResourceError):
                    server.allocate(cpu_or_workload=bad_cpu, ram=4)

        for bad_ram in [0, -2, -0.5]:
            with self.subTest(bad_ram=bad_ram):
                with self.assertRaises(InvalidResourceError):
                    server.allocate(cpu_or_workload=2, ram=bad_ram)

    def test_allocate_rejects_non_numeric_and_nan_inf(self) -> None:
        """allocate rejects booleans, strings, None, NaN, and Inf."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)
        invalid_inputs = [True, False, "4", None, [], {}, float("nan"), float("inf")]

        for bad_val in invalid_inputs:
            with self.subTest(bad_val=bad_val):
                with self.assertRaises(InvalidResourceError):
                    server.allocate(cpu_or_workload=bad_val, ram=4)  # type: ignore
                with self.assertRaises(InvalidResourceError):
                    server.allocate(cpu_or_workload=2, ram=bad_val)  # type: ignore

    def test_allocate_numeric_missing_ram(self) -> None:
        """Allocating numeric CPU without specifying RAM raises InvalidResourceError."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)
        with self.assertRaises(InvalidResourceError):
            server.allocate(cpu_or_workload=2)

    def test_release_rejects_negative_or_zero_values(self) -> None:
        """release rejects negative or zero CPU/RAM values."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)
        server.allocate(4, 8)

        for bad_val in [0, -1, -0.5]:
            with self.subTest(bad_val=bad_val):
                with self.assertRaises(InvalidResourceError):
                    server.release(bad_val, 2)
                with self.assertRaises(InvalidResourceError):
                    server.release(2, bad_val)

    def test_release_rejects_non_numeric_values(self) -> None:
        """release rejects booleans and non-numeric inputs."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)
        server.allocate(4, 8)
        invalid_inputs = [True, False, "2", None, [], float("nan"), float("inf")]

        for bad_val in invalid_inputs:
            with self.subTest(bad_val=bad_val):
                with self.assertRaises(InvalidResourceError):
                    server.release(bad_val, 2)  # type: ignore
                with self.assertRaises(InvalidResourceError):
                    server.release(2, bad_val)  # type: ignore

    # -------------------------------------------------------------------------
    # 7. Allocation beyond capacity
    # -------------------------------------------------------------------------

    def test_allocation_exceeding_total_capacity_directly(self) -> None:
        """Attempting to allocate more than total server capacity on idle server fails."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)

        # Exceeds total CPU
        with self.assertRaises(InsufficientCPUError):
            server.allocate(cpu_or_workload=10, ram=8)

        # Exceeds total RAM
        with self.assertRaises(InsufficientRAMError):
            server.allocate(cpu_or_workload=4, ram=32)

        # Both exceed total
        with self.assertRaises(InsufficientCapacityError):
            server.allocate(cpu_or_workload=20, ram=40)

        # Server remains completely idle
        self.assertEqual(server.used_cpu, 0)
        self.assertEqual(server.used_ram, 0)

    # -------------------------------------------------------------------------
    # 8. Releasing unallocated resources
    # -------------------------------------------------------------------------

    def test_release_from_idle_server(self) -> None:
        """Releasing resources from an idle server raises ReleaseError."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)

        with self.assertRaises(ReleaseError) as ctx:
            server.release(2, 4)

        self.assertIn("only 0 currently allocated", str(ctx.exception))
        self.assertIsInstance(ctx.exception, ValueError)

    def test_release_more_cpu_than_allocated(self) -> None:
        """Releasing more CPU than currently used raises ReleaseError."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)
        server.allocate(2, 8)

        with self.assertRaises(ReleaseError) as ctx:
            server.release(3, 4)

        self.assertIn("only 2 currently allocated", str(ctx.exception))
        self.assertEqual(server.used_cpu, 2)
        self.assertEqual(server.used_ram, 8)

    def test_release_more_ram_than_allocated(self) -> None:
        """Releasing more RAM than currently used raises ReleaseError."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)
        server.allocate(4, 4)

        with self.assertRaises(ReleaseError) as ctx:
            server.release(2, 8)

        self.assertIn("only 4 currently allocated", str(ctx.exception))
        self.assertEqual(server.used_cpu, 4)
        self.assertEqual(server.used_ram, 4)

    # -------------------------------------------------------------------------
    # 9. State consistency after failed operations
    # -------------------------------------------------------------------------

    def test_state_consistency_on_failed_allocation(self) -> None:
        """State is strictly preserved when allocation fails due to RAM constraint."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)
        server.allocate(2, 4)

        initial_used_cpu = server.used_cpu
        initial_used_ram = server.used_ram

        # CPU request is feasible (4 <= 6), but RAM request exceeds available (15 > 12)
        with self.assertRaises(InsufficientRAMError):
            server.allocate(cpu_or_workload=4, ram=15)

        # Neither CPU nor RAM should have changed!
        self.assertEqual(server.used_cpu, initial_used_cpu)
        self.assertEqual(server.used_ram, initial_used_ram)
        self.assertEqual(server.available_cpu(), 6)
        self.assertEqual(server.available_ram(), 12)

    def test_state_consistency_on_invalid_allocation_args(self) -> None:
        """State is strictly preserved when allocation fails due to invalid arguments."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)
        server.allocate(2, 4)

        # Valid CPU, but negative RAM
        with self.assertRaises(InvalidResourceError):
            server.allocate(cpu_or_workload=2, ram=-5)

        self.assertEqual(server.used_cpu, 2)
        self.assertEqual(server.used_ram, 4)

    def test_state_consistency_on_failed_release(self) -> None:
        """State is strictly preserved when release fails (atomic release guarantee)."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)
        server.allocate(4, 6)

        # CPU to release is valid (2 <= 4), but RAM to release exceeds used (10 > 6)
        with self.assertRaises(ReleaseError):
            server.release(cpu_or_workload=2, ram=10)

        # Even though CPU portion was valid, atomic guarantee means CPU was NOT released
        self.assertEqual(server.used_cpu, 4)
        self.assertEqual(server.used_ram, 6)

    def test_can_allocate_helper(self) -> None:
        """can_allocate queries feasibility without altering state or throwing."""
        server = CloudServer(server_id="s1", total_cpu=8, total_ram=16)
        server.allocate(4, 8)

        self.assertTrue(server.can_allocate(2, 4))
        self.assertTrue(server.can_allocate(4, 8))
        self.assertFalse(server.can_allocate(5, 4))  # insufficient CPU
        self.assertFalse(server.can_allocate(2, 10))  # insufficient RAM
        self.assertFalse(server.can_allocate(-1, 2))  # invalid input
        self.assertFalse(server.can_allocate(Workload("big", 16, 32)))

        # State remains intact
        self.assertEqual(server.used_cpu, 4)
        self.assertEqual(server.used_ram, 8)

    def test_representation(self) -> None:
        """__str__ and __repr__ render expected state formatting."""
        server = CloudServer(server_id="node-1", total_cpu=8, total_ram=16)
        server.allocate(4, 8)
        self.assertIn("node-1", repr(server))
        self.assertIn("node-1", str(server))
        self.assertIn("50.0%", str(server))


if __name__ == "__main__":
    unittest.main()
