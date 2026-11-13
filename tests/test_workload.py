"""Unit tests for the Workload model."""

import unittest

from allocation.exceptions import InvalidResourceError
from allocation.workload import Workload


class TestWorkload(unittest.TestCase):
    """Test suite for Workload model validation and behavior."""

    def test_valid_initialization(self) -> None:
        """Workload initializes correctly with valid attributes."""
        wl = Workload(workload_id="wl-1", cpu_requirement=4, ram_requirement=8)
        self.assertEqual(wl.workload_id, "wl-1")
        self.assertEqual(wl.cpu_requirement, 4)
        self.assertEqual(wl.ram_requirement, 8)
        self.assertEqual(wl.cpu, 4)
        self.assertEqual(wl.ram, 8)

    def test_valid_float_requirements(self) -> None:
        """Workload supports floating point CPU and RAM requirements."""
        wl = Workload(workload_id="wl-float", cpu_requirement=0.5, ram_requirement=1.5)
        self.assertEqual(wl.cpu_requirement, 0.5)
        self.assertEqual(wl.ram_requirement, 1.5)

    def test_workload_id_stripping(self) -> None:
        """Workload ID with surrounding whitespace is stripped."""
        wl = Workload(workload_id="  wl-padded  ", cpu_requirement=2, ram_requirement=4)
        self.assertEqual(wl.workload_id, "wl-padded")

    def test_invalid_workload_id(self) -> None:
        """Non-string or empty workload IDs are rejected."""
        invalid_ids = ["", "   ", None, 123, [], {}]
        for bad_id in invalid_ids:
            with self.subTest(bad_id=bad_id):
                with self.assertRaises(InvalidResourceError):
                    Workload(workload_id=bad_id, cpu_requirement=2, ram_requirement=4)  # type: ignore

    def test_non_positive_cpu_requirement(self) -> None:
        """CPU requirements <= 0 are rejected."""
        for bad_cpu in [0, -1, -0.01]:
            with self.subTest(bad_cpu=bad_cpu):
                with self.assertRaises(InvalidResourceError):
                    Workload(workload_id="wl-test", cpu_requirement=bad_cpu, ram_requirement=4)

    def test_non_positive_ram_requirement(self) -> None:
        """RAM requirements <= 0 are rejected."""
        for bad_ram in [0, -2, -0.5]:
            with self.subTest(bad_ram=bad_ram):
                with self.assertRaises(InvalidResourceError):
                    Workload(workload_id="wl-test", cpu_requirement=2, ram_requirement=bad_ram)

    def test_invalid_numeric_types(self) -> None:
        """Boolean and non-numeric requirements are rejected."""
        invalid_values = [True, False, "4", None, [2], {"cpu": 2}, float("nan"), float("inf")]
        for bad_val in invalid_values:
            with self.subTest(bad_val=bad_val):
                with self.assertRaises(InvalidResourceError):
                    Workload(workload_id="wl-test", cpu_requirement=bad_val, ram_requirement=4)  # type: ignore
                with self.assertRaises(InvalidResourceError):
                    Workload(workload_id="wl-test", cpu_requirement=4, ram_requirement=bad_val)  # type: ignore

    def test_equality_and_hashing(self) -> None:
        """Workloads with identical parameters are equal and hashable."""
        wl1 = Workload("wl-1", 2, 4)
        wl2 = Workload("wl-1", 2, 4)
        wl3 = Workload("wl-2", 2, 4)
        self.assertEqual(wl1, wl2)
        self.assertNotEqual(wl1, wl3)
        self.assertEqual(hash(wl1), hash(wl2))
        self.assertIn(wl1, {wl2})

    def test_string_representation(self) -> None:
        """Workload __repr__ and __str__ produce informative output."""
        wl = Workload("wl-db", 4, 16)
        self.assertIn("wl-db", repr(wl))
        self.assertIn("CPU: 4", str(wl))
        self.assertIn("16GB", str(wl))


if __name__ == "__main__":
    unittest.main()

