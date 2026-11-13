"""Unit tests for synthetic workload generator."""

import random
import unittest

from datasets.synthetic import generate_workloads


class TestSyntheticWorkloads(unittest.TestCase):
    """Test suite for generate_workloads."""

    def test_same_seed_gives_identical_workloads(self) -> None:
        """Generating workloads with the same seed yields identical Workload sequences."""
        run_a = generate_workloads(50, seed=42)
        run_b = generate_workloads(50, seed=42)

        self.assertEqual(len(run_a), 50)
        self.assertEqual(len(run_b), 50)
        self.assertEqual(run_a, run_b)

    def test_different_seeds_give_different_workloads(self) -> None:
        """Generating workloads with different seeds yields different Workload sequences."""
        run_a = generate_workloads(50, seed=42)
        run_b = generate_workloads(50, seed=99)

        self.assertNotEqual(run_a, run_b)

    def test_workload_resource_bounds(self) -> None:
        """Generated CPU and RAM demands respect clipping bounds."""
        workloads = generate_workloads(200, seed=12345)

        for wl in workloads:
            # CPU clipped to [0.25, 16.0]
            self.assertGreaterEqual(wl.cpu, 0.25)
            self.assertLessEqual(wl.cpu, 16.0)
            # RAM clipped to [0.5, 64.0]
            self.assertGreaterEqual(wl.ram, 0.5)
            self.assertLessEqual(wl.ram, 64.0)

    def test_no_global_random_state_side_effects(self) -> None:
        """generate_workloads uses a local generator without altering global random state."""
        random.seed(777)
        global_val_before = random.random()

        random.seed(777)
        # Generate synthetic workloads with a different local seed
        generate_workloads(50, seed=999)
        global_val_after = random.random()

        self.assertEqual(global_val_before, global_val_after)


if __name__ == "__main__":
    unittest.main()

