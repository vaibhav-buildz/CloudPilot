"""Synthetic workload generator for cloud resource allocation benchmarks."""

from __future__ import annotations

import random
from typing import List, Optional

from allocation.workload import Workload


def _clip(val: float, low: float, high: float) -> float:
    """Clamp a value within the closed interval [low, high]."""
    return max(low, min(val, high))


def generate_workloads(n: int, seed: Optional[int] = None) -> List[Workload]:
    """Generate a sequence of synthetic compute workloads with realistic demands.

    Workload characteristics:
    - Skewed CPU demands following a log-normal distribution, clipped to [0.25, 16.0] vCPUs.
    - RAM demands correlated with CPU demands via a uniform multiplier [1.5, 5.0],
      clipped to [0.5, 64.0] GB.
    - Deterministic generation using a local `random.Random(seed)` instance to avoid
      polluting global random state.

    Args:
        n: Number of workloads to generate.
        seed: Optional integer seed for reproducible generation.

    Returns:
        List of generated Workload instances.
    """
    rng = random.Random(seed)
    workloads: List[Workload] = []

    for i in range(n):
        cpu = _clip(rng.lognormvariate(0.8, 0.9), 0.25, 16.0)
        ram = _clip(cpu * rng.uniform(1.5, 5.0), 0.5, 64.0)
        workloads.append(
            Workload(
                workload_id=f"wl-{i+1}",
                cpu_requirement=cpu,
                ram_requirement=ram,
            )
        )

    return workloads

