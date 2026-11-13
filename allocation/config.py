"""Cluster and simulation configuration constants and factory helpers."""

from typing import List, Tuple


def build_heterogeneous_specs(num_servers: int = 20) -> List[Tuple[str, float, float]]:
    """Builds a 60% small (8 CPU, 16 GB) / 40% large (32 CPU, 64 GB) heterogeneous cluster.

    Args:
        num_servers: Total number of servers in the fleet (default: 20).

    Returns:
        List of (server_id, cpu_capacity, ram_capacity) tuples.
    """
    num_small = int(round(num_servers * 0.6))
    num_large = max(0, num_servers - num_small)
    specs: List[Tuple[str, float, float]] = []
    for i in range(num_small):
        specs.append((f"small-{i:03d}", 8.0, 16.0))
    for i in range(num_large):
        specs.append((f"large-{i:03d}", 32.0, 64.0))
    return specs
