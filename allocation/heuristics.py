"""Heuristic resource allocation policies for cloud server clusters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from allocation.cluster import Cluster
from allocation.resource import CloudServer
from allocation.workload import Workload


def _normalized_leftover(server: CloudServer, workload: Workload) -> float:
    """Calculate normalized leftover CPU and RAM fraction after placing a workload."""
    return (
        (server.available_cpu() - workload.cpu) / server.total_cpu
        + (server.available_ram() - workload.ram) / server.total_ram
    )


class Allocator(ABC):
    """Abstract base class for cluster placement algorithms.

    Subclasses implement the `_select` method to choose a candidate server
    according to their specific scheduling heuristic.
    """

    @abstractmethod
    def _select(self, workload: Workload, cluster: Cluster) -> Optional[CloudServer]:
        """Select a feasible server for the workload, or return None if rejected."""
        pass

    def place(self, workload: Workload, cluster: Cluster) -> Optional[CloudServer]:
        """Attempt to place and allocate the workload onto a server in the cluster.

        Args:
            workload: The compute workload requesting allocation.
            cluster: The cluster of available servers.

        Returns:
            The CloudServer where the workload was placed, or None if rejected.
        """
        server = self._select(workload, cluster)
        if server:
            server.allocate(workload)
        return server


class FirstFit(Allocator):
    """First-Fit placement algorithm.

    Plain Language Explanation:
    First-Fit examines servers sequentially in order from first to last. It assigns
    the incoming workload to the very first server that has enough free CPU and RAM
    capacity. Once a feasible server is found, the search terminates immediately.
    If no server has sufficient resources, the workload is rejected.
    """

    def _select(self, workload: Workload, cluster: Cluster) -> Optional[CloudServer]:
        for server in cluster.servers:
            if server.can_allocate(workload):
                return server
        return None


class BestFit(Allocator):
    """Best-Fit placement algorithm.

    Plain Language Explanation:
    Best-Fit evaluates all feasible servers that can satisfy the workload's CPU
    and RAM demands, and calculates how much unused capacity would remain on each
    server as a fraction of its total capacity (normalized leftover). It chooses
    the server that minimizes this leftover, packing workloads as tightly as
    possible into already-utilized or appropriately sized servers to keep other
    servers completely empty for future larger demands.
    """

    def _select(self, workload: Workload, cluster: Cluster) -> Optional[CloudServer]:
        feasible = [s for s in cluster.servers if s.can_allocate(workload)]
        if not feasible:
            return None
        return min(feasible, key=lambda s: _normalized_leftover(s, workload))


class WorstFit(Allocator):
    """Worst-Fit placement algorithm.

    Plain Language Explanation:
    Worst-Fit evaluates all feasible servers and places the workload onto the server
    that will have the greatest normalized leftover capacity after placement. By
    always picking the server with the most remaining headroom relative to its size,
    Worst-Fit disperses workloads evenly across the cluster to avoid hot spots and
    balance utilization across servers.
    """

    def _select(self, workload: Workload, cluster: Cluster) -> Optional[CloudServer]:
        feasible = [s for s in cluster.servers if s.can_allocate(workload)]
        if not feasible:
            return None
        return max(feasible, key=lambda s: _normalized_leftover(s, workload))


class RoundRobin(Allocator):
    """Round-Robin placement algorithm.

    Plain Language Explanation:
    Round-Robin cycles through servers in a circular sequence to distribute
    placement attempts evenly across the fleet. When a new workload arrives,
    it starts scanning from the server immediately following the last successful
    placement. It checks servers in circular turn-taking order until it finds
    one with enough capacity, balancing placement opportunities across all nodes.
    """

    def __init__(self) -> None:
        self._current_index: int = 0

    def reset(self) -> None:
        """Reset the circular pointer back to the first server."""
        self._current_index = 0

    def _select(self, workload: Workload, cluster: Cluster) -> Optional[CloudServer]:
        num_servers = len(cluster.servers)
        if num_servers == 0:
            return None

        start_index = self._current_index % num_servers
        for step in range(num_servers):
            idx = (start_index + step) % num_servers
            server = cluster.servers[idx]
            if server.can_allocate(workload):
                self._current_index = (idx + 1) % num_servers
                return server
        return None

