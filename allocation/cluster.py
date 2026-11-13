"""Cluster model representing an aggregate fleet of cloud compute servers."""

from __future__ import annotations

import statistics
from typing import Iterator, List, Sequence, Tuple, Union

from allocation.resource import CloudServer


class Cluster:
    """Represents an aggregate pool of CloudServer instances.

    A Cluster is initialized from a sequence of server specifications:
    `(server_id, total_cpu, total_ram)`. It provides cluster-wide utilization
    metrics, load balancing analytics, and reset capabilities.
    """

    def __init__(
        self,
        specs: Sequence[Tuple[str, Union[int, float], Union[int, float]]],
    ) -> None:
        """Initialize the cluster with server specifications.

        Args:
            specs: A sequence of tuples in the form `(server_id, cpu_capacity, ram_capacity)`.
        """
        self.specs: List[Tuple[str, Union[int, float], Union[int, float]]] = list(specs)
        self.servers: List[CloudServer] = []
        self.reset()

    def reset(self) -> None:
        """Rebuild clean, idle servers from the original cluster specifications."""
        self.servers = [
            CloudServer(server_id=s_id, total_cpu=cpu, total_ram=ram)
            for s_id, cpu, ram in self.specs
        ]

    @property
    def total_cpu(self) -> Union[int, float]:
        """Aggregate total CPU capacity across all servers in the cluster."""
        return sum(s.total_cpu for s in self.servers)

    @property
    def total_ram(self) -> Union[int, float]:
        """Aggregate total RAM capacity across all servers in the cluster."""
        return sum(s.total_ram for s in self.servers)

    @property
    def used_cpu(self) -> Union[int, float]:
        """Aggregate allocated CPU across all servers in the cluster."""
        return sum(s.used_cpu for s in self.servers)

    @property
    def used_ram(self) -> Union[int, float]:
        """Aggregate allocated RAM across all servers in the cluster."""
        return sum(s.used_ram for s in self.servers)

    @property
    def cpu_utilization(self) -> float:
        """Cluster-wide CPU utilization percentage (0.0% to 100.0%)."""
        total = self.total_cpu
        if total <= 0:
            return 0.0
        return (self.used_cpu / total) * 100.0

    @property
    def ram_utilization(self) -> float:
        """Cluster-wide RAM utilization percentage (0.0% to 100.0%)."""
        total = self.total_ram
        if total <= 0:
            return 0.0
        return (self.used_ram / total) * 100.0

    @property
    def servers_in_use(self) -> int:
        """Count of servers with at least one active resource allocation."""
        return sum(1 for s in self.servers if s.used_cpu > 1e-9 or s.used_ram > 1e-9)

    @property
    def load_imbalance(self) -> float:
        """Population standard deviation of average utilization across servers in use.

        For each active server, average utilization is defined as the mean of its
        CPU utilization (%) and RAM utilization (%). If fewer than 2 servers are in
        use, the population standard deviation is 0.0.
        """
        active_servers = [
            s for s in self.servers if s.used_cpu > 1e-9 or s.used_ram > 1e-9
        ]
        if len(active_servers) < 2:
            return 0.0

        avg_utils = [
            (s.cpu_utilization + s.ram_utilization) / 2.0
            for s in active_servers
        ]
        return statistics.pstdev(avg_utils)

    def __iter__(self) -> Iterator[CloudServer]:
        return iter(self.servers)

    def __len__(self) -> int:
        return len(self.servers)

    def __getitem__(self, index: int) -> CloudServer:
        return self.servers[index]

    def __repr__(self) -> str:
        return (
            f"Cluster(servers={len(self.servers)}, "
            f"cpu_util={self.cpu_utilization:.1f}%, "
            f"ram_util={self.ram_utilization:.1f}%, "
            f"active={self.servers_in_use})"
        )

