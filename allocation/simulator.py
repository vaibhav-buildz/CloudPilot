"""Discrete-event simulator for dynamic cloud workload trace replay."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from allocation.cluster import Cluster
from allocation.heuristics import Allocator
from allocation.resource import CloudServer
from allocation.workload import Workload

EVENT_DEPARTURE = 0
EVENT_ARRIVAL = 1


@dataclass(frozen=True)
class SimulationResult:
    """Encapsulates the metrics and execution telemetry of a trace simulation run."""

    algorithm: str
    total_workloads: int
    placed_count: int
    rejected_count: int
    rejection_rate_pct: float
    avg_cpu_utilization_pct: float
    avg_ram_utilization_pct: float
    peak_cpu_utilization_pct: float
    peak_ram_utilization_pct: float
    avg_load_imbalance: float
    peak_active_servers: int
    allocated_cpu_hours: float
    allocated_ram_gb_hours: float
    simulation_duration_sec: float


class TraceSimulator:
    """Discrete-event simulation engine executing dynamic workload arrivals and departures.

    Maintains a priority queue of chronological events:
    - ARRIVAL: Workload enters the system and is offered to the Allocator.
    - DEPARTURE: Successfully placed workload completes its runtime and is released from its server.

    Departures take precedence over arrivals at identical timestamps to ensure released
    resources are immediately available for concurrent arrivals.
    """

    def __init__(self, cluster: Cluster, allocator: Allocator) -> None:
        self.cluster = cluster
        self.allocator = allocator

    def run(self, workloads: Iterable[Workload], reset_cluster: bool = True) -> SimulationResult:
        """Run the discrete-event simulation across the given workload stream.

        Args:
            workloads: Iterable sequence of Workload objects to simulate.
            reset_cluster: Whether to reset the cluster to an idle state before starting.

        Returns:
            SimulationResult containing comprehensive utilization, SLA, and capacity metrics.
        """
        if reset_cluster:
            self.cluster.reset()

        workload_list: List[Workload] = list(workloads)
        if not workload_list:
            return SimulationResult(
                algorithm=self.allocator.__class__.__name__,
                total_workloads=0,
                placed_count=0,
                rejected_count=0,
                rejection_rate_pct=0.0,
                avg_cpu_utilization_pct=0.0,
                avg_ram_utilization_pct=0.0,
                peak_cpu_utilization_pct=0.0,
                peak_ram_utilization_pct=0.0,
                avg_load_imbalance=0.0,
                peak_active_servers=0,
                allocated_cpu_hours=0.0,
                allocated_ram_gb_hours=0.0,
                simulation_duration_sec=0.0,
            )

        event_queue: List[Tuple[float, int, int, Workload]] = []
        for idx, w in enumerate(workload_list):
            heapq.heappush(event_queue, (w.arrival_time, EVENT_ARRIVAL, idx, w))

        active_placements: Dict[str, CloudServer] = {}
        placed_count = 0
        rejected_count = 0
        allocated_cpu_sec = 0.0
        allocated_ram_sec = 0.0

        start_time = min(w.arrival_time for w in workload_list)
        last_time = start_time

        weighted_cpu_util = 0.0
        weighted_ram_util = 0.0
        weighted_imbalance = 0.0

        peak_cpu_util = 0.0
        peak_ram_util = 0.0
        peak_servers = 0
        tie_counter = len(workload_list)

        while event_queue:
            current_time, event_type, _, workload = heapq.heappop(event_queue)
            dt = current_time - last_time

            if dt > 0.0:
                weighted_cpu_util += self.cluster.cpu_utilization * dt
                weighted_ram_util += self.cluster.ram_utilization * dt
                weighted_imbalance += self.cluster.load_imbalance * dt
                last_time = current_time

            if event_type == EVENT_DEPARTURE:
                server = active_placements.pop(workload.workload_id, None)
                if server is not None:
                    server.release(workload)
            else:
                server = self.allocator.place(workload, self.cluster)
                if server is not None:
                    placed_count += 1
                    eff_duration = workload.duration if workload.duration is not None else 0.0
                    allocated_cpu_sec += workload.cpu * eff_duration
                    allocated_ram_sec += workload.ram * eff_duration

                    if workload.duration is not None:
                        active_placements[workload.workload_id] = server
                        tie_counter += 1
                        heapq.heappush(
                            event_queue,
                            (workload.arrival_time + workload.duration, EVENT_DEPARTURE, tie_counter, workload),
                        )
                else:
                    rejected_count += 1

            peak_cpu_util = max(peak_cpu_util, self.cluster.cpu_utilization)
            peak_ram_util = max(peak_ram_util, self.cluster.ram_utilization)
            peak_servers = max(peak_servers, self.cluster.servers_in_use)

        total_duration = max(last_time - start_time, 0.0)
        if total_duration > 0.0:
            avg_cpu = weighted_cpu_util / total_duration
            avg_ram = weighted_ram_util / total_duration
            avg_imb = weighted_imbalance / total_duration
        else:
            avg_cpu = self.cluster.cpu_utilization
            avg_ram = self.cluster.ram_utilization
            avg_imb = self.cluster.load_imbalance

        total_jobs = len(workload_list)
        return SimulationResult(
            algorithm=self.allocator.__class__.__name__,
            total_workloads=total_jobs,
            placed_count=placed_count,
            rejected_count=rejected_count,
            rejection_rate_pct=round((rejected_count / total_jobs) * 100.0, 2),
            avg_cpu_utilization_pct=round(avg_cpu, 2),
            avg_ram_utilization_pct=round(avg_ram, 2),
            peak_cpu_utilization_pct=round(peak_cpu_util, 2),
            peak_ram_utilization_pct=round(peak_ram_util, 2),
            avg_load_imbalance=round(avg_imb, 2),
            peak_active_servers=peak_servers,
            allocated_cpu_hours=round(allocated_cpu_sec / 3600.0, 2),
            allocated_ram_gb_hours=round(allocated_ram_sec / 3600.0, 2),
            simulation_duration_sec=round(total_duration, 2),
        )

