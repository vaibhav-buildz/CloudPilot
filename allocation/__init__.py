"""CloudPilot Resource Allocation Engine.

Provides core abstractions for modeling cloud compute resources, workloads, clusters,
heuristic placement algorithms, and discrete-event trace simulation.
"""

from allocation.cluster import Cluster
from allocation.exceptions import (
    AllocationError,
    InsufficientCapacityError,
    InsufficientCPUError,
    InsufficientRAMError,
    InvalidResourceError,
    ReleaseError,
)
from allocation.heuristics import (
    Allocator,
    BestFit,
    FirstFit,
    RoundRobin,
    WorstFit,
)
from allocation.resource import CloudServer
from allocation.simulator import SimulationResult, TraceSimulator
from allocation.workload import Workload

__all__ = [
    "CloudServer",
    "Workload",
    "Cluster",
    "Allocator",
    "FirstFit",
    "BestFit",
    "WorstFit",
    "RoundRobin",
    "TraceSimulator",
    "SimulationResult",
    "AllocationError",
    "InvalidResourceError",
    "InsufficientCapacityError",
    "InsufficientCPUError",
    "InsufficientRAMError",
    "ReleaseError",
]
