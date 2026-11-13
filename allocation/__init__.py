"""CloudPilot Resource Allocation Engine.

Provides core abstractions for modeling cloud compute resources, workloads, clusters,
heuristic placement algorithms, discrete-event trace simulation, and intelligent ML optimizers.
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
from ml.optimizer import PredictiveAllocator, UCBBanditAllocator

__all__ = [
    "CloudServer",
    "Workload",
    "Cluster",
    "Allocator",
    "FirstFit",
    "BestFit",
    "WorstFit",
    "RoundRobin",
    "PredictiveAllocator",
    "UCBBanditAllocator",
    "TraceSimulator",
    "SimulationResult",
    "AllocationError",
    "InvalidResourceError",
    "InsufficientCapacityError",
    "InsufficientCPUError",
    "InsufficientRAMError",
    "ReleaseError",
]
