"""Thread-safe control-plane service for live cluster allocation and simulation."""

import threading
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from allocation.cluster import Cluster
from allocation.exceptions import AllocationError, InvalidResourceError, ReleaseError
from allocation.config import build_heterogeneous_specs
from allocation.heuristics import Allocator, BestFit, FirstFit, RoundRobin, WorstFit
from allocation.resource import CloudServer
from allocation.simulator import TraceSimulator
from allocation.workload import Workload
from datasets.trace_loader import generate_alibaba_style_trace
from experiments.benchmark_suite import compute_sla_efficiency_score
from ml.features import bucketize_trace
from ml.optimizer import PredictiveAllocator, UCBBanditAllocator
from ml.predictor import EWMAForecaster, LinearARForecaster


class CloudPilotService:
    """Thread-safe production control plane wrapping Cluster, Allocators, and TraceSimulator."""

    def __init__(
        self,
        server_specs: Optional[List[Tuple[str, float, float]]] = None,
        default_policy: str = "predictive_ar",
    ) -> None:
        self._lock = threading.RLock()
        self._started_at = time.time()
        self._specs = server_specs if server_specs is not None else build_heterogeneous_specs(20)
        self.cluster = Cluster(self._specs)

        # Pre-fit LinearARForecaster on warmup synthetic Alibaba trace
        warmup_trace = generate_alibaba_style_trace(n=300, seed=42, horizon_seconds=1200.0)
        warmup_buckets = bucketize_trace(warmup_trace, window_size=30.0)
        self._ar_model = LinearARForecaster(lags=4, ridge_lambda=2.0)
        self._ar_model.fit(warmup_buckets)

        self._allocators: Dict[str, Allocator] = {
            "first_fit": FirstFit(),
            "best_fit": BestFit(),
            "worst_fit": WorstFit(),
            "round_robin": RoundRobin(),
            "predictive_ewma": PredictiveAllocator(predictor=EWMAForecaster(alpha=0.35, beta=0.1)),
            "predictive_ar": PredictiveAllocator(predictor=self._ar_model),
            "ucb_bandit": UCBBanditAllocator(),
        }

        if default_policy not in self._allocators:
            raise ValueError(f"Unknown policy '{default_policy}'.")
        self.active_policy: str = default_policy

        # Maps workload_id -> (Workload, server_id)
        self._active_workloads: Dict[str, Tuple[Workload, str]] = {}
        self._placed_total: int = 0
        self._rejected_total: int = 0

    @property
    def available_policies(self) -> List[str]:
        return list(self._allocators.keys())

    def get_health(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "status": "healthy",
                "service": "CloudPilot Control Plane",
                "uptime_sec": round(time.time() - self._started_at, 2),
                "active_policy": self.active_policy,
                "available_policies": self.available_policies,
                "total_servers": len(self.cluster.servers),
                "active_workloads": len(self._active_workloads),
            }

    def set_policy(self, policy_name: str) -> Dict[str, Any]:
        with self._lock:
            key = str(policy_name).strip().lower()
            if key not in self._allocators:
                raise InvalidResourceError(
                    f"Unsupported policy '{policy_name}'. Available: {self.available_policies}"
                )
            self.active_policy = key
            return {"active_policy": self.active_policy}

    def get_cluster_status(self) -> Dict[str, Any]:
        with self._lock:
            servers_data = []
            for s in self.cluster.servers:
                servers_data.append({
                    "server_id": s.server_id,
                    "total_cpu": s.total_cpu,
                    "total_ram": s.total_ram,
                    "used_cpu": round(s.used_cpu, 4),
                    "used_ram": round(s.used_ram, 4),
                    "available_cpu": round(s.available_cpu(), 4),
                    "available_ram": round(s.available_ram(), 4),
                    "cpu_utilization": round(s.cpu_utilization, 2),
                    "ram_utilization": round(s.ram_utilization, 2),
                    "workload_count": len(s.allocated_workloads),
                })

            return {
                "active_policy": self.active_policy,
                "total_servers": len(self.cluster.servers),
                "servers_in_use": self.cluster.servers_in_use,
                "cpu_utilization_pct": round(self.cluster.cpu_utilization, 2),
                "ram_utilization_pct": round(self.cluster.ram_utilization, 2),
                "load_imbalance": round(self.cluster.load_imbalance, 2),
                "placed_total": self._placed_total,
                "rejected_total": self._rejected_total,
                "active_workloads_count": len(self._active_workloads),
                "servers": servers_data,
            }

    def allocate_workload(
        self,
        workload_id: str,
        cpu: float,
        ram: float,
        arrival_time: float = 0.0,
        duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if not isinstance(workload_id, str) or not workload_id.strip():
                raise InvalidResourceError("workload_id must be a non-empty string.")
            clean_id = workload_id.strip()
            if clean_id in self._active_workloads:
                raise InvalidResourceError(f"Workload '{clean_id}' is already active in the cluster.")

            workload = Workload(
                workload_id=clean_id,
                cpu_requirement=cpu,
                ram_requirement=ram,
                arrival_time=arrival_time,
                duration=duration,
            )
            allocator = self._allocators[self.active_policy]
            if hasattr(allocator, "observe"):
                allocator.observe(workload)
            server: Optional[CloudServer] = allocator.place(workload, self.cluster)

            if server is None:
                self._rejected_total += 1
                return {
                    "placed": False,
                    "workload_id": clean_id,
                    "policy": self.active_policy,
                    "reason": "Insufficient cluster capacity (SLA rejection).",
                }

            self._active_workloads[clean_id] = (workload, server.server_id)
            self._placed_total += 1
            return {
                "placed": True,
                "workload_id": clean_id,
                "server_id": server.server_id,
                "policy": self.active_policy,
                "server_cpu_utilization_pct": round(server.cpu_utilization, 2),
                "server_ram_utilization_pct": round(server.ram_utilization, 2),
            }

    def release_workload(self, workload_id: str) -> Dict[str, Any]:
        with self._lock:
            clean_id = str(workload_id).strip()
            if clean_id not in self._active_workloads:
                raise ReleaseError(f"Active workload '{clean_id}' not found.")

            workload, server_id = self._active_workloads.pop(clean_id)
            target_server = next(
                (s for s in self.cluster.servers if s.server_id == server_id), None
            )
            if target_server is None:
                raise ReleaseError(f"Server '{server_id}' not found.")

            target_server.release(workload)
            return {
                "released": True,
                "workload_id": clean_id,
                "server_id": server_id,
                "server_cpu_utilization_pct": round(target_server.cpu_utilization, 2),
                "server_ram_utilization_pct": round(target_server.ram_utilization, 2),
            }

    def reset_cluster(self) -> Dict[str, Any]:
        with self._lock:
            self.cluster.reset()
            self._active_workloads.clear()
            self._placed_total = 0
            self._rejected_total = 0
            for alloc in self._allocators.values():
                if hasattr(alloc, "reset_state"):
                    alloc.reset_state()
            return {
                "reset": True,
                "servers_in_use": self.cluster.servers_in_use,
                "active_workloads_count": 0,
            }

    def run_simulation(
        self,
        policy: Optional[str] = None,
        num_workloads: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        with self._lock:
            if not isinstance(policy, (str, type(None))):
                raise InvalidResourceError("policy must be a string or None.")
            chosen_policy = (policy or self.active_policy).strip().lower()
            if chosen_policy not in self._allocators:
                raise InvalidResourceError(
                    f"Unsupported policy '{chosen_policy}'. Available: {self.available_policies}"
                )
            if isinstance(num_workloads, bool) or not isinstance(num_workloads, int) or not (1 <= num_workloads <= 2000):
                raise InvalidResourceError("num_workloads must be an integer between 1 and 2000.")
            if isinstance(seed, bool) or not isinstance(seed, (int, float)):
                raise InvalidResourceError("seed must be an integer.")

            sim_cluster = Cluster(self._specs)
            trace = generate_alibaba_style_trace(n=num_workloads, seed=int(seed), horizon_seconds=1800.0)

            # Use a fresh allocator instance for isolated simulation runs
            if chosen_policy == "first_fit":
                alloc: Allocator = FirstFit()
            elif chosen_policy == "best_fit":
                alloc = BestFit()
            elif chosen_policy == "worst_fit":
                alloc = WorstFit()
            elif chosen_policy == "round_robin":
                alloc = RoundRobin()
            elif chosen_policy == "predictive_ewma":
                alloc = PredictiveAllocator(predictor=EWMAForecaster(alpha=0.35, beta=0.1))
            elif chosen_policy == "predictive_ar":
                alloc = PredictiveAllocator(predictor=self._ar_model)
            else:
                alloc = UCBBanditAllocator()

        # Run simulation OUTSIDE service._lock to prevent blocking live API requests
        simulator = TraceSimulator(cluster=sim_cluster, allocator=alloc)
        result = simulator.run(trace, reset_cluster=True)
        data = asdict(result)
        data["policy_key"] = chosen_policy
        data["sla_efficiency_score"] = compute_sla_efficiency_score(result)
        return data

