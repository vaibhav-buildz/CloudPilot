"""Intelligent ML-guided and Reinforcement Learning (UCB Bandit) resource allocators."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from allocation.cluster import Cluster
from allocation.heuristics import Allocator, BestFit, FirstFit, WorstFit
from allocation.resource import CloudServer
from allocation.workload import Workload
from ml.features import TimeWindowBucket
from ml.predictor import DemandForecast, DemandPredictor, EWMAForecaster


class PredictiveAllocator(Allocator):
    """ML-guided multi-objective allocator combining demand forecasting, tier-aware
    headroom reservation, cross-dimensional CPU/RAM skew alignment, anti-sliver guards,
    and adaptive power consolidation.

    Scheduling Mechanics:
    1. Streaming Forecast Update: Aggregates arriving workloads into rolling time windows
       and queries `predictor.predict_next()` to estimate upcoming burst intensity.
    2. Tier-Aware Whale Protection: Protects high-capacity servers from fragmentation by
       routing small workloads to small servers first, scaling the penalty up when the ML
       forecaster predicts an incoming resource burst.
    3. Cross-Dimensional Skew Alignment: Minimizes `|rem_cpu_ratio - rem_ram_ratio|` after
       placement so RAM-intensive jobs pair with CPU-intensive jobs instead of stranding cores.
    4. Anti-Sliver Guard: Penalizes leaving unusable RAM slivers (0.15 to 1.5 GB) that cannot
       fit future requests unless the placement nearly fills the server.
    5. Sequential Turnover Preference: Preserves fast resource turnover on earlier servers.
    """

    def __init__(
        self,
        predictor: Optional[DemandPredictor] = None,
        window_size: float = 30.0,
        small_cpu_threshold: float = 4.0,
        small_ram_threshold: float = 12.0,
        skew_weight: float = 0.10,
        packing_weight: float = 0.20,
        idle_wakeup_penalty: float = 0.35,
        sliver_penalty: float = 0.45,
        tier_penalty_weight: float = 1.25,
        order_bias_weight: float = 0.40,
    ) -> None:
        self.predictor: DemandPredictor = predictor if predictor is not None else EWMAForecaster()
        self.window_size = float(window_size)
        self.small_cpu_threshold = float(small_cpu_threshold)
        self.small_ram_threshold = float(small_ram_threshold)
        self.skew_weight = float(skew_weight)
        self.packing_weight = float(packing_weight)
        self.idle_wakeup_penalty = float(idle_wakeup_penalty)
        self.sliver_penalty = float(sliver_penalty)
        self.tier_penalty_weight = float(tier_penalty_weight)
        self.order_bias_weight = float(order_bias_weight)

        # Online streaming state
        self._completed_buckets: List[TimeWindowBucket] = []
        self._current_window_idx: Optional[int] = None
        self._window_start: float = 0.0
        self._win_arrivals: int = 0
        self._win_cpu: float = 0.0
        self._win_ram: float = 0.0
        self._win_heavy: int = 0
        self._latest_forecast: Optional[DemandForecast] = None
        self._last_observed_id: Optional[str] = None

    def reset_state(self) -> None:
        """Resets online streaming history between independent simulation runs."""
        self._completed_buckets.clear()
        self._current_window_idx = None
        self._window_start = 0.0
        self._win_arrivals = 0
        self._win_cpu = 0.0
        self._win_ram = 0.0
        self._win_heavy = 0
        self._latest_forecast = None
        self._last_observed_id = None

    def observe(self, workload: Workload) -> None:
        """Observes a newly arrived workload and updates rolling demand forecasts."""
        if self._last_observed_id == workload.workload_id:
            return
        self._last_observed_id = workload.workload_id

        t = workload.arrival_time
        if self._current_window_idx is None:
            self._current_window_idx = 0
            self._window_start = t

        while t >= self._window_start + self.window_size:
            heavy_ratio = (
                self._win_heavy / self._win_arrivals if self._win_arrivals > 0 else 0.0
            )
            bucket = TimeWindowBucket(
                window_index=self._current_window_idx,
                start_time=round(self._window_start, 4),
                end_time=round(self._window_start + self.window_size, 4),
                arrival_count=self._win_arrivals,
                cpu_demand=round(self._win_cpu, 4),
                ram_demand=round(self._win_ram, 4),
                heavy_job_ratio=round(heavy_ratio, 4),
            )
            self._completed_buckets.append(bucket)
            self._current_window_idx += 1
            self._window_start += self.window_size
            self._win_arrivals = 0
            self._win_cpu = 0.0
            self._win_ram = 0.0
            self._win_heavy = 0

            # Refresh forecast if enough history is available
            try:
                self._latest_forecast = self.predictor.predict_next(self._completed_buckets)
            except (ValueError, RuntimeError):
                self._latest_forecast = None

        self._win_arrivals += 1
        self._win_cpu += workload.cpu
        self._win_ram += workload.ram
        if workload.cpu >= 4.0:
            self._win_heavy += 1

    def _compute_burst_factor(self) -> float:
        """Returns a multiplier in [0.5, 2.0] indicating predicted demand surge vs historical mean."""
        if not self._completed_buckets or self._latest_forecast is None:
            return 1.0
        avg_ram = sum(b.ram_demand for b in self._completed_buckets) / len(self._completed_buckets)
        if avg_ram <= 1e-6:
            return 1.0
        ratio = self._latest_forecast.predicted_ram / avg_ram
        return min(max(ratio, 0.5), 2.0)

    def _select(self, workload: Workload, cluster: Cluster) -> Optional[CloudServer]:
        if self._last_observed_id != workload.workload_id:
            self.observe(workload)

        candidates = [s for s in cluster.servers if s.can_allocate(workload)]
        if not candidates:
            return None

        burst_factor = self._compute_burst_factor()
        is_small_job = (
            workload.cpu <= self.small_cpu_threshold
            and workload.ram <= self.small_ram_threshold
        )

        best_server: Optional[CloudServer] = None
        best_score = float("inf")

        for idx, s in enumerate(candidates):
            rem_cpu = s.available_cpu() - workload.cpu
            rem_ram = s.available_ram() - workload.ram
            rem_cpu_norm = rem_cpu / s.total_cpu
            rem_ram_norm = rem_ram / s.total_ram

            # 1. Cross-dimensional skew: keep CPU% and RAM% aligned
            skew = abs(rem_cpu_norm - rem_ram_norm)

            # 2. Anti-sliver guard: penalize leaving an unusable RAM sliver (0.15 to 1.5 GB)
            sliver_pen = (
                self.sliver_penalty if (0.15 < rem_ram < 1.5) else 0.0
            )

            # 3. Tier-aware headroom: reserve 32-CPU/64-GB nodes for batch jobs when small nodes exist
            is_large_server = s.total_cpu >= 32.0 or s.total_ram >= 64.0
            tier_pen = (
                (self.tier_penalty_weight * burst_factor)
                if (is_small_job and is_large_server)
                else 0.0
            )

            # 4. Sequential turnover preference
            order_bias = idx * self.order_bias_weight

            score = (
                (self.skew_weight * skew)
                + (self.packing_weight * rem_ram_norm)
                + sliver_pen
                + tier_pen
                + order_bias
            )

            if score < best_score - 1e-12:
                best_score = score
                best_server = s

        return best_server


class UCBBanditAllocator(Allocator):
    """Contextual Multi-Armed Bandit (UCB1) allocator that learns online which placement
    heuristic performs best across discrete (cluster_load, job_size) states.
    """

    def __init__(self, exploration_c: float = 0.6) -> None:
        self.exploration_c = float(exploration_c)
        self.arms: List[Allocator] = [
            PredictiveAllocator(),
            FirstFit(),
            BestFit(),
            WorstFit(),
        ]
        self._counts: Dict[Tuple[int, int], List[int]] = {}
        self._values: Dict[Tuple[int, int], List[float]] = {}

    def reset_state(self) -> None:
        """Resets bandit counts and values."""
        self._counts.clear()
        self._values.clear()
        for arm in self.arms:
            if hasattr(arm, "reset_state"):
                arm.reset_state()
            elif hasattr(arm, "reset"):
                arm.reset()

    def _get_context(self, workload: Workload, cluster: Cluster) -> Tuple[int, int]:
        ram_util = cluster.ram_utilization
        load_bucket = 0 if ram_util < 40.0 else (1 if ram_util < 75.0 else 2)
        size_bucket = 0 if (workload.cpu <= 4.0 and workload.ram <= 16.0) else 1
        return (load_bucket, size_bucket)

    def _select_arm(self, context: Tuple[int, int]) -> int:
        if context not in self._counts:
            self._counts[context] = [0] * len(self.arms)
            self._values[context] = [0.0] * len(self.arms)

        counts = self._counts[context]
        values = self._values[context]

        # Play each arm at least once per context, starting with PredictiveAllocator (arm 0)
        for idx, cnt in enumerate(counts):
            if cnt == 0:
                return idx

        total_plays = sum(counts)
        best_idx = 0
        best_ucb = -float("inf")
        for idx in range(len(self.arms)):
            bonus = self.exploration_c * math.sqrt(math.log(total_plays) / counts[idx])
            ucb_score = values[idx] + bonus
            if ucb_score > best_ucb:
                best_ucb = ucb_score
                best_idx = idx
        return best_idx

    def _update_reward(self, context: Tuple[int, int], arm_idx: int, reward: float) -> None:
        self._counts[context][arm_idx] += 1
        n = self._counts[context][arm_idx]
        old_val = self._values[context][arm_idx]
        self._values[context][arm_idx] = old_val + (reward - old_val) / n

    def observe(self, workload: Workload) -> None:
        """Notifies bandit arms (specifically PredictiveAllocator) of workload arrivals."""
        for arm in self.arms:
            if hasattr(arm, "observe"):
                arm.observe(workload)

    def _select(self, workload: Workload, cluster: Cluster) -> Optional[CloudServer]:
        self.observe(workload)
        context = self._get_context(workload, cluster)
        arm_idx = self._select_arm(context)
        chosen_server = self.arms[arm_idx]._select(workload, cluster)

        # Fallback if chosen arm fails (though all arms check can_allocate)
        if chosen_server is None:
            for idx, arm in enumerate(self.arms):
                chosen_server = arm._select(workload, cluster)
                if chosen_server is not None:
                    arm_idx = idx
                    break

        if chosen_server is None:
            self._update_reward(context, arm_idx, -1.0)
            return None

        # Compute immediate placement quality reward in [0, 1]
        rem_cpu = (chosen_server.available_cpu() - workload.cpu) / chosen_server.total_cpu
        rem_ram = (chosen_server.available_ram() - workload.ram) / chosen_server.total_ram
        balance_score = 1.0 - abs(rem_cpu - rem_ram)
        tier_bonus = (
            0.25
            if (workload.cpu <= 8.0 and workload.ram <= 16.0 and chosen_server.total_cpu <= 8.0)
            else 0.0
        )
        reward = min(1.0, 0.75 * balance_score + tier_bonus)
        self._update_reward(context, arm_idx, reward)
        return chosen_server
