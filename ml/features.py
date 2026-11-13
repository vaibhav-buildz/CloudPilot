"""Time-series aggregation and lag feature engineering for cloud workload traces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple

from allocation.exceptions import InvalidResourceError
from allocation.workload import Workload


@dataclass(frozen=True)
class TimeWindowBucket:
    """Aggregated resource demand within a discrete time window [start_time, end_time)."""

    window_index: int
    start_time: float
    end_time: float
    arrival_count: int
    cpu_demand: float
    ram_demand: float
    heavy_job_ratio: float


def bucketize_trace(
    workloads: Iterable[Workload], window_size: float = 30.0
) -> List[TimeWindowBucket]:
    """Groups temporal workloads into contiguous fixed-width time windows."""
    if isinstance(window_size, bool) or not isinstance(window_size, (int, float)) or window_size <= 0:
        raise InvalidResourceError("window_size must be a positive number.")

    workload_list = sorted(workloads, key=lambda w: w.arrival_time)
    if not workload_list:
        return []

    min_time = workload_list[0].arrival_time
    max_time = workload_list[-1].arrival_time
    num_windows = max(int(math.floor((max_time - min_time) / window_size)) + 1, 1)

    counts = [0] * num_windows
    cpu_sums = [0.0] * num_windows
    ram_sums = [0.0] * num_windows
    heavy_counts = [0] * num_windows

    for w in workload_list:
        idx = int(math.floor((w.arrival_time - min_time) / window_size))
        idx = min(max(idx, 0), num_windows - 1)
        counts[idx] += 1
        cpu_sums[idx] += w.cpu
        ram_sums[idx] += w.ram
        if w.cpu >= 4.0:
            heavy_counts[idx] += 1

    buckets: List[TimeWindowBucket] = []
    for i in range(num_windows):
        start_t = round(min_time + i * window_size, 4)
        end_t = round(start_t + window_size, 4)
        cnt = counts[i]
        heavy_ratio = (heavy_counts[i] / cnt) if cnt > 0 else 0.0
        buckets.append(
            TimeWindowBucket(
                window_index=i,
                start_time=start_t,
                end_time=end_t,
                arrival_count=cnt,
                cpu_demand=round(cpu_sums[i], 4),
                ram_demand=round(ram_sums[i], 4),
                heavy_job_ratio=round(heavy_ratio, 4),
            )
        )
    return buckets


def build_feature_vector(
    history: List[TimeWindowBucket],
    lags: int = 4,
    horizon_period: float = 3600.0,
) -> List[float]:
    """Constructs a feature vector from the last `lags` buckets for autoregressive prediction."""
    if len(history) < lags:
        raise ValueError(f"Need at least {lags} history buckets, got {len(history)}.")

    recent = history[-lags:]
    features: List[float] = [1.0]  # Bias term

    # Lagged CPU, RAM, arrival counts, and heavy job ratio
    for b in recent:
        features.extend([
            b.cpu_demand,
            b.ram_demand,
            float(b.arrival_count),
            b.heavy_job_ratio,
        ])

    # Rolling statistics over the lag window
    mean_cpu = sum(b.cpu_demand for b in recent) / lags
    mean_ram = sum(b.ram_demand for b in recent) / lags
    mean_cnt = sum(b.arrival_count for b in recent) / lags

    # First-order momentum (latest minus earliest in lag window)
    cpu_momentum = recent[-1].cpu_demand - recent[0].cpu_demand
    ram_momentum = recent[-1].ram_demand - recent[0].ram_demand

    # Cyclical time encoding for diurnal/burst wave awareness
    next_time = recent[-1].end_time
    angle = (2.0 * math.pi * next_time) / max(horizon_period, 1.0)

    features.extend([
        mean_cpu,
        mean_ram,
        mean_cnt,
        cpu_momentum,
        ram_momentum,
        math.sin(angle),
        math.cos(angle),
    ])
    return features


# Synonym for backward and spec compatibility
extract_lag_features = build_feature_vector


def create_supervised_dataset(
    buckets: List[TimeWindowBucket],
    lags: int = 4,
    horizon_period: float = 3600.0,
) -> Tuple[List[List[float]], List[Tuple[float, float, float]]]:
    """Converts a sequence of buckets into (X, Y) where each y is (cpu_demand, ram_demand, arrival_count)."""
    if len(buckets) <= lags:
        return [], []

    X: List[List[float]] = []
    Y: List[Tuple[float, float, float]] = []
    for idx in range(lags, len(buckets)):
        history_slice = buckets[idx - lags : idx]
        target = buckets[idx]
        X.append(build_feature_vector(history_slice, lags=lags, horizon_period=horizon_period))
        Y.append((target.cpu_demand, target.ram_demand, float(target.arrival_count)))
    return X, Y

