"""Cloud workload trace loader and realistic trace generator (Alibaba / Google Borg schemas)."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Iterable, List, Union

from allocation.exceptions import InvalidResourceError
from allocation.workload import Workload


def load_csv_trace(
    filepath: Union[str, Path],
    cpu_scale: float = 1.0,
    ram_scale: float = 1.0,
    min_cpu: float = 0.05,
    min_ram: float = 0.05,
) -> List[Workload]:
    """Load a cloud workload trace from a CSV file.

    Supports multiple column naming conventions found in real-world traces:
    - Alibaba Cluster Trace (job_id / task_id, plan_cpu, plan_mem)
    - Google Borg Trace (timestamp, cpu_request, mem_request, runtime)
    - CloudPilot Native Schema (workload_id, arrival_time, duration, cpu_requirement, ram_requirement)

    Args:
        filepath: Path to the CSV trace file.
        cpu_scale: Multiplier for scaling normalized CPU values (e.g. Borg fraction to cores).
        ram_scale: Multiplier for scaling normalized RAM values (e.g. Borg fraction to GB).
        min_cpu: Minimum allowed CPU capacity floor.
        min_ram: Minimum allowed RAM capacity floor.

    Returns:
        List of Workload instances sorted by (arrival_time, workload_id).

    Raises:
        FileNotFoundError: If the specified trace file does not exist.
        InvalidResourceError: If essential CPU or RAM columns are missing or malformed.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    workloads: List[Workload] = []
    with path.open(mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            w_id = (
                row.get("workload_id")
                or row.get("job_id")
                or row.get("task_id")
                or f"trace-job-{idx:05d}"
            )
            raw_cpu = (
                row.get("cpu_requirement")
                or row.get("cpu")
                or row.get("plan_cpu")
                or row.get("cpu_request")
            )
            raw_ram = (
                row.get("ram_requirement")
                or row.get("ram")
                or row.get("plan_mem")
                or row.get("mem_request")
            )
            if raw_cpu is None or raw_ram is None or raw_cpu == "" or raw_ram == "":
                raise InvalidResourceError(f"Row {idx} missing CPU or RAM column: {row}")

            cpu_val = max(float(raw_cpu) * cpu_scale, min_cpu)
            ram_val = max(float(raw_ram) * ram_scale, min_ram)

            raw_arrival = (
                row.get("arrival_time")
                or row.get("timestamp")
                or row.get("submit_time")
                or row.get("start_time")
            )
            arrival_val = float(raw_arrival) if raw_arrival not in (None, "") else 0.0

            raw_duration = row.get("duration") or row.get("runtime") or row.get("exec_time")
            duration_val = float(raw_duration) if raw_duration not in (None, "") else None

            workloads.append(
                Workload(
                    workload_id=str(w_id).strip(),
                    cpu_requirement=round(cpu_val, 4),
                    ram_requirement=round(ram_val, 4),
                    arrival_time=round(arrival_val, 4),
                    duration=round(duration_val, 4) if duration_val is not None else None,
                )
            )

    workloads.sort(key=lambda w: (w.arrival_time, w.workload_id))
    return workloads


def save_csv_trace(workloads: Iterable[Workload], filepath: Union[str, Path]) -> Path:
    """Export an iterable of Workload instances to a CSV file.

    Args:
        workloads: Workloads to export.
        filepath: Destination file path.

    Returns:
        Resolved Path object of the exported file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["workload_id", "arrival_time", "duration", "cpu_requirement", "ram_requirement"])
        for w in workloads:
            writer.writerow([
                w.workload_id,
                f"{w.arrival_time:.2f}",
                f"{w.duration:.2f}" if w.duration is not None else "",
                f"{w.cpu_requirement:.2f}",
                f"{w.ram_requirement:.2f}",
            ])
    return path


def generate_alibaba_style_trace(
    n: int = 1000, seed: int = 42, horizon_seconds: float = 3600.0
) -> List[Workload]:
    """Synthesize a realistic cloud trace modeling Alibaba Cluster Trace characteristics.

    Models:
    - Diurnal and bursty job arrivals governed by modulated Poisson inter-arrival intervals.
    - Bimodal workload size distribution: 75% lightweight microservices/containers
      and 25% heavy batch analytics/ML jobs.
    - Correlated memory-to-CPU ratios with log-normally distributed runtimes.

    Args:
        n: Number of workloads to generate.
        seed: Random seed for deterministic generation.
        horizon_seconds: Nominal duration of the observation time window in seconds.

    Returns:
        List of synthetic Workload objects sorted by arrival_time.
    """
    rng = random.Random(seed)
    workloads: List[Workload] = []
    current_time = 0.0

    for i in range(n):
        progress = min(current_time / max(horizon_seconds, 1.0), 1.0)
        burst_multiplier = 1.0 + 0.75 * math.sin(progress * 2.0 * math.pi)
        mean_interarrival = (horizon_seconds / max(n, 1)) / burst_multiplier
        current_time += rng.expovariate(1.0 / max(mean_interarrival, 0.01))

        if rng.random() < 0.75:
            cpu = rng.choice([0.5, 1.0, 2.0, 4.0])
            ram = round(min(max(cpu * rng.uniform(1.5, 3.0), 0.5), 16.0), 2)
            duration = round(min(max(rng.lognormvariate(4.2, 0.7), 10.0), 1200.0), 2)
        else:
            cpu = rng.choice([4.0, 8.0, 12.0, 16.0])
            ram = round(min(max(cpu * rng.uniform(2.5, 4.0), 8.0), 64.0), 2)
            duration = round(min(max(rng.lognormvariate(5.5, 0.8), 30.0), 2400.0), 2)

        workloads.append(
            Workload(
                workload_id=f"alibaba-job-{i:04d}",
                cpu_requirement=cpu,
                ram_requirement=ram,
                arrival_time=round(current_time, 2),
                duration=duration,
            )
        )

    return workloads

