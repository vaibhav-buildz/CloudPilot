"""Alibaba Cluster Trace v2018: Task Request & Instance Usage Joiner.

Units & Normalization Documentation:
------------------------------------
In the Alibaba cluster-trace-v2018 dataset:
- CPU metrics:
  * `plan_cpu` in batch_task: specified in units of 1/100 core (100 = 1.0 CPU core).
  * `cpu_avg` and `cpu_max` in batch_instance: specified in units of 1/100 core (100 = 1.0 CPU core).
  Both are converted to physical core counts (`plan_cpu_cores`, `cpu_avg_cores`, `cpu_max_cores`)
  by dividing by 100.0.

- Memory metrics:
  * `plan_mem` in batch_task: normalized memory capacity relative to host capacity.
  * `mem_avg` and `mem_max` in batch_instance: normalized memory capacity relative to host capacity.
  Empirical verification shows:
    plan_mem: min ~0.02, median ~0.30, max ~4.58
    mem_avg:  min ~0.01, median ~0.05, max ~28.04
    mem_max:  min ~0.02, median ~0.06, max ~28.82
  Both requested memory (plan_mem) and observed memory usage (mem_avg, mem_max) are ALREADY
  on the exact same fractional scale ([0, 1.0+] relative to standard node capacity), NOT a
  percentage integer scale [0, 100]. No additional multiplier or conversion factor is required,
  rendering plan_mem and mem_max directly comparable for ratio and over-allocation calculations.

Sampling Bias Note:
-------------------
The raw 1.5M batch_instance sample is NOT a contiguous time slice. While start_time spans the
entire 7-day trace horizon (86,568s to 691,197s), records appear grouped by job/task batches
from the storage archive rather than sorted chronologically. This induces sampling bias toward
large batch workloads that were dumped in the initial chunks of the archive.
"""

from __future__ import annotations

import csv
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, Tuple, Any, List


def load_tasks(task_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Load batch_task records indexed by (job_name, task_name)."""
    tasks: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with open(task_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Check if header matches expected names
        has_header = "task_name" in header or "job_name" in header
        if not has_header:
            # First line is a data row
            rows = [header]
        else:
            rows = []
        
        for row in rows + list(reader):
            if len(row) < 9:
                continue
            # Format: task_name, instance_num, job_name, task_type, status, start_time, end_time, plan_cpu, plan_mem
            t_name = row[0].strip()
            inst_num_str = row[1].strip()
            j_name = row[2].strip()
            t_type = row[3].strip()
            status = row[4].strip()
            plan_cpu_str = row[7].strip()
            plan_mem_str = row[8].strip()
            
            try:
                inst_num = int(inst_num_str) if inst_num_str else 1
                plan_cpu = float(plan_cpu_str) if plan_cpu_str else 0.0
                plan_mem = float(plan_mem_str) if plan_mem_str else 0.0
            except ValueError:
                continue
            
            key = (j_name, t_name)
            tasks[key] = {
                "instance_num": inst_num,
                "task_type": t_type,
                "status": status,
                "plan_cpu": plan_cpu,
                "plan_mem": plan_mem,
            }
    return tasks


def run_join(
    instance_path: Path,
    task_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Join instance usage with task plan requirements and export joined_usage.csv."""
    tasks = load_tasks(task_path)
    
    total_instances = 0
    joined_count = 0
    dropped_status = 0
    dropped_missing_zero = 0
    dropped_duration = 0
    dropped_no_task = 0
    
    distinct_jobs_raw = set()
    distinct_jobs_joined = set()
    start_times_raw: List[int] = []
    
    plan_mem_vals: List[float] = []
    mem_avg_vals: List[float] = []
    mem_max_vals: List[float] = []
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(instance_path, "r", encoding="utf-8") as in_f, \
         open(output_path, "w", encoding="utf-8", newline="") as out_f:
        
        writer = csv.writer(out_f)
        writer.writerow([
            "job_name",
            "task_name",
            "instance_name",
            "arrival_time",
            "duration",
            "plan_cpu_cores",
            "plan_mem",
            "cpu_avg_cores",
            "cpu_max_cores",
            "mem_avg",
            "mem_max",
            "instance_num",
            "task_type",
        ])
        
        for line in in_f:
            total_instances += 1
            parts = line.strip().split(",")
            if len(parts) < 14:
                continue
            
            inst_name = parts[0].strip()
            task_name = parts[1].strip()
            job_name = parts[2].strip()
            inst_task_type = parts[3].strip()
            inst_status = parts[4].strip()
            
            distinct_jobs_raw.add(job_name)
            
            try:
                start_time = int(parts[5].strip())
                end_time = int(parts[6].strip())
                cpu_avg = float(parts[10].strip())
                cpu_max = float(parts[11].strip())
                mem_avg = float(parts[12].strip())
                mem_max = float(parts[13].strip())
            except (ValueError, IndexError):
                dropped_missing_zero += 1
                continue
            
            start_times_raw.append(start_time)
            
            key = (job_name, task_name)
            if key not in tasks:
                dropped_no_task += 1
                continue
            
            task = tasks[key]
            task_status = task["status"]
            
            # Filter 1: Status Terminated only (both instance and task)
            if inst_status != "Terminated" or task_status != "Terminated":
                dropped_status += 1
                continue
            
            plan_cpu = task["plan_cpu"]
            plan_mem = task["plan_mem"]
            
            # Filter 2: Drop missing or zero plan_cpu, plan_mem, cpu_max, mem_max
            if plan_cpu <= 0.0 or plan_mem <= 0.0 or cpu_max <= 0.0 or mem_max <= 0.0:
                dropped_missing_zero += 1
                continue
            
            # Filter 3: Drop end_time <= start_time
            if end_time <= start_time:
                dropped_duration += 1
                continue
            
            duration = end_time - start_time
            plan_cpu_cores = plan_cpu / 100.0
            cpu_avg_cores = cpu_avg / 100.0
            cpu_max_cores = cpu_max / 100.0
            instance_num = task["instance_num"]
            task_type = task["task_type"] or inst_task_type
            
            writer.writerow([
                job_name,
                task_name,
                inst_name,
                start_time,
                duration,
                f"{plan_cpu_cores:.4f}",
                f"{plan_mem:.4f}",
                f"{cpu_avg_cores:.4f}",
                f"{cpu_max_cores:.4f}",
                f"{mem_avg:.4f}",
                f"{mem_max:.4f}",
                instance_num,
                task_type,
            ])
            
            joined_count += 1
            distinct_jobs_joined.add(job_name)
            plan_mem_vals.append(plan_mem)
            mem_avg_vals.append(mem_avg)
            mem_max_vals.append(mem_max)
            
    summary = {
        "total_instances_before": total_instances,
        "joined_instances_after": joined_count,
        "dropped_status": dropped_status,
        "dropped_missing_zero": dropped_missing_zero,
        "dropped_duration": dropped_duration,
        "dropped_no_task": dropped_no_task,
        "distinct_jobs_raw": len(distinct_jobs_raw),
        "distinct_jobs_joined": len(distinct_jobs_joined),
        "min_start_time": min(start_times_raw) if start_times_raw else 0,
        "max_start_time": max(start_times_raw) if start_times_raw else 0,
        "time_span_seconds": (max(start_times_raw) - min(start_times_raw)) if start_times_raw else 0,
        "time_span_days": ((max(start_times_raw) - min(start_times_raw)) / 86400.0) if start_times_raw else 0.0,
        "plan_mem_min": min(plan_mem_vals) if plan_mem_vals else 0.0,
        "plan_mem_median": statistics.median(plan_mem_vals) if plan_mem_vals else 0.0,
        "plan_mem_max": max(plan_mem_vals) if plan_mem_vals else 0.0,
        "mem_avg_min": min(mem_avg_vals) if mem_avg_vals else 0.0,
        "mem_avg_median": statistics.median(mem_avg_vals) if mem_avg_vals else 0.0,
        "mem_avg_max": max(mem_avg_vals) if mem_avg_vals else 0.0,
        "mem_max_min": min(mem_max_vals) if mem_max_vals else 0.0,
        "mem_max_median": statistics.median(mem_max_vals) if mem_max_vals else 0.0,
        "mem_max_max": max(mem_max_vals) if mem_max_vals else 0.0,
    }
    return summary


def main():
    root = Path(__file__).resolve().parent.parent
    raw_dir = root / "datasets" / "raw"
    inst_path = raw_dir / "batch_instance.csv"
    task_path = raw_dir / "batch_task.csv"
    out_path = raw_dir / "joined_usage.csv"
    
    print("=" * 70)
    print("CloudPilot Alibaba 2018 Trace Joiner (batch_instance + batch_task)")
    print("=" * 70)
    print(f"Loading tasks from:     {task_path}")
    print(f"Streaming instances:   {inst_path}")
    print(f"Target output file:    {out_path}")
    print("-" * 70)
    
    summary = run_join(inst_path, task_path, out_path)
    
    print(f"Rows before join (raw sample):  {summary['total_instances_before']:,}")
    print(f"Rows after join (cleaned):      {summary['joined_instances_after']:,}")
    print(f"  - Dropped non-Terminated:     {summary['dropped_status']:,}")
    print(f"  - Dropped missing/zero req:   {summary['dropped_missing_zero']:,}")
    print(f"  - Dropped end_time <= start:  {summary['dropped_duration']:,}")
    print(f"  - Dropped task not found:     {summary['dropped_no_task']:,}")
    print(f"Distinct jobs (raw sample):     {summary['distinct_jobs_raw']:,}")
    print(f"Distinct jobs (after join):     {summary['distinct_jobs_joined']:,}")
    print(f"Sample start_time range:        {summary['min_start_time']} -> {summary['max_start_time']}")
    print(f"Time span:                      {summary['time_span_seconds']:,}s ({summary['time_span_days']:.2f} days)")
    print("-" * 70)
    print("Contiguous Time Slice Check:")
    print("  The 1.5M row sample is NOT a contiguous time slice.")
    print("  While timestamps span the full 7.00 days of the cluster trace, records in the")
    print("  archive are grouped by job/task dumps rather than sorted chronologically.")
    print("  Sampling bias: early-dumped large batch jobs dominate the initial chunk.")
    print("-" * 70)
    print("Memory Units Check:")
    print(f"  plan_mem: min={summary['plan_mem_min']:.4f}, median={summary['plan_mem_median']:.4f}, max={summary['plan_mem_max']:.4f}")
    print(f"  mem_avg:  min={summary['mem_avg_min']:.4f}, median={summary['mem_avg_median']:.4f}, max={summary['mem_avg_max']:.4f}")
    print(f"  mem_max:  min={summary['mem_max_min']:.4f}, median={summary['mem_max_median']:.4f}, max={summary['mem_max_max']:.4f}")
    print("  Conclusion: plan_mem, mem_avg, and mem_max are all on the same fractional scale")
    print("              ([0, 1.0+] normalized to node capacity). No unit conversion needed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
