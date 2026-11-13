# CloudPilot: ML-Based Cloud Resource Allocation Simulator

CloudPilot is an extensible research simulator for exploring intelligent cloud compute resource allocation, bin-packing heuristics, and ML-driven predictive scheduling.

> **Note**: This project is currently a discrete resource allocation simulation engine, not a live cloud deployment. It models compute nodes, workloads, capacity constraints, and lifecycle events without requiring real cloud infrastructure or third-party orchestrators.

---

## 1. Project Overview

Cloud resource management requires allocating heterogeneous compute workloads (demanding varying amounts of CPU and RAM) across a pool of servers while optimizing utilization and avoiding resource exhaustion. CloudPilot provides the core domain abstractions and verification mechanisms necessary to simulate and benchmark allocation policies, spanning:

- **Resource Modeling**: Modeling servers with finite CPU and RAM capacities.
- **Workload Modeling**: Modeling discrete jobs/services with explicit compute demands.
- **Atomic Operations**: Ensuring state consistency across allocations and releases.
- **Extensible Architecture**: Structured for future integration of heuristic algorithms (e.g., First-Fit, Best-Fit) and machine learning models (e.g., predictive autoscaling, reinforcement learning allocators).

---

## 2. Architecture & Design Decisions

### Directory Structure

```
CloudPilot/
├── allocation/              # Core domain package
│   ├── __init__.py          # Public API exports
│   ├── cluster.py           # Multi-server Cluster fleet manager
│   ├── exceptions.py        # Domain-specific allocation error hierarchy
│   ├── heuristics.py        # Placement policies (FirstFit, BestFit, WorstFit, RoundRobin)
│   ├── resource.py          # CloudServer resource model and capacity logic
│   ├── simulator.py         # Discrete-event TraceSimulator and SimulationResult
│   └── workload.py          # Workload model with temporal dynamics (arrival & duration)
├── backend/                 # Future API / backend services placeholder
├── datasets/                # Workload datasets and trace loaders
│   ├── __init__.py
│   ├── sample_alibaba_trace.csv # Generated 1,000-workload Alibaba cluster trace
│   ├── synthetic.py         # Skewed synthetic workload generator
│   └── trace_loader.py      # Real cloud trace loader (Alibaba / Google Borg schemas)
├── docs/                    # Research documentation and benchmarks
├── experiments/             # Allocation experiments and evaluation scripts
│   ├── __init__.py
│   ├── run_baselines.py     # Static benchmark suite for baseline heuristics
│   ├── run_ml_prediction.py # ML workload demand forecasting benchmark
│   └── run_trace_benchmark.py # Dynamic discrete-event trace replay benchmark
├── ml/                      # Machine learning demand forecasting models
│   ├── __init__.py          # Public ML API exports
│   ├── features.py          # TimeWindowBucket and lag/diurnal feature engineering
│   ├── metrics.py           # Regression metrics (MAE, RMSE, sMAPE, R2)
│   └── predictor.py         # Forecasters (Naive, Holt-EWMA, and Ridge LinearAR)
├── results/                 # Benchmark output metrics (.json and .csv)
│   ├── baseline_metrics.csv
│   ├── baseline_metrics.json
│   ├── ml_prediction_metrics.csv
│   ├── ml_prediction_metrics.json
│   ├── trace_metrics.csv
│   └── trace_metrics.json
├── tests/                   # Automated unit and integration test suite
│   ├── __init__.py
│   ├── test_cluster.py      # Cluster aggregate metrics and lifecycle tests
│   ├── test_heuristics.py   # Heuristic placement and SLA constraint tests
│   ├── test_ml_prediction.py # ML feature engineering, forecasters, and metrics tests
│   ├── test_resource.py     # CloudServer test cases (capacity, atomic safety)
│   ├── test_synthetic.py    # Workload generator determinism and bounds tests
│   ├── test_trace_simulator.py # Discrete-event simulator & trace loader tests
│   └── test_workload.py     # Workload test cases (validation, equality)
├── main.py                  # Simulator entry point and demonstration
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

### Architectural Decisions

1. **Atomic State Guarantees**:
   When allocating or releasing multi-dimensional resources (CPU and RAM), all constraints are validated *prior* to modifying server state. If either CPU or RAM exceeds capacity or bounds, the operation fails and leaves the server in its exact prior state.
2. **Polymorphic Allocation Interfaces**:
   `CloudServer.allocate()` and `CloudServer.release()` support both explicit numeric amounts (`cpu`, `ram`) and `Workload` domain objects, enabling flexibility for algorithm implementations.
3. **Floating-Point Precision Safeguards**:
   Resource calculations incorporate floating-point epsilon tolerances (`1e-9`) to prevent precision drift (e.g., `0.30000000000000004 > 0.3`) when allocating fractional compute cores.
4. **Zero-Dependency Core**:
   The simulator core relies purely on the Python standard library. External dependencies are reserved for developer tooling (such as `pytest`).
5. **Domain Exception Hierarchy**:
   All custom exceptions inherit from `AllocationError` and standard `ValueError` to facilitate standard Python error handling while allowing granular inspection (`InsufficientCPUError`, `InsufficientRAMError`, `ReleaseError`).

---

## 3. Setup Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.11)

### Setup Virtual Environment

```bash
# Clone the repository (if not already local)
git clone https://github.com/vaibhav-buildz/CloudPilot.git
cd CloudPilot

# Create and activate a virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (CMD):
.venv\Scripts\activate.bat
# On Linux / macOS:
source .venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
```

---

## 4. How to Run the Simulator

Run the demonstration script to view server provisioning, workload allocation, constraint enforcement, and resource deallocation:

```bash
# Using python executable
python main.py

# Or via virtual environment directly (Windows)
.venv\Scripts\python.exe main.py
```

### Sample Output

```
======================================================================
 CloudPilot - Cloud Resource Allocation Simulator
======================================================================

--- 1. Initial Server State (Idle) ---
  Server-Alpha    | CPU:  0.0/ 8.0 vCPU (  0.0%) | RAM:  0.0/16.0 GB (  0.0%) | Available: 8.0 vCPU, 16.0 GB
  Server-Beta     | CPU:  0.0/16.0 vCPU (  0.0%) | RAM:  0.0/32.0 GB (  0.0%) | Available: 16.0 vCPU, 32.0 GB

--- 2. Sample Workloads Created ---
  Workload[WL-Web-Frontend] (CPU: 4, RAM: 8GB)
  Workload[WL-Database] (CPU: 8, RAM: 16GB)
  Workload[WL-Analytics-Worker] (CPU: 4, RAM: 8GB)

--- 3. Allocating Workloads ---
  Allocating WL-Web-Frontend -> Server-Alpha...
  Allocating WL-Database -> Server-Beta...
  Allocating WL-Analytics-Worker -> Server-Beta...

  Attempting to allocate oversized WL-Oversized (8 vCPU) -> Server-Alpha (available: 4 vCPU)...
  [Rejected Expectedly] Insufficient CPU on server 'Server-Alpha': requested 8, available 4 (total 8)

--- 4. Server Utilization Post-Allocation ---
  Server-Alpha    | CPU:  4.0/ 8.0 vCPU ( 50.0%) | RAM:  8.0/16.0 GB ( 50.0%) | Available: 4.0 vCPU, 8.0 GB
  Server-Beta     | CPU: 12.0/16.0 vCPU ( 75.0%) | RAM: 24.0/32.0 GB ( 75.0%) | Available: 4.0 vCPU, 8.0 GB

--- 5. Releasing Resources ---
  Releasing WL-Web-Frontend from Server-Alpha...
  Releasing WL-Analytics-Worker from Server-Beta...

--- 6. Server Utilization Post-Release ---
  Server-Alpha    | CPU:  0.0/ 8.0 vCPU (  0.0%) | RAM:  0.0/16.0 GB (  0.0%) | Available: 8.0 vCPU, 16.0 GB
  Server-Beta     | CPU:  8.0/16.0 vCPU ( 50.0%) | RAM: 16.0/32.0 GB ( 50.0%) | Available: 8.0 vCPU, 16.0 GB

======================================================================
 Simulation completed successfully.
======================================================================
```

---

## 5. How to Run Baseline Experiments

### Static Heuristic Benchmark
Execute the static baseline benchmarking suite (comparing First-Fit, Best-Fit, Worst-Fit, Round-Robin, First-Fit Decreasing, and Best-Fit Decreasing across 500 workloads):

```bash
python experiments/run_baselines.py
```

Results are printed to stdout and saved in `results/baseline_metrics.json` and `results/baseline_metrics.csv`.

### Dynamic Trace Replay Benchmark (Step 3)
Execute the discrete-event trace simulator across 1,000 Alibaba-style workloads (evaluating arrival timestamps, runtime durations, dynamic departures, and server capacity recycling):

```bash
python experiments/run_trace_benchmark.py
```

Results are printed to stdout and saved in `results/trace_metrics.json` and `results/trace_metrics.csv`.

### ML Workload Demand Forecasting Benchmark (Step 4)
Evaluate time-window aggregation (30s buckets) and walk-forward prediction comparing Naive Last-Value, EWMA (Holt Trend), and Multivariate Ridge Linear Autoregression:

```bash
python experiments/run_ml_prediction.py
```

Results are printed to stdout and saved in `results/ml_prediction_metrics.json` and `results/ml_prediction_metrics.csv`.

---

## 6. How to Execute Tests

The test suite runs using either Python's built-in `unittest` runner or `pytest`.

### Running with Python's built-in `unittest` (Zero Dependencies)
```bash
python -m unittest discover -s tests -v
```

### Running with `pytest`
```bash
python -m pytest -v
```

---

## 7. Roadmap

- [x] Multi-server placement heuristics (First-Fit, Best-Fit, Worst-Fit, Round-Robin).
- [x] Synthetic workload generation and baseline benchmarks.
- [x] Real-world trace replay (Alibaba Cluster Trace / Google Borg schemas and discrete-event TraceSimulator).
- [x] Machine learning model integration for predictive workload sizing (Naive, Holt-EWMA, and Ridge LinearAR).
- [ ] Reinforcement learning and predictive dynamic auto-scaler for multi-cluster sizing.

