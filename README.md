---
title: CloudPilot Control Plane
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
---

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
│   ├── config.py            # Heterogeneous server specs and cluster configurations
│   ├── exceptions.py        # Domain-specific allocation error hierarchy
│   ├── heuristics.py        # Placement policies (FirstFit, BestFit, WorstFit, RoundRobin)
│   ├── resource.py          # CloudServer resource model and capacity logic
│   ├── simulator.py         # Discrete-event TraceSimulator and SimulationResult
│   └── workload.py          # Workload model with temporal dynamics (arrival & duration)
├── backend/                 # Production HTTP Control Plane & Live Dashboard
│   ├── __init__.py          # Public backend exports
│   ├── server.py            # Zero-dependency ThreadingHTTPServer & REST API
│   ├── service.py           # Thread-safe CloudPilotService fleet manager
│   └── smoke_test.py        # CLI end-to-end smoke test & healthcheck suite
├── datasets/                # Workload datasets and trace loaders
│   ├── __init__.py
│   ├── sample_alibaba_trace.csv # Generated 1,000-workload Alibaba cluster trace
│   ├── synthetic.py         # Skewed synthetic workload generator
│   └── trace_loader.py      # Real cloud trace loader (Alibaba / Google Borg schemas)
├── docs/                    # Research documentation and deployment guides
│   └── DEPLOY.md            # Multi-cloud deployment guide (Render, HF Spaces, Cloud Run)
├── experiments/             # Allocation experiments and evaluation scripts
│   ├── __init__.py
│   ├── benchmark_suite.py   # Multi-scenario benchmark engine and latency profiler
│   ├── run_baselines.py     # Static benchmark suite for baseline heuristics
│   ├── run_benchmark_suite.py # Master benchmark runner and report generator
│   ├── run_ml_prediction.py # ML workload demand forecasting benchmark
│   ├── run_optimization_benchmark.py # Intelligent ML & RL allocation benchmark
│   ├── run_statistical_eval.py # Zero-leakage 30-seed statistical evaluation suite
│   └── run_trace_benchmark.py # Dynamic discrete-event trace replay benchmark
├── ml/                      # Machine learning demand forecasting models & optimizers
│   ├── __init__.py          # Public ML API exports
│   ├── features.py          # TimeWindowBucket and lag/diurnal feature engineering
│   ├── metrics.py           # Regression metrics (MAE, RMSE, sMAPE, R2)
│   ├── optimizer.py         # PredictiveAllocator (ML-guided) and UCBBanditAllocator (RL)
│   └── predictor.py         # Forecasters (Naive, Holt-EWMA, and Ridge LinearAR)
├── results/                 # Benchmark output metrics (.json and .csv)
│   ├── audit_report.md
│   ├── baseline_metrics.csv
│   ├── baseline_metrics.json
│   ├── benchmark_report.md
│   ├── benchmark_suite.csv
│   ├── benchmark_suite.json
│   ├── ml_prediction_metrics.csv
│   ├── ml_prediction_metrics.json
│   ├── optimization_metrics.csv
│   ├── optimization_metrics.json
│   ├── statistical_eval.csv
│   ├── statistical_eval.json
│   ├── statistical_eval_report.md
│   ├── trace_metrics.csv
│   └── trace_metrics.json
├── tests/                   # Automated unit and integration test suite
│   ├── __init__.py
│   ├── test_backend_api.py  # Control plane API unit & live HTTP integration tests
│   ├── test_benchmark_suite.py # Benchmark suite runner & latency telemetry tests
│   ├── test_cluster.py      # Cluster aggregate metrics and lifecycle tests
│   ├── test_heuristics.py   # Heuristic placement and SLA constraint tests
│   ├── test_ml_prediction.py # ML feature engineering, forecasters, and metrics tests
│   ├── test_optimizer.py    # PredictiveAllocator & UCBBanditAllocator tests
│   ├── test_resource.py     # CloudServer test cases (capacity, atomic safety)
│   ├── test_synthetic.py    # Workload generator determinism and bounds tests
│   ├── test_trace_simulator.py # Discrete-event simulator & trace loader tests
│   └── test_workload.py     # Workload test cases (validation, equality)
├── main.py                  # Simulator entry point and demonstration
├── requirements.txt         # Project dependencies
├── Dockerfile               # Production container image definition (python:3.12-slim)
├── .dockerignore            # Container build exclusions
├── render.yaml              # Render Blueprint deployment definition
├── .github/workflows/ci.yml # Multi-version Python CI pipeline
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

### Intelligent ML & RL Resource Optimization Benchmark (Step 5)
Benchmark `PredictiveAllocator` (with EWMA & LinearAR) and `UCBBanditAllocator` (contextual UCB1 multi-armed bandit meta-scheduler) head-to-head against traditional baselines on the Alibaba cluster trace:

```bash
python experiments/run_optimization_benchmark.py
```

Results are printed to stdout and saved in `results/optimization_metrics.json` and `results/optimization_metrics.csv`.

### Comprehensive Performance Benchmark & Scalability Sweep (Step 6)
Execute the multi-scenario benchmark engine across steady, diurnal, batch, and flash-crowd stress workloads, as well as a cluster scalability sweep (20, 50, 100 heterogeneous servers) with microsecond-level decision latency profiling:

```bash
python experiments/run_benchmark_suite.py
```

Results are exported to `results/benchmark_suite.json`, `results/benchmark_suite.csv`, and a formatted Markdown report in `results/benchmark_report.md`.

### Zero-Leakage 30-Seed Statistical Evaluation Suite (Step 8)
Execute rigorous cross-validation across 20, 50, and 100 heterogeneous servers under low (0.6x), standard (1.0x), and surge (1.5x) offered loads with `LinearARForecaster` fitted strictly on disjoint training seeds (seeds 901–903):

```bash
python experiments/run_statistical_eval.py
```

Results are saved to `results/statistical_eval.json`, `results/statistical_eval.csv`, and a formatted Markdown summary in `results/statistical_eval_report.md`.

---

## 6. Production Control-Plane API & Live Dashboard (Step 7)

CloudPilot includes a thread-safe HTTP REST API server and embedded dark-theme telemetry dashboard built purely on the Python standard library (`http.server` and `threading`), requiring zero third-party web frameworks.

### Launching the Control Plane Server
```bash
# Launch control plane server on default host (127.0.0.1:8000)
python -m backend.server --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in any web browser to view the real-time cluster telemetry dashboard, switch placement policies, and trigger on-demand trace simulations.

### REST API Endpoints
| Method | Endpoint | Description | Status Code |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Embedded Live Dark-Theme Control Plane Dashboard | `200 OK` |
| `GET` | `/api/v1/health` | Service uptime, active policy, and server count | `200 OK` |
| `GET` | `/api/v1/cluster` | Real-time CPU/RAM utilization and server status | `200 OK` |
| `POST` | `/api/v1/policy` | Dynamically switch policy (`{"policy": "predictive_ar"}`) | `200 OK`, `400 Bad Request` |
| `POST` | `/api/v1/allocate` | Allocate workload (`{"workload_id": "job-1", "cpu": 4, "ram": 8}`) | `201 Created`, `409 Conflict`, `400 Bad Request` |
| `POST` | `/api/v1/release` | Release workload (`{"workload_id": "job-1"}`) | `200 OK`, `404 Not Found` |
| `POST` | `/api/v1/reset` | Reset cluster fleet to idle empty state | `200 OK` |
| `POST` | `/api/v1/simulate` | Run on-demand trace simulation (`{"policy": "...", "num_workloads": 200}`) | `200 OK`, `400 Bad Request` |

### CLI Smoke Test & Healthcheck Suite
CloudPilot provides an automated 7-step smoke test verifying end-to-end API lifecycle operations (healthcheck, policy switching, allocation, telemetry, release, simulation, and cluster reset) with latency profiling:

```bash
# Run against an auto-spawned ephemeral server
python -m backend.smoke_test

# Or run against an existing running instance
python -m backend.smoke_test --url http://127.0.0.1:8000
```

### Docker Container Deployment
CloudPilot packages as a zero-dependency, non-root `python:3.12-slim` container with an internal `urllib.request` healthcheck:

```bash
# Build the container image
docker build -t cloudpilot .

# Run the container with port forwarding
docker run -d -p 8000:8000 --name cloudpilot-server cloudpilot

# Check health and container logs
docker ps
docker logs cloudpilot-server
```

---

## 7. Multi-Platform Free Cloud Deployment (Step 8)

CloudPilot is production-hardened with rate-limiting, security headers (`X-Content-Type-Options: nosniff`), concurrency gates, API key authentication (`CLOUDPILOT_API_KEY`), and pre-configured deployment targets. See [docs/DEPLOY.md](docs/DEPLOY.md) for full walkthroughs:
- **Render**: Blueprint-driven web deployment via `render.yaml`.
- **Hugging Face Spaces**: Interactive containerized demo via Docker SDK and `app_port: 8000`.
- **Google Cloud Run**: Serverless container execution with auto-scaling down to zero (`gcloud run deploy --source .`).

---

## 8. How to Execute Tests

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

## 9. Roadmap

- [x] Multi-server placement heuristics (First-Fit, Best-Fit, Worst-Fit, Round-Robin).
- [x] Synthetic workload generation and baseline benchmarks.
- [x] Real-world trace replay (Alibaba Cluster Trace / Google Borg schemas and discrete-event TraceSimulator).
- [x] Machine learning model integration for predictive workload sizing (Naive, Holt-EWMA, and Ridge LinearAR).
- [x] Intelligent resource optimization with PredictiveAllocator (whale protection, skew alignment) & Contextual UCB Multi-Armed Bandit meta-scheduling.
- [x] Comprehensive performance benchmarking, latency profiling, and multi-scenario stress testing (Step 6).
- [x] Production HTTP Control-Plane API, Live Dashboard, CLI smoke test & Docker deployment (Step 7).
- [x] Production hardening, security headers, concurrency gates, zero-leakage statistical evaluation, CI pipeline & multi-platform deployment (Step 8).

