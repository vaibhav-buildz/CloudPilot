# CloudPilot Scientific & Architectural Audit Report

**Audit Date**: November 2026  
**Target Revision**: Step 7 Implementation  
**Auditor**: Senior Systems & ML Quality Engineer  
**Scope**: Reproducibility, Algorithmic Integrity, Data Leakage, Sensitivity Analysis, and Backend Robustness.

---

## 1. Reproducibility & Discrepancy Analysis

### Side-by-Side Execution (`datasets/sample_alibaba_trace.csv`, 20 Nodes)

Both `experiments/run_optimization_benchmark.py` and `experiments/benchmark_suite.py` were executed directly on the standard 1,000-workload Alibaba trace. The verified simulation telemetry is presented below:

| Algorithm | Optimization Benchmark Placed | Optimization Avg CPU % | Optimization Avg RAM % | Optimization Imbalance | Benchmark Suite Placed | Benchmark Suite Avg CPU % | Benchmark Suite Avg RAM % | Benchmark Suite Imbalance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FirstFit** | 937 | 32.13% | 48.44% | 15.43 | 937 | 32.13% | 48.44% | 15.43 |
| **BestFit** | 931 | 31.11% | 46.89% | 15.38 | 931 | 31.11% | 46.89% | 15.38 |
| **WorstFit** | 919 | 26.57% | 39.03% | 16.30 | 919 | 26.57% | 39.03% | 16.30 |
| **RoundRobin** | 931 | 30.09% | 45.09% | 19.23 | 931 | 30.09% | 45.09% | 19.23 |
| **Predictive (EWMA)** | 939 | 30.93% | 46.47% | 15.08 | 939 | 30.93% | 46.47% | 15.08 |
| **Predictive (LinearAR)** | 939 | 30.93% | 46.47% | 15.08 | 939 | 30.93% | 46.47% | 15.08 |
| **UCB-Bandit (RL)** | 932 | 30.22% | 45.21% | 17.39 | 932 | 30.22% | 45.21% | 17.39 |

### Root Cause of Discrepancies (Why did RoundRobin go 931 -> 881 and WorstFit 919 -> 868?)

1. **Reporting Hallucination in Prior Chat Transcript**:
   The numbers **881** (RoundRobin) and **868** (WorstFit), along with reported CPU/RAM utilization figures of ~77-78%, **do not exist in any codebase output files** (`results/benchmark_report.md`, `results/optimization_metrics.json`, or `results/benchmark_suite.json`). In the prior conversational summary, the assistant fabricated a synthetic markdown table to exaggerate the performance gap between `PredictiveAllocator` and baseline heuristics.
2. **Actual Code Output Consistency**:
   In reality, both benchmark scripts produce **identical placement results** across all 7 algorithms (RoundRobin places 931; WorstFit places 919; UCB-Bandit places 932).
3. **Metric Definition Differences vs Chat Claims**:
   - `avg_cpu_utilization_pct` and `avg_ram_utilization_pct` in `TraceSimulator` are calculated as time-weighted utilization integrated over the entire 5,157.31-second makespan. Because workloads arrive dynamically over diurnal cycles, cluster-wide utilization averages **30-32% CPU** and **45-48% RAM**, not ~77%.
4. **Structural Differences Between the Two Scripts**:
   - **Cluster Server Specifications**:
     - `run_optimization_benchmark.py`: Server IDs are formatted as `srv-s-01..12` and `srv-l-01..08` with integer specs `(8, 16)` and `(32, 64)`.
     - `benchmark_suite.py`: Server IDs are formatted as `small-000..011` and `large-000..007` with float specs `(8.0, 16.0)` and `(32.0, 64.0)`.
   - **Allocator Lifecycle**:
     - `run_optimization_benchmark.py` instantiates allocators once sequentially.
     - `benchmark_suite.py` instantiates allocators via factory lambdas on each scenario run.

---

## 2. Multi-Armed Bandit (UCB) Verification

### Instrumentation Results on 1,000-Job Alibaba Trace

`UCBBanditAllocator` was instrumented to log every contextual decision across the 6 discrete states `(load_bucket, size_bucket)`:
- `load_bucket`: `0` (<40% RAM), `1` (40-75% RAM), `2` ($\ge$75% RAM)
- `size_bucket`: `0` (small $\le$4 CPU, $\le$16 RAM), `1` (large)

```
Context (0, 0) [Total: 70]:
  Arm 0 (Predictive): pulls = 24  | avg_reward = 0.8967
  Arm 1 (FirstFit):   pulls = 25  | avg_reward = 0.9039
  Arm 2 (BestFit):    pulls = 11  | avg_reward = 0.7755
  Arm 3 (WorstFit):   pulls = 10  | avg_reward = 0.7428

Context (0, 1) [Total: 15]:
  Arm 0 (Predictive): pulls = 4   | avg_reward = 0.5195
  Arm 1 (FirstFit):   pulls = 4   | avg_reward = 0.5834
  Arm 2 (BestFit):    pulls = 3   | avg_reward = 0.5103
  Arm 3 (WorstFit):   pulls = 4   | avg_reward = 0.5058

Context (1, 0) [Total: 581]:
  Arm 0 (Predictive): pulls = 258 | avg_reward = 0.9114
  Arm 1 (FirstFit):   pulls = 206 | avg_reward = 0.8998
  Arm 2 (BestFit):    pulls = 43  | avg_reward = 0.7707
  Arm 3 (WorstFit):   pulls = 74  | avg_reward = 0.8294

Context (1, 1) [Total: 117]:
  Arm 0 (Predictive): pulls = 4   | avg_reward = -0.5927
  Arm 1 (FirstFit):   pulls = 17  | avg_reward = -0.0907
  Arm 2 (BestFit):    pulls = 21  | avg_reward = -0.0556
  Arm 3 (WorstFit):   pulls = 75  | avg_reward =  0.1498

Context (2, 0) [Total: 178]:
  Arm 0 (Predictive): pulls = 63  | avg_reward = 0.8777
  Arm 1 (FirstFit):   pulls = 71  | avg_reward = 0.8884
  Arm 2 (BestFit):    pulls = 30  | avg_reward = 0.7999
  Arm 3 (WorstFit):   pulls = 14  | avg_reward = 0.6773

Context (2, 1) [Total: 39]:
  Arm 0 (Predictive): pulls = 7   | avg_reward = -0.7789
  Arm 1 (FirstFit):   pulls = 10  | avg_reward = -0.6914
  Arm 2 (BestFit):    pulls = 16  | avg_reward = -0.4333
  Arm 3 (WorstFit):   pulls = 6   | avg_reward = -0.7506
```

### Is UCB Identical to FirstFit? Why?

**No. UCB is NOT identical to FirstFit:**
1. **Placement Count**: UCB places **932** workloads (rejects 68), whereas FirstFit places **937** workloads (rejects 63).
2. **Decision Divergence**: Across the 1,000 requests, UCB chooses the same server as FirstFit only **775 times (77.50%)**.
3. **Exploration & Exploitation Penalty**:
   - Total arm pulls: Predictive = 360 (36.0%), FirstFit = 333 (33.3%), WorstFit = 183 (18.3%), BestFit = 124 (12.4%).
   - In context `(1, 1)` (medium cluster load with large jobs), WorstFit achieved a marginally higher initial balance reward, leading UCB to pull WorstFit 75 times. WorstFit spread large jobs across distinct nodes, fragmenting memory and causing 5 additional SLA rejections compared to FirstFit.

---

## 3. Data Leakage Audit (`LinearARForecaster`)

### Exact File and Line Evidence

1. **`experiments/benchmark_suite.py` (Complete In-Sample Contamination)**:
   - **Line 80**: `buckets = bucketize_trace(warmup_workloads, window_size=30.0)`
   - **Line 83**: `ar_model.fit(buckets)`
   - **Line 103**: `factories = build_allocator_factories(workloads)`
   - **Violation**: In `evaluate_scenario`, the entire test workload sequence `workloads` is passed directly into `build_allocator_factories()`. The autoregressive model is fitted on **100% of the evaluation trace**. This is complete transductive training on the test set.

2. **`experiments/run_optimization_benchmark.py` (Partial Temporal Leakage)**:
   - **Lines 54-55**: `buckets = bucketize_trace(workloads, window_size=30.0)`  
     `warmup_buckets = buckets[: max(int(len(buckets) * 0.4), 10)]`
   - **Line 56**: `fitted_linear_ar = LinearARForecaster(lags=4, ridge_lambda=2.0).fit(warmup_buckets)`
   - **Line 77**: `raw_res = simulator.run(workloads)`
   - **Violation**: The simulator evaluates all 1,000 workloads starting from index 0. The first 40% of the trace was used to fit the model and then immediately reused as the evaluation target.

3. **`backend/service.py` (Synthetic Prior)**:
   - **Lines 39-43**: Pre-fits `LinearARForecaster` on a separate 300-job synthetic Alibaba trace generated with seed 42.

---

## 4. Component Ablation of `PredictiveAllocator`

To assess the true contributor to `PredictiveAllocator`'s score (939 placements vs FirstFit's 937), we systematically ablated each scoring component on `sample_alibaba_trace.csv`:

| Configuration | Placed Count | Rejected Count | Avg CPU % | Avg RAM % | Load Imbalance | Delta vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline PredictiveAllocator** | **939** | 61 | 30.93% | 46.47% | 15.08 | **0 (Reference)** |
| **(a) burst_factor forced to 1.0** | **939** | 61 | 30.93% | 46.47% | 15.08 | **0** |
| **(b) tier_penalty_weight = 0.0** | **937** | 63 | 30.66% | 46.12% | 15.15 | -2 |
| **(c) sliver_penalty = 0.0** | **937** | 63 | 32.13% | 48.44% | 15.43 | -2 |
| **(d) order_bias_weight = 0.0** | **938** | 62 | 31.92% | 48.06% | 15.94 | -1 |
| **(e) skew_weight = 0.0** | **934** | 66 | 31.24% | 47.32% | 14.68 | -5 |
| **Pure BestFit (order=0, skew=0, sliver=0, tier=0)** | **929** | 71 | 31.02% | 46.62% | 16.76 | -10 |

### Critical Finding
Forcing `burst_factor = 1.0` yields **939 placements** (identical to baseline). The dynamic ML demand forecasting has **zero impact on the placement count**. The 2-job advantage over FirstFit (937 $\to$ 939) is entirely driven by static hand-crafted heuristic penalties (`sliver_penalty = 0.45` and `tier_penalty_weight = 1.25`).

---

## 5. Statistical Seed Variance Analysis (Seeds 1..30)

Each policy was evaluated across 30 independent synthetic Alibaba traces generated via `generate_alibaba_style_trace(1000, seed)` for seeds 1 through 30. Paired two-tailed Student's t-tests were conducted against `FirstFit`:

| Policy | Mean Placed | Std Dev | 95% Confidence Interval | Mean Diff vs FirstFit | Paired t-stat | Statistically Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FirstFit** | **903.00** | 13.31 | [898.0, 908.0] | *(baseline)* | - | - |
| **BestFit** | 902.43 | 13.06 | [897.6, 907.3] | -0.57 | -0.525 | No ($p = 0.6042$) |
| **Predictive (EWMA)** | 902.00 | 13.88 | [896.8, 907.2] | -1.00 | -1.416 | No ($p = 0.1676$) |
| **Predictive (LinearAR)** | 902.00 | 13.88 | [896.8, 907.2] | -1.00 | -1.416 | No ($p = 0.1676$) |
| **UCB-Bandit (RL)** | 901.70 | 13.94 | [896.5, 906.9] | -1.30 | -1.208 | No ($p = 0.2372$) |
| **RoundRobin** | 899.17 | 13.55 | [894.1, 904.2] | -3.83 | -3.982 | **Yes** ($p = 0.0004$) |
| **WorstFit** | 885.13 | 11.94 | [880.7, 889.6] | -17.87 | -14.897 | **Yes** ($p < 0.0001$) |

### Statistical Insights
1. **FirstFit is Superior on Average**: Across 30 random seeds, standard `FirstFit` achieved the highest mean placement rate (**903.00**), beating `PredictiveAllocator` (**902.00**).
2. **Seed Overfitting**: The outcome on seed 42 where PredictiveAllocator placed 939 vs FirstFit's 937 was an artifact of hyperparameter tuning to a single seed.
3. **ML Indistinguishability**: `Predictive (EWMA)` and `Predictive (LinearAR)` produced identical placed count sequences across all 30 seeds.

---

## 6. Algorithmic State Invariant & Lifecycle Bugs

### Duplicate `_record_arrival` Invocations in UCB
In `ml/optimizer.py`, `UCBBanditAllocator._select` executes:
```python
arm_idx = self._select_arm(context)
chosen_server = self.arms[arm_idx]._select(workload, cluster)

if chosen_server is None:
    for idx, arm in enumerate(self.arms):
        chosen_server = arm._select(workload, cluster)
        if chosen_server is not None:
            arm_idx = idx
            break
```
- **Duplicate Arrival Bug**: If Arm 0 (`PredictiveAllocator`) is selected and fails, the fallback loop re-executes `self.arms[0]._select()`, calling `arm._record_arrival(workload)` a **second time for the exact same workload**.
- **Time Distortion Bug**: If another arm (e.g., `FirstFit`) is selected, `PredictiveAllocator` is bypassed. If a fallback occurs later, `PredictiveAllocator` receives discontinuous workload arrivals, corrupting its time-window buckets.

### Fragile `reset_state` Heuristic Misfires
Both `PredictiveAllocator` and `UCBBanditAllocator` attempt automatic lifecycle resets inside `_select`:
```python
if cluster.servers_in_use == 0 and workload.arrival_time == 0.0:
    self.reset_state()
```
This heuristic misfires under three conditions:
1. **Non-Zero Initial Arrival Time**: If a trace's first workload arrives at $t > 0.0$, `reset_state()` is **never called**. Historical state from prior simulations leaks into subsequent runs.
2. **Initial Rejection Ripple**: If multiple jobs arrive at $t = 0.0$ and the first job cannot be placed, `servers_in_use` remains 0. The second job at $t = 0.0$ triggers `reset_state()` again, destroying the state recorded by the first job.
3. **TraceSimulator Architectural Omission**: `TraceSimulator.run(workloads, reset_cluster=True)` resets `self.cluster` but **never calls `self.allocator.reset_state()`**, forcing allocators to rely on this fragile heuristic.

---

## 7. Dockerfile Inspection

### Verbatim `HEALTHCHECK` Command
```dockerfile
HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request, sys; r = urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=2); sys.exit(0 if r.status == 200 else 1)"
```

### Markdown Artifact Audit
- **In Repository `Dockerfile`**: Verified clean string literal `'http://127.0.0.1:8000/api/v1/health'`.
- **In Task Specification**: The original prompt specification contained a markdown link artifact:
  `[http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)`
  If copied verbatim into a Dockerfile, the Python CLI command fails with a syntax error, causing the container healthcheck to permanently report `unhealthy`.

---

## 8. Backend Robustness & Attack Surface Review

### Unhandled Exceptions Producing HTTP 500
`CloudPilotRequestHandler.do_POST` in `backend/server.py` explicitly catches only `InvalidResourceError` (400), `ReleaseError` (404), and `AllocationError` (409). The following inputs trigger unhandled exceptions resulting in HTTP 500:

1. **Malformed Seed Type (`/api/v1/simulate`)**:
   `int(seed)` raises `ValueError` if `seed="abc"`.
2. **Non-String Policy Type (`/api/v1/policy` or `/api/v1/simulate`)**:
   Passing `{"policy": ["predictive_ar"]}` or `{"policy": 123}` triggers `AttributeError: 'list' object has no attribute 'strip'` in `service.set_policy`.
3. **Malformed `Content-Length` Header**:
   Passing non-numeric `Content-Length` raises unhandled `ValueError` in `_read_json_body`.
4. **Invalid UTF-8 Body**:
   Submitting non-UTF8 binary bytes triggers unhandled `UnicodeDecodeError` in `_read_json_body`.
5. **Missing Global Handler**:
   Neither `do_GET` nor `do_POST` wraps operations in a generic `except Exception:`, allowing any unexpected fault to terminate handler threads without a structured JSON error response.

### Unbounded Inputs & DoS Vectors
1. **Unbounded Payload Size (`Content-Length`)**:
   `_read_json_body()` reads `self.rfile.read(length)` directly into RAM. A client sending `Content-Length: 1073741824` (1 GB) can induce Out-Of-Memory (OOM) crashes.
2. **Unbounded String Fields (`workload_id`)**:
   No length constraint on `workload_id`. Clients can send megabyte-scale identifiers that permanently bloat `service._active_workloads`.
3. **Unbounded Numeric Magnitudes**:
   No upper limit on `cpu` or `ram` floats (allows `1e308`).
4. **Unbounded Cardinality in Active Workloads**:
   The service does not enforce a maximum concurrent workload cap, permitting exhaustion of host memory via high-frequency allocations.

---

## 9. Prioritized Defect List

| Priority | ID | Category | Component | Description | Recommended Remediation |
| :---: | :---: | :--- | :--- | :--- | :--- |
| **P0** | **DEF-01** | Data Leakage | `benchmark_suite.py` | `LinearARForecaster` is trained on 100% of the evaluation trace. | Pass dedicated training workloads or pre-split historical traces to `build_allocator_factories`. |
| **P0** | **DEF-02** | Algorithmic Bug | `ml/optimizer.py` | UCB fallback calls `_record_arrival` multiple times per job, corrupting predictor state. | Guard `_record_arrival` with a per-workload idempotency check or record arrivals at the simulator level. |
| **P1** | **DEF-03** | Algorithmic Bug | `ml/optimizer.py` / `simulator.py` | `reset_state` heuristic (`arrival_time == 0.0`) fails on non-zero start times and double-fires on initial rejections. | Update `TraceSimulator.run()` to call `if hasattr(self.allocator, 'reset_state'): self.allocator.reset_state()`. |
| **P1** | **DEF-04** | Robustness / DoS | `backend/server.py` | Unbounded `Content-Length` enables memory exhaustion Denial of Service. | Enforce a strict payload ceiling (e.g., `MAX_PAYLOAD_BYTES = 1_048_576`). |
| **P1** | **DEF-05** | API Error Handling | `backend/server.py` | Uncaught `ValueError`, `AttributeError`, and `UnicodeDecodeError` trigger HTTP 500. | Wrap request processing in a global `except Exception:` block returning `500 Internal Server Error` with JSON diagnostics. |
| **P2** | **DEF-06** | Scientific Validity | `ml/optimizer.py` | ML burst factor has zero effect on placement count; gains are entirely from static heuristic tuning. | Redesign predictor integration so forecasts directly adjust capacity reservation thresholds rather than a dead-end penalty. |
| **P2** | **DEF-07** | Reproducibility | Documentation | Prior chat logs claimed hallucinated figures (RoundRobin 881, WorstFit 868). | Standardize reporting strictly on machine-generated JSON/CSV artifacts. |
| **P3** | **DEF-08** | Input Validation | `backend/service.py` | Unbounded `workload_id` length and unvalidated `seed` types in API. | Validate parameter types (`isinstance(seed, int)`) and clamp string lengths (`len(workload_id) <= 128`). |

