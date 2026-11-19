# Phase 2 — Gate 3b Final Empirical Results & Tables

Generated automatically by `scripts/gate3b.py` from `results/gate3b.json`. Zero hand-typed numbers.

---

## Table 1: Policy Feasibility & Validation Selection at 1.0% Target (Margin Grid up to 4.00)

| Split | Policy | Feasible on Val? | Margin ($m$) | Val Viol (%) | Val CPU Saved (%) | Test Viol (%) | Test CPU Saved (%) | Exceeds 2x Test? | True Gen Failure? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TIME | `LinearRegression` | No (Dropped) | 4.00 | 1.15% | 57.76% | 1.03% | 53.94% | No | No |
| TIME | `DecisionTree` | **Yes** | 3.05 | 0.76% | 68.46% | 0.82% | 65.23% | No | No |
| TIME | `RandomForest` | **Yes** | 2.70 | 0.99% | 72.00% | 1.11% | 69.11% | No | No |
| TIME | `GradientBoosting` | **Yes** | 2.30 | 0.98% | 76.13% | 1.18% | 73.81% | No | No |
| TIME | `GBR_quantile_95` | **Yes** | 1.30 | 0.85% | 79.12% | 1.17% | 77.36% | No | No |
| TIME | `GBR_quantile_99` | **Yes** | 1.00 | 0.74% | 75.15% | 0.99% | 73.99% | No | No |
| TIME | `Naive` | **Yes** | 1.75 | 0.91% | 71.94% | 1.06% | 69.85% | No | No |
| TIME | `RollingMax48` | **Yes** | 1.20 | 0.82% | 74.87% | 0.85% | 72.50% | No | No |
| TIME | `RollingP99_7d` | **Yes** | 1.10 | 0.84% | 76.89% | 1.27% | 75.94% | No | No |
| TIME | `SeasonalNaive` | **Yes** | 3.15 | 0.99% | 63.08% | 0.89% | 58.24% | No | No |
| TIME | `EWMA` | **Yes** | 2.95 | 0.99% | 68.39% | 1.13% | 66.08% | No | No |
| TIME | `FleetPeakOfTraining` | **Yes** | 1.00 | 0.00% | -376.57% | 0.00% | -376.55% | No | No |
| TIME | `FleetPercentile95` | **Yes** | 1.75 | 0.98% | 41.62% | 1.25% | 41.62% | No | No |
| TIME | `FleetPercentile99` | **Yes** | 1.00 | 0.83% | 39.63% | 1.07% | 39.64% | No | No |
| TIME | `PeakOfTraining` | **Yes** | 1.00 | 0.86% | 72.54% | 2.08% | 72.54% | Yes | **Yes** |
| TIME | `Percentile95` | **Yes** | 1.50 | 0.88% | 75.08% | 2.13% | 75.07% | Yes | **Yes** |
| TIME | `Percentile99` | **Yes** | 1.20 | 0.85% | 75.47% | 2.20% | 75.47% | Yes | **Yes** |
| TIME | `Static` | **Yes** | 1.00 | 0.00% | 0.00% | 0.00% | 0.00% | No | No |
| APP | `LinearRegression` | **Yes** | 2.55 | 0.99% | 71.55% | 1.23% | 75.32% | No | No |
| APP | `DecisionTree` | **Yes** | 3.30 | 0.69% | 63.01% | 0.72% | 67.80% | No | No |
| APP | `RandomForest` | **Yes** | 2.40 | 0.98% | 73.14% | 1.10% | 76.82% | No | No |
| APP | `GradientBoosting` | **Yes** | 2.10 | 0.96% | 76.50% | 1.22% | 79.74% | No | No |
| APP | `GBR_quantile_95` | **Yes** | 1.25 | 0.87% | 78.29% | 0.89% | 80.80% | No | No |
| APP | `GBR_quantile_99` | **Yes** | 1.00 | 0.67% | 75.49% | 0.63% | 77.11% | No | No |
| APP | `Naive` | **Yes** | 1.70 | 0.85% | 72.26% | 1.06% | 75.61% | No | No |
| APP | `RollingMax48` | **Yes** | 1.15 | 0.84% | 75.41% | 0.94% | 78.00% | No | No |
| APP | `RollingP99_7d` | **Yes** | 1.10 | 0.85% | 75.70% | 1.02% | 80.32% | No | No |
| APP | `SeasonalNaive` | **Yes** | 2.50 | 0.99% | 67.53% | 1.15% | 72.37% | No | No |
| APP | `EWMA` | **Yes** | 2.60 | 0.99% | 71.01% | 1.26% | 75.08% | No | No |
| APP | `FleetPeakOfTraining` | **Yes** | 1.00 | 0.00% | -371.17% | 0.00% | -392.84% | No | No |
| APP | `FleetPercentile95` | **Yes** | 2.05 | 0.93% | 29.17% | 0.79% | 25.91% | No | No |
| APP | `FleetPercentile99` | **Yes** | 1.30 | 0.82% | 26.50% | 0.79% | 23.12% | No | No |
| APP | `PeakOfTraining` | **Yes** | 1.00 | 0.00% | 0.00% | 0.00% | 0.00% | No | No |
| APP | `Percentile95` | **Yes** | 1.00 | 0.00% | 0.00% | 0.00% | 0.00% | No | No |
| APP | `Percentile99` | **Yes** | 1.00 | 0.00% | 0.00% | 0.00% | 0.00% | No | No |
| APP | `Static` | **Yes** | 1.00 | 0.00% | 0.00% | 0.00% | 0.00% | No | No |
| APP_TIME | `LinearRegression` | **Yes** | 3.00 | 0.96% | 67.42% | 1.46% | 68.33% | No | No |
| APP_TIME | `DecisionTree` | No (Dropped) | 4.00 | 1.13% | 57.37% | 2.63% | 58.08% | Yes | No |
| APP_TIME | `RandomForest` | **Yes** | 2.35 | 0.95% | 74.50% | 1.65% | 75.17% | No | No |
| APP_TIME | `GradientBoosting` | **Yes** | 2.10 | 0.89% | 77.40% | 1.74% | 77.91% | No | No |
| APP_TIME | `GBR_quantile_95` | **Yes** | 1.25 | 0.91% | 79.07% | 1.36% | 79.23% | No | No |
| APP_TIME | `GBR_quantile_99` | **Yes** | 1.00 | 0.65% | 75.13% | 0.96% | 75.15% | No | No |
| APP_TIME | `Naive` | **Yes** | 1.70 | 0.84% | 72.96% | 1.25% | 72.76% | No | No |
| APP_TIME | `RollingMax48` | **Yes** | 1.15 | 0.79% | 75.70% | 1.13% | 75.44% | No | No |
| APP_TIME | `RollingP99_7d` | **Yes** | 1.10 | 0.81% | 75.77% | 1.47% | 79.33% | No | No |
| APP_TIME | `SeasonalNaive` | **Yes** | 2.50 | 0.93% | 68.12% | 1.84% | 70.02% | No | No |
| APP_TIME | `EWMA` | **Yes** | 2.55 | 0.99% | 72.28% | 1.71% | 73.03% | No | No |
| APP_TIME | `FleetPeakOfTraining` | **Yes** | 1.00 | 0.00% | -371.16% | 0.00% | -392.90% | No | No |
| APP_TIME | `FleetPercentile95` | **Yes** | 2.10 | 0.97% | 30.74% | 0.86% | 27.54% | No | No |
| APP_TIME | `FleetPercentile99` | **Yes** | 1.30 | 0.86% | 28.54% | 0.84% | 25.24% | No | No |
| APP_TIME | `PeakOfTraining` | **Yes** | 1.00 | 0.00% | 0.00% | 0.00% | 0.00% | No | No |
| APP_TIME | `Percentile95` | **Yes** | 1.00 | 0.00% | 0.00% | 0.00% | 0.00% | No | No |
| APP_TIME | `Percentile99` | **Yes** | 1.00 | 0.00% | 0.00% | 0.00% | 0.00% | No | No |
| APP_TIME | `Static` | **Yes** | 1.00 | 0.00% | 0.00% | 0.00% | 0.00% | No | No |

---

## Table 2: Paired 500-Draw App-Level Bootstrap Comparison at 1.0% Target

| Split | Policy | Status | Margin | Test Viol (%) | Test CPU Saved [95% CI] | Diff vs Best ML [95% CI] | Diff Spans 0? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TIME** | **Best ML: `GBR_quantile_95`** | **Selected** | **1.30** | **1.17%** | **77.36% [75.68%, 79.1%]** | Baseline Reference | — |
| TIME | `RollingP99_7d` | Feasible | 1.10 | 1.27% | 75.94% [73.93%, 77.93%] | +1.43% [+0.66%, +2.32%] | **No** |
| TIME | `EWMA` | Feasible | 2.95 | 1.13% | 66.08% [63.09%, 69.09%] | +11.29% [+9.86%, +12.69%] | **No** |
| TIME | `SeasonalNaive` | Feasible | 3.15 | 0.89% | 58.24% [54.59%, 61.95%] | +19.12% [+17.04%, +21.31%] | **No** |
| TIME | `RollingMax48` | Feasible | 1.20 | 0.85% | 72.50% [70.4%, 74.44%] | +4.85% [+4.05%, +5.66%] | **No** |
| TIME | `RollingP99_7d` | Feasible | 1.10 | 1.27% | 75.94% [73.93%, 77.93%] | +1.43% [+0.66%, +2.32%] | **No** |
| TIME | `FleetPercentile99` | Feasible | 1.00 | 1.07% | 39.64% [37.22%, 41.94%] | +37.72% [+34.79%, +40.86%] | **No** |
| TIME | `Percentile99` | Feasible | 1.20 | 2.20% | 75.47% [73.3%, 77.63%] | +1.90% [+1.04%, +2.99%] | **No** |
| TIME | `LinearRegression` | **DROPPED** | 4.00 | 1.03% | Never reached <=1% validation target even at m=4.00 (val_viol=1.15%) | N/A | N/A |
| TIME | `DecisionTree` | Feasible | 3.05 | 0.82% | 65.23% [62.28%, 68.09%] | +12.14% [+10.85%, +13.59%] | **No** |
| TIME | `RandomForest` | Feasible | 2.70 | 1.11% | 69.11% [66.39%, 71.84%] | +8.25% [+7.11%, +9.44%] | **No** |
| TIME | `GradientBoosting` | Feasible | 2.30 | 1.18% | 73.81% [71.51%, 76.1%] | +3.55% [+2.86%, +4.30%] | **No** |
| TIME | `GBR_quantile_99` | Feasible | 1.00 | 0.99% | 73.99% [72.62%, 75.37%] | +3.37% [+2.76%, +3.97%] | **No** |
| **APP** | **Best ML: `GBR_quantile_95`** | **Selected** | **1.25** | **0.89%** | **80.80% [77.89%, 83.71%]** | Baseline Reference | — |
| APP | `RollingP99_7d` | Feasible | 1.10 | 1.02% | 80.32% [76.89%, 83.95%] | +0.46% [-0.49%, +1.41%] | Yes |
| APP | `EWMA` | Feasible | 2.60 | 1.26% | 75.08% [70.2%, 80.03%] | +5.69% [+3.71%, +7.88%] | **No** |
| APP | `SeasonalNaive` | Feasible | 2.50 | 1.15% | 72.37% [67.24%, 77.58%] | +8.39% [+6.11%, +10.76%] | **No** |
| APP | `RollingMax48` | Feasible | 1.15 | 0.94% | 78.00% [74.51%, 81.59%] | +2.78% [+1.78%, +3.86%] | **No** |
| APP | `RollingP99_7d` | Feasible | 1.10 | 1.02% | 80.32% [76.89%, 83.95%] | +0.46% [-0.49%, +1.41%] | Yes |
| APP | `FleetPercentile99` | Feasible | 1.30 | 0.79% | 23.12% [16.57%, 28.93%] | +57.91% [+51.80%, +64.51%] | **No** |
| APP | `Percentile99` | **DROPPED** | 1.00 | 0.00% | Dropped on unseen apps (per-container training usage undefined) | N/A | N/A |
| APP | `LinearRegression` | Feasible | 2.55 | 1.23% | 75.32% [70.69%, 80.02%] | +5.46% [+3.72%, +7.42%] | **No** |
| APP | `DecisionTree` | Feasible | 3.30 | 0.72% | 67.80% [62.19%, 73.68%] | +12.95% [+9.83%, +15.87%] | **No** |
| APP | `RandomForest` | Feasible | 2.40 | 1.10% | 76.82% [72.54%, 81.32%] | +3.95% [+2.40%, +5.53%] | **No** |
| APP | `GradientBoosting` | Feasible | 2.10 | 1.22% | 79.74% [76.11%, 83.54%] | +1.05% [+0.06%, +2.02%] | **No** |
| APP | `GBR_quantile_99` | Feasible | 1.00 | 0.63% | 77.11% [74.54%, 79.72%] | +3.71% [+3.02%, +4.36%] | **No** |
| **APP_TIME** | **Best ML: `GBR_quantile_95`** | **Selected** | **1.25** | **1.36%** | **79.23% [76.11%, 82.48%]** | Baseline Reference | — |
| APP_TIME | `RollingP99_7d` | Feasible | 1.10 | 1.47% | 79.33% [75.88%, 83.11%] | -0.13% [-1.17%, +0.92%] | Yes |
| APP_TIME | `EWMA` | Feasible | 2.55 | 1.71% | 73.03% [67.93%, 78.29%] | +6.15% [+4.12%, +8.26%] | **No** |
| APP_TIME | `SeasonalNaive` | Feasible | 2.50 | 1.84% | 70.02% [64.49%, 75.77%] | +9.16% [+6.56%, +11.74%] | **No** |
| APP_TIME | `RollingMax48` | Feasible | 1.15 | 1.13% | 75.44% [71.81%, 79.28%] | +3.74% [+2.17%, +5.09%] | **No** |
| APP_TIME | `RollingP99_7d` | Feasible | 1.10 | 1.47% | 79.33% [75.88%, 83.11%] | -0.13% [-1.17%, +0.92%] | Yes |
| APP_TIME | `FleetPercentile99` | Feasible | 1.30 | 0.84% | 25.24% [18.9%, 30.88%] | +54.21% [+47.82%, +60.92%] | **No** |
| APP_TIME | `Percentile99` | **DROPPED** | 1.00 | 0.00% | Dropped on unseen apps (per-container training usage undefined) | N/A | N/A |
| APP_TIME | `LinearRegression` | Feasible | 3.00 | 1.46% | 68.33% [62.35%, 74.47%] | +10.83% [+7.88%, +13.79%] | **No** |
| APP_TIME | `DecisionTree` | **DROPPED** | 4.00 | 2.63% | Never reached <=1% validation target even at m=4.00 (val_viol=1.13%) | N/A | N/A |
| APP_TIME | `RandomForest` | Feasible | 2.35 | 1.65% | 75.17% [70.51%, 79.98%] | +4.01% [+2.36%, +5.71%] | **No** |
| APP_TIME | `GradientBoosting` | Feasible | 2.10 | 1.74% | 77.91% [73.86%, 82.04%] | +1.29% [+0.19%, +2.41%] | **No** |
| APP_TIME | `GBR_quantile_99` | Feasible | 1.00 | 0.96% | 75.15% [72.48%, 77.77%] | +4.10% [+3.14%, +5.03%] | **No** |

---

## Table 2b: Paired 500-Draw Bootstrap: Point-ML GradientBoosting vs Rules at 1.0% Target

| Split | Comparison | GB Margin | GB Test Viol (%) | Rule Margin | Rule Test Viol (%) | GB Saved (%) [95% CI] | Rule Saved (%) [95% CI] | Paired Diff (GB - Rule) [95% CI] | Diff Spans 0? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TIME | `GradientBoosting vs RollingMax48` | 2.30 | 1.18% | 1.20 | 0.85% | 73.81% [71.51%, 76.1%] | 72.50% [70.4%, 74.44%] | +1.31% [+0.29%, +2.45%] | **No** |
| TIME | `GradientBoosting vs RollingP99_7d` | 2.30 | 1.18% | 1.10 | 1.27% | 73.81% [71.51%, 76.1%] | 75.94% [73.93%, 77.93%] | -2.12% [-3.14%, -1.12%] | **No** |
| TIME | `GradientBoosting vs EWMA` | 2.30 | 1.18% | 2.95 | 1.13% | 73.81% [71.51%, 76.1%] | 66.08% [63.09%, 69.09%] | +7.74% [+6.97%, +8.48%] | **No** |
| APP | `GradientBoosting vs RollingMax48` | 2.10 | 1.22% | 1.15 | 0.94% | 79.74% [76.11%, 83.54%] | 78.00% [74.51%, 81.59%] | +1.73% [+0.58%, +2.82%] | **No** |
| APP | `GradientBoosting vs RollingP99_7d` | 2.10 | 1.22% | 1.10 | 1.02% | 79.74% [76.11%, 83.54%] | 80.32% [76.89%, 83.95%] | -0.58% [-1.70%, +0.57%] | Yes |
| APP | `GradientBoosting vs EWMA` | 2.10 | 1.22% | 2.60 | 1.26% | 79.74% [76.11%, 83.54%] | 75.08% [70.2%, 80.03%] | +4.64% [+3.55%, +5.88%] | **No** |
| APP_TIME | `GradientBoosting vs RollingMax48` | 2.10 | 1.74% | 1.15 | 1.13% | 77.91% [73.86%, 82.04%] | 75.44% [71.81%, 79.28%] | +2.44% [+0.56%, +4.19%] | **No** |
| APP_TIME | `GradientBoosting vs RollingP99_7d` | 2.10 | 1.74% | 1.10 | 1.47% | 77.91% [73.86%, 82.04%] | 79.33% [75.88%, 83.11%] | -1.42% [-2.82%, -0.10%] | **No** |
| APP_TIME | `GradientBoosting vs EWMA` | 2.10 | 1.74% | 2.55 | 1.71% | 77.91% [73.86%, 82.04%] | 73.03% [67.93%, 78.29%] | +4.86% [+3.73%, +5.98%] | **No** |

---

## Table 3: Hypothesis 2 (H2) Generalization Breakdown (135 Core Checks)

### A. Breakdown by Target (Raw vs. Conditional on Validation Feasibility)

| Target | 2x Test Limit | Total Active Checks | Exceeded 2x (Raw) | Raw Failure Rate (%) | Met Val Target | Exceeded 2x among Val-Feasible | Conditional Failure Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5.0%** | <= 10.0% | 42 | 0 | 0.0% | 42 | 0 | 0.0% |
| **1.0%** | <= 2.0% | 42 | 4 | 9.5% | 40 | 3 | 7.5% |
| **0.1%** | <= 0.20% | 42 | 23 | 54.8% | 20 | 4 | 20.0% |

### B. Breakdown by Split (Raw vs. Conditional on Validation Feasibility)

| Split | Total Active Checks | Exceeded 2x (Raw) | Raw Failure Rate (%) | Met Val Target | Exceeded 2x among Val-Feasible | Conditional Failure Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TIME** | 42 | 13 | 31.0% | 33 | 6 | 18.2% |
| **APP** | 42 | 6 | 14.3% | 35 | 0 | 0.0% |
| **APP_TIME** | 42 | 8 | 19.0% | 34 | 1 | 2.9% |

### C. Overall Generalization Summary

- **Total Checks**: 126 (15 core policies x 3 targets x 3 splits across 126 active test series)
- **Raw Test Exceedances (> 2x target, As Written in Pre-Registration)**: **27 / 126** (21.4%) -> **H2 NOT SUPPORTED as written**
- **Policies Meeting Validation Target (Feasible)**: 102 / 126 (81.0%)
- **Conditional True Generalization Failures (met val <= target, but test > 2x target)**: **7 / 102** (6.9%) -> **Secondary Finding: 93.1% Calibration Fidelity**

---

## Table 4: Exceeded Configurations Breakdown

| Split | Target | Policy | Selected Margin | Val Viol (%) | Test Viol (%) | Exceed Factor | Failure Classification |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| TIME | 1.0% | `PeakOfTraining` | 1.00 | 0.86% | 2.08% | 2.08x | **True Generalization Failure** (Met Val Target) |
| TIME | 1.0% | `Percentile95` | 1.50 | 0.88% | 2.13% | 2.13x | **True Generalization Failure** (Met Val Target) |
| TIME | 1.0% | `Percentile99` | 1.20 | 0.85% | 2.20% | 2.20x | **True Generalization Failure** (Met Val Target) |
| APP_TIME | 1.0% | `DecisionTree` | 4.00 | 1.13% | 2.63% | 2.63x | Unmet Validation Target (Failed on Val) |
| TIME | 0.1% | `RollingP99_7d` | 2.25 | 0.10% | 0.21% | 2.10x | **True Generalization Failure** (Met Val Target) |
| TIME | 0.1% | `SeasonalNaive` | 4.00 | 0.70% | 0.64% | 6.40x | Unmet Validation Target (Failed on Val) |
| TIME | 0.1% | `EWMA` | 4.00 | 0.55% | 0.59% | 5.90x | Unmet Validation Target (Failed on Val) |
| TIME | 0.1% | `PeakOfTraining` | 2.15 | 0.10% | 0.31% | 3.10x | **True Generalization Failure** (Met Val Target) |
| TIME | 0.1% | `Percentile95` | 4.00 | 0.14% | 0.33% | 3.30x | Unmet Validation Target (Failed on Val) |
| TIME | 0.1% | `Percentile99` | 3.50 | 0.10% | 0.24% | 2.40x | **True Generalization Failure** (Met Val Target) |
| TIME | 0.1% | `LinearRegression` | 4.00 | 1.15% | 1.03% | 10.30x | Unmet Validation Target (Failed on Val) |
| TIME | 0.1% | `DecisionTree` | 4.00 | 0.41% | 0.45% | 4.50x | Unmet Validation Target (Failed on Val) |
| TIME | 0.1% | `RandomForest` | 4.00 | 0.50% | 0.54% | 5.40x | Unmet Validation Target (Failed on Val) |
| TIME | 0.1% | `GradientBoosting` | 4.00 | 0.23% | 0.29% | 2.90x | Unmet Validation Target (Failed on Val) |
| APP | 0.1% | `SeasonalNaive` | 4.00 | 0.39% | 0.57% | 5.70x | Unmet Validation Target (Failed on Val) |
| APP | 0.1% | `EWMA` | 4.00 | 0.40% | 0.43% | 4.30x | Unmet Validation Target (Failed on Val) |
| APP | 0.1% | `LinearRegression` | 4.00 | 0.35% | 0.43% | 4.30x | Unmet Validation Target (Failed on Val) |
| APP | 0.1% | `DecisionTree` | 4.00 | 0.54% | 0.58% | 5.80x | Unmet Validation Target (Failed on Val) |
| APP | 0.1% | `RandomForest` | 4.00 | 0.36% | 0.40% | 4.00x | Unmet Validation Target (Failed on Val) |
| APP | 0.1% | `GradientBoosting` | 4.00 | 0.18% | 0.21% | 2.10x | Unmet Validation Target (Failed on Val) |
| APP_TIME | 0.1% | `RollingP99_7d` | 2.50 | 0.10% | 0.34% | 3.40x | **True Generalization Failure** (Met Val Target) |
| APP_TIME | 0.1% | `SeasonalNaive` | 4.00 | 0.37% | 0.98% | 9.80x | Unmet Validation Target (Failed on Val) |
| APP_TIME | 0.1% | `EWMA` | 4.00 | 0.38% | 0.54% | 5.40x | Unmet Validation Target (Failed on Val) |
| APP_TIME | 0.1% | `LinearRegression` | 4.00 | 0.49% | 0.76% | 7.60x | Unmet Validation Target (Failed on Val) |
| APP_TIME | 0.1% | `DecisionTree` | 4.00 | 1.13% | 2.63% | 26.30x | Unmet Validation Target (Failed on Val) |
| APP_TIME | 0.1% | `RandomForest` | 4.00 | 0.31% | 0.53% | 5.30x | Unmet Validation Target (Failed on Val) |
| APP_TIME | 0.1% | `GradientBoosting` | 4.00 | 0.17% | 0.37% | 3.70x | Unmet Validation Target (Failed on Val) |
