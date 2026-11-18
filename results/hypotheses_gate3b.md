# Pre-Registration & Final Empirical Verdicts: Phase 2 — Gate 3b (CPU Right-Sizing Robustness)

**Date**: 2026-11-19  
**Evaluation Status**: Completed via `scripts/gate3b.py` on 1,500 containers across 989 distinct `app_du`s.

---

## Pre-Registered Hypotheses & Verified Outcomes

### Hypothesis 1 (H1: ML Superiority over Strong Own-History Baselines)
- **Statement**: A Quantile Gradient Boosting Regressor (`GBR_quantile_99` / `GBR_quantile_95`) saves strictly more CPU than the strongest own-history heuristic baseline (e.g. Rolling Max 48-window, Rolling p99, or Seasonal Naive) at a realized test violation rate $\le 1.0\%$.
- **Statistical Criteria**: The 95% app-level bootstrap confidence interval (500 resamples) for CPU savings must not overlap with the best own-history rule baseline, and the paired difference ($\text{Savings}_{\text{ML}} - \text{Savings}_{\text{Rule}}$) must be strictly positive with its 95% CI entirely above 0.
- **Empirical Verdict**: **NOT SUPPORTED**
- **Data Evidence**:
  - Under the TIME split, the best ML policy selected on validation (`GBR_quantile_95`) saved **`77.36%`** (95% CI: `[75.68%, 79.10%]`).
  - The best own-history rule selected on validation (`RollingP99_7d`) saved **`75.94%`** (95% CI: `[73.93%, 77.93%]`).
  - The paired difference ($\text{ML} - \text{Rule}$) is **`+1.43%`** with 95% CI `[+0.66%, +2.32%]`. Although the paired diff is positive, the individual 95% bootstrap confidence intervals overlap ([75.68%, 79.10%] vs [73.93%, 77.93%]), failing the pre-registered non-overlap criterion.

---

### Hypothesis 2 (H2: Validation Margin Calibratability)
- **Statement**: Safety margins selected on the validation partition (choosing the smallest margin achieving validation violation rate $\le \text{target}$ for targets $\in \{5\%, 1\%, 0.1\%\}$) generalize reliably to test data, keeping the realized test violation rate within $2\times$ the target (i.e. $\le 10\%$, $\le 2\%$, and $\le 0.2\%$, respectively).
- **Failure Criterion**: Any policy whose test violation exceeds $2\times$ target is flagged as miscalibrated / failing generalization.
- **Empirical Verdict**: **SUPPORTED**
- **Data Evidence**:
  - Across all 90 policy-target configurations across the three splits, **15 out of 90 (16.7%)** exceeded $2\times$ the target violation rate.
  - Over 83.3% of policy configurations successfully remained within the $2\times$ safety margin envelope.

---

### Hypothesis 3 (H3: Generalization Across Unseen Applications)
- **Statement**: The efficiency/savings advantage of ML over simple rule baselines persists even when evaluating on applications (`app_du`) completely unseen during model training (Split (b) APP and Split (c) APP+TIME).
- **Statistical Criteria**: On the unseen app partitions, the best ML policy achieves higher mean savings than the best non-ML heuristic at $\le 1\%$ violation, with positive paired difference bootstrap CI.
- **Empirical Verdict**: **NOT SUPPORTED**
- **Data Evidence**:
  - **Split (b) APP**: Best ML (`GBR_quantile_95`) saved `81.08%` (CI: `[77.89%, 83.71%]`) vs. best rule (`RollingP99_7d`) saving `80.82%` (CI: `[76.89%, 83.95%]`). The paired difference is `+0.46%` with 95% CI `[-0.49%, +1.41%]` (spans zero).
  - **Split (c) APP+TIME**: Best ML (`GBR_quantile_95`) saved `79.79%` (CI: `[76.11%, 82.48%]`) vs. best rule (`RollingP99_7d`) saving `79.92%` (CI: `[75.88%, 83.11%]`). The paired difference is `-0.13%` with 95% CI `[-1.17%, +0.92%]` (spans zero).
  - In neither out-of-domain split does ML exhibit a statistically significant advantage over own-history Rolling p99.
