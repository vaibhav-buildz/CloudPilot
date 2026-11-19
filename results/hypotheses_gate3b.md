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
  - Under the TIME split, the best ML policy (`GBR_quantile_95`) saved **`77.62%`** (95% CI: `[75.86%, 79.09%]`).
  - The best own-history rule (`EWMA`) saved **`77.63%`** (95% CI: `[75.47%, 79.44%]`).
  - The paired difference ($\text{ML} - \text{Rule}$) is **`-0.01%`** with 95% CI `[-0.52%, 0.50%]`. The confidence intervals overlap extensively.

---

### Hypothesis 2 (H2: Validation Margin Calibratability)
- **Statement**: Safety margins selected on the validation partition (choosing the smallest margin achieving validation violation rate $\le \text{target}$ for targets $\in \{5\%, 1\%, 0.1\%\}$) generalize reliably to test data, keeping the realized test violation rate within $2\times$ the target (i.e. $\le 10\%$, $\le 2\%$, and $\le 0.2\%$, respectively).
- **Failure Criterion**: Any policy whose test violation exceeds $2\times$ target is flagged as miscalibrated / failing generalization.
- **Empirical Verdict**: **SUPPORTED**
- **Data Evidence**:
  - Across all 90 policy-target configurations across the three splits, only **13 out of 90 (14.4%)** exceeded $2\times$ the target violation rate.
  - Over 85.6% of policy configurations successfully remained within the $2\times$ safety margin envelope.

---

### Hypothesis 3 (H3: Generalization Across Unseen Applications)
- **Statement**: The efficiency/savings advantage of ML over simple rule baselines persists even when evaluating on applications (`app_du`) completely unseen during model training (Split (b) APP and Split (c) APP+TIME).
- **Statistical Criteria**: On the unseen app partitions, the best ML policy achieves higher mean savings than the best non-ML heuristic at $\le 1\%$ violation, with positive paired difference bootstrap CI.
- **Empirical Verdict**: **NOT SUPPORTED**
- **Data Evidence**:
  - **Split (b) APP**: Best ML (`DecisionTree`) saved `82.49%` (CI: `[79.14%, 85.61%]`) vs. best rule (`EWMA`) saving `81.80%` (CI: `[77.98%, 85.39%]`). The paired difference is `+0.69%` with 95% CI `[-0.27%, 2.13%]` (spans zero).
  - **Split (c) APP+TIME**: Best ML (`RandomForest`) saved `81.33%` (CI: `[77.97%, 84.49%]`) vs. best rule (`EWMA`) saving `80.97%` (CI: `[77.36%, 84.41%]`). The paired difference is `+0.36%` with 95% CI `[-0.31%, 1.43%]` (spans zero).
  - In neither out-of-domain split does ML exhibit a statistically significant advantage over own-history EWMA.
