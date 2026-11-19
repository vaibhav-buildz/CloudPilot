# Addendum to Gate 3b Hypotheses & Empirical Verdicts

**Date**: 2026-11-19  
**Status of Initial Commit**:
Git history confirms that commit `62a07bf` is the first commit of `results/hypotheses_gate3b.md`. There is no prior commit of this file before the initial Gate 3b execution. The data evidence and verdict lines in `62a07bf` reflect the initial exploratory pipeline (which contained future-value `bfill` artifacts and test-set selection) and are hereby **superseded** by the leakage-free, validation-selected results presented below. The original file `results/hypotheses_gate3b.md` remains strictly preserved in its initial committed state.

**Reasons for Addendum**:
1. **Data Integrity (Zero Future Leakage)**: Eliminated all `.bfill()` calls from lag/rolling feature construction in `scripts/gate3b.py`. Initial undefined-lag warmup rows ($t < 49$) are now dropped per container. Train-only median imputation is applied (with 0 remaining NaNs).
2. **Strict Validation-Based Policy Selection**: Replaced test-set peeking with strict out-of-sample selection on the validation partition (`val_violation <= target` maximizing `val_saved_pct`).
3. **Disambiguation of Percentile Baselines**: Explicitly differentiated `Percentile99` (per-container training history, which defaults to request on unseen containers) from `FleetPercentile99` (fleet-wide training 99th percentile across all three splits).

---

## Updated Empirical Evidence & Verdicts (Validation-Selected Rerun)

### Hypothesis 1 (H1: ML Superiority over Strong Own-History Baselines)
- **Verdict**: **NOT SUPPORTED**
- **Evidence**:
  - In TIME split at 1.0% target:
    - Best ML (`GBR_quantile_95`): **77.36%** test saved (95% CI: `[75.68%, 79.10%]`).
    - Best Own-History Rule (`RollingP99_7d`): **75.94%** test saved (95% CI: `[73.93%, 77.93%]`).
    - Paired Difference ($\text{ML} - \text{Rule}$): **+1.43%** (95% CI: `[+0.66%, +2.32%]`).
  - **Verdict Criterion Check**: Although the paired difference is positive, the individual 95% bootstrap confidence intervals overlap ([75.68%, 79.10%] vs [73.93%, 77.93%]), failing the pre-registered requirement of non-overlapping bootstrap CIs.

---

### Hypothesis 2 (H2: Validation Margin Calibratability)
- **Pre-Registered Criterion**: "The margin chosen on validation keeps test violation within 2x target (<=10%, <=2%, and <=0.2%, respectively)."
- **Verdict (As Written)**: **NOT SUPPORTED**
  - **Raw Test Exceedances**: Across all active checks (126 evaluated policy-target-split series), **27 configurations (21.4%)** exceeded 2x target violation on test, failing the pre-registered calibratability criterion.
- **Secondary Finding (Conditional on Validation Feasibility)**:
  - When restricting analysis conditionally to policies that successfully met the validation target ($\text{val\_violation} \le \text{target}$), only **7 of 102 configurations (6.9%)** exceeded 2x target violation on test.
  - Over **93.1%** of validation-feasible policies remained strictly within the 2x target envelope.
  - The raw exceedances are driven by uncalibrated high-variance point models (such as `LinearRegression`, `DecisionTree`, and fixed heuristics at 0.1% target) that had already failed on the validation set, rather than post-selection distribution drift.
  - **By Target (Conditional True Failures)**:
    - **5.0% Target**: 0 / 42 (**0.0%**).
    - **1.0% Target**: 3 / 40 (**7.5%**), all three occurring in per-container historical percentiles (`PeakOfTraining`, `Percentile95`, `Percentile99`) on TIME.
    - **0.1% Target**: 4 / 20 (**20.0%**), driven by extreme tail volatility in historical rules. ML policies did not suffer true generalization failures at 0.1% (those exceeding test limits failed validation first).
  - **By Split (Conditional True Failures)**:
    - **TIME**: 6 / 33 (18.2%).
    - **APP**: 0 / 35 (**0.0%**).
    - **APP+TIME**: 1 / 34 (**2.9%**).

---

### Hypothesis 3 (H3: Generalization Across Unseen Applications)
- **Verdict**: **NOT SUPPORTED**
- **Evidence**:
  - **Split (b) APP**: Best ML (`GBR_quantile_95`) paired diff vs `RollingP99_7d` = **+0.46%** (95% CI: `[-0.49%, +1.41%]`).
  - **Split (c) APP+TIME**: Best ML (`GBR_quantile_95`) paired diff vs `RollingP99_7d` = **-0.13%** (95% CI: `[-1.17%, +0.92%]`).
  - Paired difference spans zero in both out-of-distribution splits; ML provides no statistically significant advantage over own-history rolling p99.
