"""Evaluation metrics for workload demand forecasting."""

from __future__ import annotations

import math
from typing import Sequence


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean Absolute Error."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and of equal length.")
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root Mean Squared Error."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and of equal length.")
    mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)
    return math.sqrt(mse)


def smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Symmetric Mean Absolute Percentage Error (0% to 200%)."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and of equal length.")
    total = 0.0
    for t, p in zip(y_true, y_pred):
        denom = (abs(t) + abs(p)) / 2.0
        if denom > 1e-9:
            total += abs(t - p) / denom
    return (total / len(y_true)) * 100.0


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Coefficient of determination (R^2)."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and of equal length.")
    mean_y = sum(y_true) / len(y_true)
    ss_tot = sum((t - mean_y) ** 2 for t in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else 0.0
    return 1.0 - (ss_res / ss_tot)

