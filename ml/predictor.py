"""Workload demand forecasting models (Naive, Holt-EWMA, and Ridge Autoregressive ML)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from ml.features import (
    TimeWindowBucket,
    build_feature_vector,
    create_supervised_dataset,
)


@dataclass(frozen=True)
class DemandForecast:
    """Predicted resource demand for the next time window."""

    predicted_cpu: float
    predicted_ram: float
    predicted_arrivals: float


class DemandPredictor(ABC):
    """Abstract base interface for cloud workload demand forecasters."""

    @abstractmethod
    def fit(self, buckets: List[TimeWindowBucket]) -> "DemandPredictor":
        """Fits model parameters on historical time-window buckets."""

    @abstractmethod
    def predict_next(self, recent_buckets: List[TimeWindowBucket]) -> DemandForecast:
        """Predicts demand for the immediately following time window."""


class NaiveLastValuePredictor(DemandPredictor):
    """Baseline forecaster that predicts the next window equals the most recent window."""

    def fit(self, buckets: List[TimeWindowBucket]) -> "NaiveLastValuePredictor":
        return self

    def predict_next(self, recent_buckets: List[TimeWindowBucket]) -> DemandForecast:
        if not recent_buckets:
            raise ValueError("recent_buckets cannot be empty.")
        last = recent_buckets[-1]
        return DemandForecast(
            predicted_cpu=max(0.0, last.cpu_demand),
            predicted_ram=max(0.0, last.ram_demand),
            predicted_arrivals=max(0.0, float(last.arrival_count)),
        )


class EWMAForecaster(DemandPredictor):
    """Double Exponential Smoothing (Holt linear trend) forecaster."""

    def __init__(self, alpha: float = 0.4, beta: float = 0.15) -> None:
        if not (0.0 < alpha <= 1.0) or not (0.0 <= beta <= 1.0):
            raise ValueError("alpha must be in (0, 1] and beta in [0, 1].")
        self.alpha = alpha
        self.beta = beta

    def fit(self, buckets: List[TimeWindowBucket]) -> "EWMAForecaster":
        return self

    def _holt_forecast(self, series: List[float]) -> float:
        if len(series) == 1:
            return max(0.0, series[0])
        level = series[0]
        trend = series[1] - series[0]
        for val in series[1:]:
            prev_level = level
            level = self.alpha * val + (1.0 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1.0 - self.beta) * trend
        return max(0.0, level + trend)

    def predict_next(self, recent_buckets: List[TimeWindowBucket]) -> DemandForecast:
        if not recent_buckets:
            raise ValueError("recent_buckets cannot be empty.")
        cpus = [b.cpu_demand for b in recent_buckets]
        rams = [b.ram_demand for b in recent_buckets]
        cnts = [float(b.arrival_count) for b in recent_buckets]
        return DemandForecast(
            predicted_cpu=round(self._holt_forecast(cpus), 4),
            predicted_ram=round(self._holt_forecast(rams), 4),
            predicted_arrivals=round(self._holt_forecast(cnts), 4),
        )


class LinearARForecaster(DemandPredictor):
    """Multivariate Ridge-Regularized Autoregressive ML model solved via pure-Python linear algebra."""

    def __init__(
        self,
        lags: int = 4,
        ridge_lambda: float = 1.5,
        horizon_period: float = 3600.0,
    ) -> None:
        if lags < 1:
            raise ValueError("lags must be >= 1.")
        if ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be >= 0.0.")
        self.lags = lags
        self.ridge_lambda = ridge_lambda
        self.horizon_period = horizon_period
        self.weights_cpu: List[float] = []
        self.weights_ram: List[float] = []
        self.weights_arrivals: List[float] = []
        self._is_fitted = False

    @staticmethod
    def _solve_ridge(X: List[List[float]], y: List[float], l2: float) -> List[float]:
        """Solves (X^T X + l2 * I) w = X^T y using Gauss-Jordan elimination with partial pivoting."""
        n_samples = len(X)
        n_features = len(X[0])

        # Compute A = X^T X + l2 * I (do not regularize bias term at index 0)
        A = [[0.0] * n_features for _ in range(n_features)]
        b = [0.0] * n_features

        for i in range(n_features):
            for j in range(i, n_features):
                s = sum(X[k][i] * X[k][j] for k in range(n_samples))
                if i == j and i > 0:
                    s += l2
                A[i][j] = s
                A[j][i] = s
            b[i] = sum(X[k][i] * y[k] for k in range(n_samples))

        # Augmented matrix [A | b]
        aug = [A[i] + [b[i]] for i in range(n_features)]

        for col in range(n_features):
            pivot_row = max(range(col, n_features), key=lambda r: abs(aug[r][col]))
            if abs(aug[pivot_row][col]) < 1e-12:
                continue
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

            pivot_val = aug[col][col]
            for j in range(col, n_features + 1):
                aug[col][j] /= pivot_val

            for row in range(n_features):
                if row != col:
                    factor = aug[row][col]
                    for j in range(col, n_features + 1):
                        aug[row][j] -= factor * aug[col][j]

        return [aug[i][n_features] for i in range(n_features)]

    def fit(self, buckets: List[TimeWindowBucket]) -> "LinearARForecaster":
        X, Y = create_supervised_dataset(
            buckets, lags=self.lags, horizon_period=self.horizon_period
        )
        if not X:
            raise ValueError(f"Insufficient buckets ({len(buckets)}) to fit with lags={self.lags}.")

        y_cpu = [target[0] for target in Y]
        y_ram = [target[1] for target in Y]
        y_cnt = [target[2] for target in Y]

        self.weights_cpu = self._solve_ridge(X, y_cpu, self.ridge_lambda)
        self.weights_ram = self._solve_ridge(X, y_ram, self.ridge_lambda)
        self.weights_arrivals = self._solve_ridge(X, y_cnt, self.ridge_lambda)
        self._is_fitted = True
        return self

    def predict_next(self, recent_buckets: List[TimeWindowBucket]) -> DemandForecast:
        if not self._is_fitted:
            raise RuntimeError("LinearARForecaster must be fitted before calling predict_next().")
        if len(recent_buckets) < self.lags:
            raise ValueError(f"Need at least {self.lags} recent buckets to predict.")

        features = build_feature_vector(
            recent_buckets[-self.lags :],
            lags=self.lags,
            horizon_period=self.horizon_period,
        )
        pred_cpu = sum(w * f for w, f in zip(self.weights_cpu, features))
        pred_ram = sum(w * f for w, f in zip(self.weights_ram, features))
        pred_cnt = sum(w * f for w, f in zip(self.weights_arrivals, features))

        return DemandForecast(
            predicted_cpu=round(max(0.0, pred_cpu), 4),
            predicted_ram=round(max(0.0, pred_ram), 4),
            predicted_arrivals=round(max(0.0, pred_cnt), 4),
        )

