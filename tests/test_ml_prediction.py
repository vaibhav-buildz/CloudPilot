"""Unit tests for ML workload demand forecasting, feature engineering, and metrics."""

import math
import unittest

from allocation.exceptions import InvalidResourceError
from allocation.workload import Workload
from ml.features import (
    TimeWindowBucket,
    bucketize_trace,
    build_feature_vector,
    create_supervised_dataset,
    extract_lag_features,
)
from ml.metrics import mae, r2_score, rmse, smape
from ml.predictor import (
    EWMAForecaster,
    LinearARForecaster,
    NaiveLastValuePredictor,
)


class TestFeaturesAndBucketing(unittest.TestCase):
    """Test suite for trace window aggregation and feature engineering."""

    def test_bucketize_trace_conservation_and_grouping(self) -> None:
        """Workloads are assigned to correct temporal windows and total CPU/RAM is conserved."""
        workloads = [
            Workload("w1", cpu_requirement=2.0, ram_requirement=4.0, arrival_time=5.0),
            Workload("w2", cpu_requirement=4.0, ram_requirement=8.0, arrival_time=15.0),
            Workload("w3", cpu_requirement=1.0, ram_requirement=2.0, arrival_time=35.0),
            Workload("w4", cpu_requirement=8.0, ram_requirement=16.0, arrival_time=55.0),
            Workload("w5", cpu_requirement=2.0, ram_requirement=4.0, arrival_time=80.0),
        ]

        buckets = bucketize_trace(workloads, window_size=30.0)

        # min_time=5.0, max_time=80.0 -> span = 75s -> 3 windows ([5,35), [35,65), [65,95))
        self.assertEqual(len(buckets), 3)

        # Window 0: w1 (t=5), w2 (t=15) -> 2 arrivals, cpu=6.0, ram=12.0
        self.assertEqual(buckets[0].arrival_count, 2)
        self.assertAlmostEqual(buckets[0].cpu_demand, 6.0)
        self.assertAlmostEqual(buckets[0].ram_demand, 12.0)
        self.assertEqual(buckets[0].heavy_job_ratio, 0.5)  # w2 has cpu=4.0 >= 4.0

        # Window 1: w3 (t=35), w4 (t=55) -> 2 arrivals, cpu=9.0, ram=18.0
        self.assertEqual(buckets[1].arrival_count, 2)
        self.assertAlmostEqual(buckets[1].cpu_demand, 9.0)
        self.assertAlmostEqual(buckets[1].ram_demand, 18.0)
        self.assertEqual(buckets[1].heavy_job_ratio, 0.5)  # w4 has cpu=8.0 >= 4.0

        # Window 2: w5 (t=80) -> 1 arrival, cpu=2.0, ram=4.0
        self.assertEqual(buckets[2].arrival_count, 1)
        self.assertAlmostEqual(buckets[2].cpu_demand, 2.0)
        self.assertAlmostEqual(buckets[2].ram_demand, 4.0)

        # Total conservation
        total_bucket_cpu = sum(b.cpu_demand for b in buckets)
        total_workload_cpu = sum(w.cpu for w in workloads)
        self.assertAlmostEqual(total_bucket_cpu, total_workload_cpu, places=3)

        total_bucket_ram = sum(b.ram_demand for b in buckets)
        total_workload_ram = sum(w.ram for w in workloads)
        self.assertAlmostEqual(total_bucket_ram, total_workload_ram, places=3)

        total_arrivals = sum(b.arrival_count for b in buckets)
        self.assertEqual(total_arrivals, len(workloads))

    def test_bucketize_trace_invalid_window_size(self) -> None:
        """Zero, negative, boolean, and non-numeric window sizes are rejected."""
        workloads = [Workload("w1", 1, 2, arrival_time=0.0)]
        bad_sizes = [0, -1, -0.5, True, False, "30", None]
        for bad_size in bad_sizes:
            with self.subTest(bad_size=bad_size):
                with self.assertRaises(InvalidResourceError):
                    bucketize_trace(workloads, window_size=bad_size)  # type: ignore

    def test_bucketize_empty_workloads(self) -> None:
        """Bucketizing empty workload sequence returns an empty list."""
        self.assertEqual(bucketize_trace([]), [])

    def test_feature_vector_construction(self) -> None:
        """build_feature_vector extracts lag, rolling, momentum, and diurnal features."""
        buckets = [
            TimeWindowBucket(i, i * 30.0, (i + 1) * 30.0, 5 + i, 10.0 + i, 20.0 + i, 0.2)
            for i in range(5)
        ]
        features = build_feature_vector(buckets, lags=4)
        # Expected features:
        # 1 bias + 4 * 4 lags (cpu, ram, count, heavy) = 17
        # + 3 rolling means (cpu, ram, count) = 20
        # + 2 momentums (cpu, ram) = 22
        # + 2 diurnal sin/cos = 24
        self.assertEqual(len(features), 24)
        self.assertEqual(features[0], 1.0)  # Bias term

        # Synonym check
        self.assertEqual(extract_lag_features(buckets, lags=4), features)

    def test_supervised_dataset_creation(self) -> None:
        """create_supervised_dataset aligns feature matrix X with target tuple Y."""
        buckets = [
            TimeWindowBucket(i, i * 30.0, (i + 1) * 30.0, 2, 5.0, 10.0, 0.0)
            for i in range(10)
        ]
        X, Y = create_supervised_dataset(buckets, lags=3)
        self.assertEqual(len(X), 7)
        self.assertEqual(len(Y), 7)
        self.assertEqual(Y[0], (5.0, 10.0, 2.0))


class TestMetrics(unittest.TestCase):
    """Test suite for forecasting evaluation metrics."""

    def test_exact_metric_calculations(self) -> None:
        """Verify mae, rmse, smape, and r2_score against known analytical values."""
        y_true = [10.0, 20.0, 30.0, 40.0]
        y_pred = [12.0, 18.0, 33.0, 39.0]
        # Errors: +2, -2, +3, -1
        # Abs errors: 2, 2, 3, 1 -> MAE = 8/4 = 2.0
        self.assertAlmostEqual(mae(y_true, y_pred), 2.0)

        # Squared errors: 4, 4, 9, 1 -> sum=18 -> MSE = 4.5 -> RMSE = sqrt(4.5)
        self.assertAlmostEqual(rmse(y_true, y_pred), math.sqrt(4.5))

        # R^2: mean(y)=25, SS_tot = 225 + 25 + 25 + 225 = 500. SS_res = 18. R^2 = 1 - 18/500 = 0.964
        self.assertAlmostEqual(r2_score(y_true, y_pred), 0.964)

        # sMAPE: sum(|t-p| / ((|t|+|p|)/2)) / 4 * 100
        # 2/11 + 2/19 + 3/31.5 + 1/39.5
        expected_smape = (
            ((2.0 / 11.0) + (2.0 / 19.0) + (3.0 / 31.5) + (1.0 / 39.5)) / 4.0
        ) * 100.0
        self.assertAlmostEqual(smape(y_true, y_pred), expected_smape)

    def test_metrics_validation_errors(self) -> None:
        """Metrics reject empty sequences and mismatched lengths."""
        for metric_fn in [mae, rmse, smape, r2_score]:
            with self.subTest(metric=metric_fn.__name__):
                with self.assertRaises(ValueError):
                    metric_fn([], [])
                with self.assertRaises(ValueError):
                    metric_fn([1.0, 2.0], [1.0])


class TestPredictors(unittest.TestCase):
    """Test suite for Naive, EWMA, and Linear Autoregressive forecasters."""

    def test_unfitted_linear_ar_raises_error(self) -> None:
        """Calling predict_next on an unfitted LinearARForecaster raises RuntimeError."""
        model = LinearARForecaster(lags=3)
        dummy_bucket = TimeWindowBucket(0, 0.0, 30.0, 1, 2.0, 4.0, 0.0)
        with self.assertRaises(RuntimeError):
            model.predict_next([dummy_bucket] * 3)

    def test_linear_ar_exact_trend_recovery(self) -> None:
        """LinearARForecaster accurately tracks a deterministic trend out-of-sample (R^2 > 0.95)."""
        # Create a smooth deterministic increasing trend + diurnal wave
        buckets: list[TimeWindowBucket] = []
        for i in range(50):
            t = i * 30.0
            cpu = 5.0 + 0.5 * i + 2.0 * math.sin(i * 0.2)
            ram = 2.0 * cpu
            cnt = int(cpu)
            buckets.append(
                TimeWindowBucket(
                    window_index=i,
                    start_time=t,
                    end_time=t + 30.0,
                    arrival_count=cnt,
                    cpu_demand=round(cpu, 4),
                    ram_demand=round(ram, 4),
                    heavy_job_ratio=0.2,
                )
            )

        train_buckets = buckets[:35]
        test_buckets = buckets[35:]

        ar_model = LinearARForecaster(lags=4, ridge_lambda=0.01)
        ar_model.fit(train_buckets)

        naive_model = NaiveLastValuePredictor()

        y_true: list[float] = []
        y_pred_ar: list[float] = []
        y_pred_naive: list[float] = []

        all_buckets = train_buckets + test_buckets
        for idx in range(35, len(all_buckets)):
            history = all_buckets[:idx]
            target = all_buckets[idx]

            pred_ar = ar_model.predict_next(history)
            pred_naive = naive_model.predict_next(history)

            y_true.append(target.cpu_demand)
            y_pred_ar.append(pred_ar.predicted_cpu)
            y_pred_naive.append(pred_naive.predicted_cpu)

        score_ar = r2_score(y_true, y_pred_ar)
        rmse_ar = rmse(y_true, y_pred_ar)
        rmse_naive = rmse(y_true, y_pred_naive)

        self.assertGreater(score_ar, 0.95)
        self.assertLess(rmse_ar, rmse_naive)

    def test_ewma_forecaster(self) -> None:
        """EWMA forecaster smoothly follows a constant or step sequence."""
        buckets = [
            TimeWindowBucket(i, i * 30.0, (i + 1) * 30.0, 2, 10.0, 20.0, 0.0)
            for i in range(10)
        ]
        ewma = EWMAForecaster(alpha=0.5, beta=0.2)
        ewma.fit(buckets)
        forecast = ewma.predict_next(buckets)
        self.assertAlmostEqual(forecast.predicted_cpu, 10.0, places=2)
        self.assertAlmostEqual(forecast.predicted_ram, 20.0, places=2)


if __name__ == "__main__":
    unittest.main()

