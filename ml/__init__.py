"""CloudPilot Machine Learning Package for Workload Demand Forecasting."""

from ml.features import (
    TimeWindowBucket,
    bucketize_trace,
    build_feature_vector,
    create_supervised_dataset,
    extract_lag_features,
)
from ml.metrics import mae, r2_score, rmse, smape
from ml.predictor import (
    DemandForecast,
    DemandPredictor,
    EWMAForecaster,
    LinearARForecaster,
    NaiveLastValuePredictor,
)

__all__ = [
    "TimeWindowBucket",
    "bucketize_trace",
    "build_feature_vector",
    "extract_lag_features",
    "create_supervised_dataset",
    "mae",
    "rmse",
    "smape",
    "r2_score",
    "DemandForecast",
    "DemandPredictor",
    "NaiveLastValuePredictor",
    "EWMAForecaster",
    "LinearARForecaster",
]

