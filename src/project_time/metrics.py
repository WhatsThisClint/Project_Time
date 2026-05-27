"""Forecast accuracy metrics."""

from __future__ import annotations

import numpy as np


def mae(actual, predicted) -> float:
    actual_arr, predicted_arr = _paired(actual, predicted)
    return float(np.mean(np.abs(actual_arr - predicted_arr)))


def rmse(actual, predicted) -> float:
    actual_arr, predicted_arr = _paired(actual, predicted)
    return float(np.sqrt(np.mean((actual_arr - predicted_arr) ** 2)))


def mape(actual, predicted) -> float:
    actual_arr, predicted_arr = _paired(actual, predicted)
    mask = actual_arr != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual_arr[mask] - predicted_arr[mask]) / actual_arr[mask])) * 100)


def smape(actual, predicted) -> float:
    actual_arr, predicted_arr = _paired(actual, predicted)
    denominator = np.abs(actual_arr) + np.abs(predicted_arr)
    mask = denominator != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(2 * np.abs(predicted_arr[mask] - actual_arr[mask]) / denominator[mask]) * 100)


def metric_summary(actual, predicted) -> dict[str, float]:
    return {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted),
        "smape": smape(actual, predicted),
    }


def _paired(actual, predicted) -> tuple[np.ndarray, np.ndarray]:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    size = min(len(actual_arr), len(predicted_arr))
    if size == 0:
        raise ValueError("Metric inputs must contain at least one value")
    actual_arr = actual_arr[:size]
    predicted_arr = predicted_arr[:size]
    mask = np.isfinite(actual_arr) & np.isfinite(predicted_arr)
    if not mask.any():
        raise ValueError("Metric inputs have no finite paired values")
    return actual_arr[mask], predicted_arr[mask]
