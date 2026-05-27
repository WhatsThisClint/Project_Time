"""Forecasting methods for cleaned time-series data."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .metrics import metric_summary
from .timesfm_integration import TimesFMConfig, forecast_with_timesfm

ForecastMethod = Literal[
    "naive",
    "seasonal_naive",
    "drift",
    "moving_average",
    "ets",
    "arima",
    "ensemble",
    "timesfm",
]


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for forecast generation."""

    horizon: int = 12
    frequency: str | None = None
    method: ForecastMethod = "ensemble"
    season_length: int | None = None
    moving_average_window: int = 6
    holdout: int = 0
    timesfm: TimesFMConfig = field(default_factory=TimesFMConfig)

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError("horizon must be greater than zero")
        if self.holdout < 0:
            raise ValueError("holdout cannot be negative")


@dataclass
class ForecastResult:
    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)


def forecast_timeseries(data: pd.DataFrame, config: ForecastConfig) -> ForecastResult:
    """Forecast each series in a cleaned dataframe."""

    _validate_clean_frame(data)
    frequency = config.frequency or _safe_infer_frequency(data) or "D"
    rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"frequency": frequency, "method": config.method}

    for series_id, group in data.groupby("series_id", sort=True):
        group = group.sort_values("timestamp").dropna(subset=["value"])
        if len(group) < 2:
            diagnostics[str(series_id)] = "skipped: fewer than two observations"
            continue

        train = group
        test = pd.DataFrame()
        if config.holdout:
            if len(group) <= config.holdout + 1:
                diagnostics[str(series_id)] = "holdout skipped: insufficient history"
            else:
                train = group.iloc[:-config.holdout]
                test = group.iloc[-config.holdout:]

        forecast_values, lower, upper, method_used = _forecast_one(group["value"].to_numpy(), config)
        future_index = _future_index(group["timestamp"].iloc[-1], frequency, config.horizon)
        rows.append(
            pd.DataFrame(
                {
                    "series_id": str(series_id),
                    "timestamp": future_index,
                    "forecast": forecast_values,
                    "lower_80": lower,
                    "upper_80": upper,
                    "method": method_used,
                }
            )
        )

        if not test.empty:
            holdout_horizon = len(test)
            holdout_cfg = ForecastConfig(
                horizon=holdout_horizon,
                frequency=frequency,
                method=config.method,
                season_length=config.season_length,
                moving_average_window=config.moving_average_window,
                holdout=0,
                timesfm=config.timesfm,
            )
            predicted, _lower, _upper, holdout_method = _forecast_one(
                train["value"].to_numpy(), holdout_cfg
            )
            metric_rows.append(
                {
                    "series_id": str(series_id),
                    "method": holdout_method,
                    **metric_summary(test["value"].to_numpy(), predicted),
                }
            )

    forecasts = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    return ForecastResult(forecasts=forecasts, metrics=metrics, diagnostics=diagnostics)


def write_forecast_outputs(result: ForecastResult, output_dir: str | Path) -> dict[str, Path]:
    """Write forecast and metric files."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    forecast_path = out / "forecast.csv"
    metrics_path = out / "metrics.csv"
    result.forecasts.to_csv(forecast_path, index=False)
    result.metrics.to_csv(metrics_path, index=False)
    return {"forecast": forecast_path, "metrics": metrics_path}


def _forecast_one(values: np.ndarray, config: ForecastConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("At least two finite observations are required")

    method = config.method
    if method == "ensemble":
        return _ensemble_forecast(values, config)
    if method == "timesfm":
        point, quantiles = forecast_with_timesfm(
            [values.astype(np.float32)], config.horizon, config.timesfm
        )
        lower, upper = _timesfm_interval(point[0], quantiles[0])
        return point[0].astype(float), lower, upper, "timesfm"
    if method == "ets":
        return _statsmodels_ets(values, config)
    if method == "arima":
        return _statsmodels_arima(values, config)

    point = _baseline_forecast(values, config, method)
    residual_std = _residual_std(values, config, method)
    lower = point - 1.28 * residual_std
    upper = point + 1.28 * residual_std
    return point, lower, upper, method


def _baseline_forecast(values: np.ndarray, config: ForecastConfig, method: str) -> np.ndarray:
    horizon = config.horizon
    if method == "naive":
        return np.repeat(values[-1], horizon)
    if method == "seasonal_naive":
        season_length = _season_length(config, len(values))
        pattern = values[-season_length:]
        return np.resize(pattern, horizon)
    if method == "drift":
        slope = (values[-1] - values[0]) / max(len(values) - 1, 1)
        return values[-1] + slope * np.arange(1, horizon + 1)
    if method == "moving_average":
        window = min(config.moving_average_window, len(values))
        return np.repeat(np.mean(values[-window:]), horizon)
    raise ValueError(f"Unknown forecast method: {method}")


def _ensemble_forecast(values: np.ndarray, config: ForecastConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    methods = ["naive", "seasonal_naive", "drift", "moving_average"]
    predictions = []
    for method in methods:
        predictions.append(_baseline_forecast(values, config, method))
    stacked = np.vstack(predictions)
    point = np.median(stacked, axis=0)
    lower = np.quantile(stacked, 0.1, axis=0)
    upper = np.quantile(stacked, 0.9, axis=0)
    return point, lower, upper, "ensemble"


def _statsmodels_ets(values: np.ndarray, config: ForecastConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError as exc:
        raise RuntimeError(
            "ETS forecasting requires statsmodels. Install with "
            "`python -m pip install -e .[classical]` or `.[all]`."
        ) from exc

    season_length = _season_length(config, len(values))
    seasonal = "add" if len(values) >= season_length * 2 else None
    model = ExponentialSmoothing(
        values,
        trend="add",
        seasonal=seasonal,
        seasonal_periods=season_length if seasonal else None,
        initialization_method="estimated",
    )
    fitted = model.fit(optimized=True)
    point = np.asarray(fitted.forecast(config.horizon), dtype=float)
    resid_std = float(np.nanstd(fitted.resid)) or _residual_std(values, config, "drift")
    return point, point - 1.28 * resid_std, point + 1.28 * resid_std, "ets"


def _statsmodels_arima(values: np.ndarray, config: ForecastConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise RuntimeError(
            "ARIMA forecasting requires statsmodels. Install with "
            "`python -m pip install -e .[classical]` or `.[all]`."
        ) from exc

    model = SARIMAX(
        values,
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.get_forecast(steps=config.horizon)
    point = np.asarray(forecast.predicted_mean, dtype=float)
    interval = np.asarray(forecast.conf_int(alpha=0.2), dtype=float)
    return point, interval[:, 0], interval[:, 1], "arima"


def _timesfm_interval(point: np.ndarray, quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if quantiles.ndim == 2 and quantiles.shape[1] >= 10:
        return quantiles[:, 1].astype(float), quantiles[:, 9].astype(float)
    spread = np.nanstd(point) or 1.0
    return point - spread, point + spread


def _residual_std(values: np.ndarray, config: ForecastConfig, method: str) -> float:
    if len(values) < 4:
        return float(np.nanstd(values)) or 1.0
    preds = []
    actuals = []
    for i in range(2, len(values)):
        hist = values[:i]
        pred = _baseline_forecast(hist, config, method)[0]
        preds.append(pred)
        actuals.append(values[i])
    resid = np.asarray(actuals) - np.asarray(preds)
    return float(np.nanstd(resid)) or 1.0


def _season_length(config: ForecastConfig, n_obs: int) -> int:
    if config.season_length and config.season_length > 1:
        return min(config.season_length, n_obs)
    return min(12, max(2, n_obs // 2))


def _future_index(last_timestamp, frequency: str, horizon: int) -> pd.DatetimeIndex:
    offset = pd.tseries.frequencies.to_offset(frequency)
    start = pd.Timestamp(last_timestamp) + offset
    return pd.date_range(start=start, periods=horizon, freq=offset)


def _safe_infer_frequency(data: pd.DataFrame) -> str | None:
    from .cleaning import infer_frequency

    return infer_frequency(data)


def _validate_clean_frame(data: pd.DataFrame) -> None:
    required = {"timestamp", "series_id", "value"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Data is missing cleaned columns: {', '.join(sorted(missing))}")
