from __future__ import annotations

import unittest

import pandas as pd

from project_time.forecasting import ForecastConfig, forecast_timeseries


class ForecastingTests(unittest.TestCase):
    def sample(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
                "series_id": ["A"] * 10,
                "value": list(range(10)),
            }
        )

    def test_drift_forecast(self) -> None:
        result = forecast_timeseries(
            self.sample(),
            ForecastConfig(horizon=3, frequency="D", method="drift", holdout=2),
        )
        self.assertEqual(len(result.forecasts), 3)
        self.assertEqual(result.forecasts.loc[0, "series_id"], "A")
        self.assertFalse(result.metrics.empty)

    def test_ensemble_forecast_has_interval(self) -> None:
        result = forecast_timeseries(
            self.sample(),
            ForecastConfig(horizon=4, frequency="D", method="ensemble"),
        )
        self.assertEqual(len(result.forecasts), 4)
        self.assertTrue((result.forecasts["upper_80"] >= result.forecasts["lower_80"]).all())


if __name__ == "__main__":
    unittest.main()
