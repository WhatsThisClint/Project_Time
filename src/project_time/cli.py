"""Command-line interface for Project Time."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .cleaning import CleanConfig, clean_timeseries, infer_columns, read_table, write_clean_outputs
from .forecasting import ForecastConfig, forecast_timeseries, write_forecast_outputs
from .timesfm_integration import TimesFMConfig, download_timesfm_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="project-time",
        description="Clean, validate, forecast, and explore time-series data.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean_parser = subparsers.add_parser("clean", help="Clean one CSV/Excel file.")
    _add_clean_args(clean_parser)
    clean_parser.add_argument("--output", type=Path, default=Path("outputs/clean"))

    forecast_parser = subparsers.add_parser("forecast", help="Clean and forecast a file.")
    _add_clean_args(forecast_parser)
    forecast_parser.add_argument("--output", type=Path, default=Path("outputs/forecast"))
    forecast_parser.add_argument("--horizon", type=int, default=12)
    forecast_parser.add_argument(
        "--method",
        default="ensemble",
        choices=[
            "naive",
            "seasonal_naive",
            "drift",
            "moving_average",
            "ets",
            "arima",
            "ensemble",
            "timesfm",
        ],
    )
    forecast_parser.add_argument("--season-length", type=int)
    forecast_parser.add_argument("--holdout", type=int, default=0)
    forecast_parser.add_argument("--timesfm-model", default=TimesFMConfig.model_name)
    forecast_parser.add_argument("--timesfm-cache-dir", type=Path)

    web_parser = subparsers.add_parser("web", help="Start the browser UI.")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8000)

    download_parser = subparsers.add_parser(
        "download-timesfm", help="Download/cache the TimesFM model."
    )
    download_parser.add_argument("--model", default=TimesFMConfig.model_name)
    download_parser.add_argument("--cache-dir", type=Path)
    download_parser.add_argument("--force", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "clean":
        data = read_table(args.input)
        config = _clean_config_from_args(args, data)
        result = clean_timeseries(data, config)
        outputs = write_clean_outputs(result, args.output)
        print(f"Cleaned data written to {outputs['csv']}")
        print(f"Report: {result.report}")
        return 0

    if args.command == "forecast":
        data = read_table(args.input)
        clean_config = _clean_config_from_args(args, data)
        clean_result = clean_timeseries(data, clean_config)
        args.output.mkdir(parents=True, exist_ok=True)
        write_clean_outputs(clean_result, args.output)
        forecast_config = ForecastConfig(
            horizon=args.horizon,
            frequency=args.frequency,
            method=args.method,
            season_length=args.season_length,
            holdout=args.holdout,
            timesfm=TimesFMConfig(
                model_name=args.timesfm_model,
                cache_dir=args.timesfm_cache_dir,
            ),
        )
        forecast_result = forecast_timeseries(clean_result.data, forecast_config)
        outputs = write_forecast_outputs(forecast_result, args.output)
        print(f"Forecast written to {outputs['forecast']}")
        if not forecast_result.metrics.empty:
            print(forecast_result.metrics.to_string(index=False))
        return 0

    if args.command == "web":
        import uvicorn

        uvicorn.run("project_time.web.app:app", host=args.host, port=args.port, reload=False)
        return 0

    if args.command == "download-timesfm":
        model = download_timesfm_model(
            TimesFMConfig(
                model_name=args.model,
                cache_dir=args.cache_dir,
                force_download=args.force,
            )
        )
        print(f"TimesFM model cached: {model}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _add_clean_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path, help="CSV or Excel file to process.")
    parser.add_argument("--date-column")
    parser.add_argument("--value-column")
    parser.add_argument("--id-column")
    parser.add_argument("--frequency")
    parser.add_argument(
        "--missing",
        default="interpolate",
        choices=["interpolate", "ffill", "bfill", "drop", "none"],
    )
    parser.add_argument(
        "--duplicates",
        default="mean",
        choices=["mean", "sum", "first", "last"],
    )
    parser.add_argument(
        "--outliers",
        default="none",
        choices=["none", "iqr", "zscore", "winsorize"],
    )


def _clean_config_from_args(args, data):
    inferred = infer_columns(data)
    date_column = args.date_column or inferred["date_column"]
    value_column = args.value_column or inferred["value_column"]
    id_column = args.id_column or inferred["id_column"]
    if not date_column or not value_column:
        raise ValueError(
            "Could not infer date/value columns. Pass --date-column and --value-column."
        )
    return CleanConfig(
        date_column=date_column,
        value_column=value_column,
        id_column=id_column,
        frequency=args.frequency,
        missing_strategy=args.missing,
        duplicate_strategy=args.duplicates,
        outlier_strategy=args.outliers,
    )
