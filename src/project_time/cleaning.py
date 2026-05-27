"""Automated cleaning utilities for time-series CSV files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd

MissingStrategy = Literal["interpolate", "ffill", "bfill", "drop", "none"]
DuplicateStrategy = Literal["mean", "sum", "first", "last"]
OutlierStrategy = Literal["none", "iqr", "zscore", "winsorize"]


@dataclass(frozen=True)
class CleanConfig:
    """Configuration for converting messy tabular data into a forecastable series."""

    date_column: str
    value_column: str
    id_column: str | None = None
    frequency: str | None = None
    duplicate_strategy: DuplicateStrategy = "mean"
    missing_strategy: MissingStrategy = "interpolate"
    outlier_strategy: OutlierStrategy = "none"
    outlier_zscore: float = 4.0
    winsorize_quantiles: tuple[float, float] = (0.01, 0.99)
    keep_columns: tuple[str, ...] = ()
    drop_empty_values: bool = False
    sort: bool = True


@dataclass
class CleanResult:
    """Cleaned data and a structured quality report."""

    data: pd.DataFrame
    report: dict[str, Any] = field(default_factory=dict)


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Excel file based on its extension."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported input file type: {suffix or '<none>'}")


def merge_tables(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Concatenate multiple CSV/Excel files after aligning columns."""

    frames = []
    for path in paths:
        frame = read_table(path)
        frame["source_file"] = Path(path).name
        frames.append(frame)
    if not frames:
        raise ValueError("At least one file is required")
    return pd.concat(frames, ignore_index=True, sort=False)


def infer_columns(data: pd.DataFrame) -> dict[str, str | None]:
    """Infer likely date, value, and ID columns from names and dtypes."""

    columns = list(data.columns)
    lower = {column: str(column).casefold() for column in columns}

    date_column = _first_match(
        lower,
        ["date", "time", "timestamp", "datetime", "observed", "measurement_date"],
    )
    if date_column is None:
        parsed_scores = {
            column: _parse_datetime(data[column]).notna().mean() for column in columns
        }
        date_column = max(parsed_scores, key=parsed_scores.get) if parsed_scores else None
        if parsed_scores and parsed_scores[date_column] < 0.5:
            date_column = None

    value_column = _first_match(
        lower,
        ["value", "level", "reading", "measurement", "target", "y", "water"],
    )
    if value_column is None:
        numeric_candidates = [
            column
            for column in columns
            if column != date_column
            and pd.to_numeric(data[column], errors="coerce").notna().mean() >= 0.5
        ]
        value_column = numeric_candidates[0] if numeric_candidates else None

    id_column = _first_match(
        lower,
        ["site", "well", "station", "sensor", "id", "series", "location"],
    )
    if id_column in {date_column, value_column}:
        id_column = None

    return {"date_column": date_column, "value_column": value_column, "id_column": id_column}


def clean_timeseries(data: pd.DataFrame, config: CleanConfig) -> CleanResult:
    """Clean and regularize one or more time series."""

    _validate_columns(data, config)
    original = data.copy()
    work = data.copy()

    selected_columns = [config.date_column, config.value_column]
    if config.id_column:
        selected_columns.append(config.id_column)
    selected_columns.extend(column for column in config.keep_columns if column in work.columns)
    work = work.loc[:, list(dict.fromkeys(selected_columns))].copy()

    work = work.rename(
        columns={
            config.date_column: "timestamp",
            config.value_column: "value",
            **({config.id_column: "series_id"} if config.id_column else {}),
        }
    )
    if "series_id" not in work.columns:
        work["series_id"] = "series_1"

    work["timestamp"] = _parse_datetime(work["timestamp"])
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work["series_id"] = work["series_id"].fillna("missing_id").astype(str)

    invalid_dates = int(work["timestamp"].isna().sum())
    invalid_values = int(work["value"].isna().sum())
    work = work.dropna(subset=["timestamp"])
    if config.drop_empty_values:
        work = work.dropna(subset=["value"])

    if config.sort:
        work = work.sort_values(["series_id", "timestamp"]).reset_index(drop=True)

    duplicate_count = int(work.duplicated(["series_id", "timestamp"]).sum())
    work = _aggregate_duplicates(work, config.duplicate_strategy)
    outlier_report = _handle_outliers(work, config)

    inferred_frequency = infer_frequency(work)
    frequency = config.frequency or inferred_frequency
    if frequency:
        work = _regularize_frequency(work, frequency)

    missing_before_fill = int(work["value"].isna().sum())
    work = _fill_missing(work, config.missing_strategy)
    missing_after_fill = int(work["value"].isna().sum())

    if config.sort:
        work = work.sort_values(["series_id", "timestamp"]).reset_index(drop=True)

    report = {
        "input_rows": int(len(original)),
        "output_rows": int(len(work)),
        "series_count": int(work["series_id"].nunique()),
        "invalid_dates_removed": invalid_dates,
        "invalid_numeric_values": invalid_values,
        "duplicates_collapsed": duplicate_count,
        "missing_before_fill": missing_before_fill,
        "missing_after_fill": missing_after_fill,
        "frequency": frequency,
        "inferred_frequency": inferred_frequency,
        "start": _iso_or_none(work["timestamp"].min()),
        "end": _iso_or_none(work["timestamp"].max()),
        "outliers": outlier_report,
        "columns": list(work.columns),
    }
    return CleanResult(data=work, report=report)


def write_clean_outputs(result: CleanResult, output_dir: str | Path) -> dict[str, Path]:
    """Write cleaned CSV, Excel, and report files."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "cleaned_timeseries.csv"
    xlsx_path = out / "cleaned_timeseries.xlsx"
    report_path = out / "cleaning_report.json"

    result.data.to_csv(csv_path, index=False)
    result.data.to_excel(xlsx_path, index=False)
    pd.Series(result.report, dtype="object").to_json(report_path, indent=2)
    return {"csv": csv_path, "excel": xlsx_path, "report": report_path}


def infer_frequency(data: pd.DataFrame) -> str | None:
    """Infer the most common time delta as a pandas frequency string."""

    if data.empty:
        return None

    inferred = []
    deltas = []
    for _series_id, group in data.groupby("series_id"):
        timestamps = group["timestamp"].dropna().sort_values().drop_duplicates()
        if len(timestamps) > 2:
            freq = pd.infer_freq(timestamps)
            if freq:
                inferred.append(freq)
                continue
        if len(timestamps) > 2:
            delta = timestamps.diff().dropna().mode()
            if not delta.empty:
                deltas.append(delta.iloc[0])
    if inferred:
        return str(pd.Series(inferred).mode().iloc[0])
    if not deltas:
        return None

    most_common = pd.Series(deltas).mode().iloc[0]
    try:
        return pd.tseries.frequencies.to_offset(most_common).freqstr
    except ValueError:
        return None


def _validate_columns(data: pd.DataFrame, config: CleanConfig) -> None:
    required = [config.date_column, config.value_column]
    if config.id_column:
        required.append(config.id_column)
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _parse_datetime(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", format="mixed")
    if parsed.notna().mean() >= 0.5:
        return parsed
    dayfirst = pd.to_datetime(values, errors="coerce", dayfirst=True, format="mixed")
    return dayfirst if dayfirst.notna().mean() > parsed.notna().mean() else parsed


def _first_match(lower_columns: dict[str, str], hints: list[str]) -> str | None:
    for hint in hints:
        for column, lowered in lower_columns.items():
            if hint == lowered or hint in lowered:
                return column
    return None


def _aggregate_duplicates(data: pd.DataFrame, strategy: DuplicateStrategy) -> pd.DataFrame:
    aggregations = {"mean": "mean", "sum": "sum", "first": "first", "last": "last"}
    value_agg = aggregations[strategy]
    other_columns = [
        column
        for column in data.columns
        if column not in {"series_id", "timestamp", "value"}
    ]
    agg_spec: dict[str, Any] = {"value": value_agg}
    agg_spec.update({column: "first" for column in other_columns})
    return (
        data.groupby(["series_id", "timestamp"], as_index=False)
        .agg(agg_spec)
        .sort_values(["series_id", "timestamp"])
        .reset_index(drop=True)
    )


def _handle_outliers(data: pd.DataFrame, config: CleanConfig) -> dict[str, Any]:
    if config.outlier_strategy == "none" or data.empty:
        return {"method": "none", "count": 0}

    total = 0
    for _series_id, idx in data.groupby("series_id").groups.items():
        values = data.loc[idx, "value"]
        if values.dropna().empty:
            continue

        if config.outlier_strategy == "iqr":
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
        elif config.outlier_strategy == "zscore":
            mean = values.mean()
            std = values.std(ddof=0)
            if not std or np.isnan(std):
                continue
            lower = mean - config.outlier_zscore * std
            upper = mean + config.outlier_zscore * std
        elif config.outlier_strategy == "winsorize":
            lower, upper = values.quantile(config.winsorize_quantiles)
        else:
            raise ValueError(f"Unsupported outlier strategy: {config.outlier_strategy}")

        mask = (values < lower) | (values > upper)
        total += int(mask.sum())
        data.loc[idx, "value"] = values.clip(lower=lower, upper=upper)

    return {"method": config.outlier_strategy, "count": total}


def _regularize_frequency(data: pd.DataFrame, frequency: str) -> pd.DataFrame:
    frames = []
    for series_id, group in data.groupby("series_id", sort=False):
        group = group.sort_values("timestamp").set_index("timestamp")
        if group.empty:
            continue
        full_index = pd.date_range(group.index.min(), group.index.max(), freq=frequency)
        regular = group.reindex(full_index)
        regular.index.name = "timestamp"
        regular["series_id"] = series_id
        frames.append(regular.reset_index())
    if not frames:
        return data
    return pd.concat(frames, ignore_index=True, sort=False)


def _fill_missing(data: pd.DataFrame, strategy: MissingStrategy) -> pd.DataFrame:
    if strategy == "none":
        return data
    if strategy == "drop":
        return data.dropna(subset=["value"]).reset_index(drop=True)

    frames = []
    for _series_id, group in data.groupby("series_id", sort=False):
        group = group.sort_values("timestamp").copy()
        if strategy == "interpolate":
            group["value"] = group["value"].interpolate(limit_direction="both")
        elif strategy == "ffill":
            group["value"] = group["value"].ffill().bfill()
        elif strategy == "bfill":
            group["value"] = group["value"].bfill().ffill()
        else:
            raise ValueError(f"Unsupported missing strategy: {strategy}")
        frames.append(group)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else data


def _iso_or_none(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()
