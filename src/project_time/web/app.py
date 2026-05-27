"""FastAPI web application for Project Time."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from project_time.cleaning import CleanConfig, clean_timeseries, infer_columns
from project_time.forecasting import ForecastConfig, forecast_timeseries
from project_time.timesfm_integration import (
    TimesFMConfig,
    TimesFMUnavailable,
    download_timesfm_model,
)

STATIC_DIR = Path(__file__).with_name("static")

app = FastAPI(
    title="Project Time",
    description="Automated time-series cleaning, validation, and forecasting.",
    version="0.2.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/preview")
async def preview(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await _read_upload(file)
    inferred = infer_columns(data)
    return _json_ready(
        {
            "columns": list(data.columns),
            "inferred": inferred,
            "rows": len(data),
            "preview": data.head(30).to_dict(orient="records"),
        }
    )


@app.post("/api/run")
async def run_analysis(
    file: UploadFile = File(...),
    date_column: str = Form(...),
    value_column: str = Form(...),
    id_column: str = Form(""),
    frequency: str = Form(""),
    missing: str = Form("interpolate"),
    duplicates: str = Form("mean"),
    outliers: str = Form("none"),
    method: str = Form("ensemble"),
    horizon: int = Form(12),
    season_length: int | None = Form(None),
    holdout: int = Form(0),
) -> dict[str, Any]:
    data = await _read_upload(file)
    try:
        clean_result = clean_timeseries(
            data,
            CleanConfig(
                date_column=date_column,
                value_column=value_column,
                id_column=id_column or None,
                frequency=frequency or None,
                missing_strategy=missing,  # type: ignore[arg-type]
                duplicate_strategy=duplicates,  # type: ignore[arg-type]
                outlier_strategy=outliers,  # type: ignore[arg-type]
            ),
        )
        forecast_result = forecast_timeseries(
            clean_result.data,
            ForecastConfig(
                horizon=horizon,
                frequency=clean_result.report.get("frequency") or frequency or None,
                method=method,  # type: ignore[arg-type]
                season_length=season_length,
                holdout=holdout,
            ),
        )
    except TimesFMUnavailable as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _json_ready(
        {
            "report": clean_result.report,
            "cleaned_preview": clean_result.data.head(200).to_dict(orient="records"),
            "forecasts": forecast_result.forecasts.to_dict(orient="records"),
            "metrics": forecast_result.metrics.to_dict(orient="records"),
            "diagnostics": forecast_result.diagnostics,
        }
    )


@app.post("/api/download-timesfm")
async def download_model(
    model: str = Form(TimesFMConfig.model_name),
    cache_dir: str = Form(""),
    force: bool = Form(False),
) -> dict[str, str]:
    try:
        cached = download_timesfm_model(
            TimesFMConfig(
                model_name=model,
                cache_dir=cache_dir or None,
                force_download=force,
            )
        )
    except TimesFMUnavailable as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"model": cached, "status": "cached"}


async def _read_upload(file: UploadFile) -> pd.DataFrame:
    suffix = Path(file.filename or "").suffix.lower()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    buffer = BytesIO(content)
    try:
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(buffer)
        return pd.read_csv(buffer)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read file: {exc}") from exc


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value
