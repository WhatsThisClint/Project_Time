# Project Time

Project Time is a small time-series workbench for messy CSV and Excel files. It automates the pre-processing steps from the original notebook, adds reproducible command-line workflows, supports multiple forecasting methods, and includes a browser UI.

The original `Pre_processing.ipynb` is kept as historical provenance. New reusable code lives in `src/project_time/`.

## What It Does

- Reads CSV and Excel files.
- Infers likely date, value, and series ID columns.
- Cleans invalid dates, non-numeric target values, duplicate timestamps, missing periods, missing values, and optional outliers.
- Supports single-series and multi-series forecasting.
- Includes baseline methods: naive, seasonal naive, drift, moving average, and ensemble.
- Adds optional classical methods: ETS and ARIMA through `statsmodels`.
- Adds optional TimesFM forecasting with first-use model download and local cache.
- Provides a FastAPI web UI for upload, cleaning, forecasting, metrics, and charts.

## Install

Core cleaning and baseline forecasting:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Web UI:

```powershell
python -m pip install -e .[web]
project-time web
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

TimesFM support:

```powershell
python -m pip install -e .[timesfm]
project-time download-timesfm
```

TimesFM downloads `google/timesfm-2.5-200m-pytorch` from Hugging Face on first use. The model is stored in the normal Hugging Face cache unless you pass `--cache-dir`.

Everything:

```powershell
python -m pip install -e .[all]
```

## CLI Examples

Clean a file with automatic column inference:

```powershell
project-time clean examples\sample_groundwater.csv --output outputs\clean
```

Forecast 12 future steps with the default ensemble:

```powershell
project-time forecast examples\sample_groundwater.csv --horizon 12 --holdout 4 --output outputs\forecast
```

Forecast with explicit columns:

```powershell
project-time forecast data.csv --date-column date --value-column level --id-column site --frequency MS --method ensemble
```

Forecast with TimesFM:

```powershell
project-time forecast data.csv --date-column date --value-column level --id-column site --method timesfm --horizon 24
```

## Tests

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests
```

## Repository Layout

- `Pre_processing.ipynb`: original interactive notebook.
- `src/project_time/cleaning.py`: automated cleaning and validation.
- `src/project_time/forecasting.py`: baseline, classical, ensemble, and TimesFM forecast routing.
- `src/project_time/timesfm_integration.py`: lazy TimesFM model loading and cache/download helper.
- `src/project_time/web/`: FastAPI app and browser UI.
- `tests/`: lightweight regression tests.
- `examples/`: small sample data for smoke tests.
