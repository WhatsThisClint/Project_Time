# Architecture

Project Time separates the original interactive notebook into four layers.

## Cleaning Layer

`project_time.cleaning` reads CSV or Excel inputs, infers likely columns, and converts the dataset into a standard schema:

| Column | Meaning |
| --- | --- |
| `timestamp` | Parsed datetime. |
| `series_id` | Series identifier. Single-series files receive `series_1`. |
| `value` | Numeric target value. |

The cleaner reports invalid dates, invalid values, duplicate timestamps, inferred frequency, missing values, outlier counts, and output row counts.

## Forecasting Layer

`project_time.forecasting` works only on cleaned frames. It includes:

- `naive`
- `seasonal_naive`
- `drift`
- `moving_average`
- `ensemble`
- `ets` with optional `statsmodels`
- `arima` with optional `statsmodels`
- `timesfm` with optional TimesFM/PyTorch dependencies

The default method is `ensemble`, which is fast and dependency-light.

## TimesFM Layer

`project_time.timesfm_integration` lazy-loads TimesFM. This keeps normal cleaning and baseline forecasting fast and avoids forcing every user to install PyTorch.

When TimesFM is selected, the package calls:

```python
timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
```

The checkpoint is downloaded and cached on first use.

## Web Layer

`project_time.web.app` exposes:

- `GET /`: browser UI
- `POST /api/preview`: upload file and infer columns
- `POST /api/run`: clean and forecast
- `POST /api/download-timesfm`: pre-cache the TimesFM model
