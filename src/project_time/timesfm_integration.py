"""Lazy TimesFM integration.

TimesFM and its PyTorch stack are intentionally optional. The first TimesFM run
downloads the model from Hugging Face into the normal cache, or into
``cache_dir`` when supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_TIMESFM_MODEL = "google/timesfm-2.5-200m-pytorch"


@dataclass(frozen=True)
class TimesFMConfig:
    model_name: str = DEFAULT_TIMESFM_MODEL
    cache_dir: str | Path | None = None
    max_context: int = 1024
    max_horizon: int = 256
    batch_size: int = 32
    force_download: bool = False


class TimesFMUnavailable(RuntimeError):
    """Raised when TimesFM is requested but optional dependencies are missing."""


def forecast_with_timesfm(
    series: Iterable[np.ndarray],
    horizon: int,
    config: TimesFMConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast one or more arrays with TimesFM."""

    cfg = config or TimesFMConfig()
    model = _load_model(cfg)
    inputs = [_clean_array(values) for values in series]
    if not inputs:
        raise ValueError("TimesFM requires at least one input series")
    return model.forecast(horizon=horizon, inputs=inputs)


def download_timesfm_model(config: TimesFMConfig | None = None) -> str:
    """Download/cache the TimesFM model and return the model name."""

    cfg = config or TimesFMConfig(force_download=False)
    _load_model(cfg)
    return cfg.model_name


def _load_model(config: TimesFMConfig):
    try:
        import torch
        import timesfm
    except ImportError as exc:
        raise TimesFMUnavailable(
            "TimesFM is optional. Install it with "
            "`python -m pip install -e .[timesfm]` or "
            "`python -m pip install -e .[all]`."
        ) from exc

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        config.model_name,
        cache_dir=str(config.cache_dir) if config.cache_dir else None,
        force_download=config.force_download,
    )
    model.compile(
        timesfm.ForecastConfig(
            max_context=config.max_context,
            max_horizon=config.max_horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
            per_core_batch_size=config.batch_size,
        )
    )
    return model


def _clean_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    while len(arr) > 0 and np.isnan(arr[-1]):
        arr = arr[:-1]
    if len(arr) < 2:
        raise ValueError("Each TimesFM input series needs at least two finite values")
    return arr
