"""Dependency-light helpers shared by UMI-FT inference and evaluation."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def derive_noise_seed(window_id: str, sampling_seed: int) -> int:
    digest = hashlib.sha256(f"{window_id}\0{sampling_seed}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF


def persistence_prediction(truth: Any) -> np.ndarray:
    value = np.asarray(truth)
    if value.ndim != 4 or value.shape[0] != 17 or value.shape[-1] not in (1, 3):
        raise ValueError(f"E1 persistence requires THWC with exactly 17 frames, got {value.shape}")
    return np.repeat(value[:1], 17, axis=0)
