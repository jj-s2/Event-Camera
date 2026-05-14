from __future__ import annotations

import numpy as np


def events_to_voxel_grid(
    events: np.ndarray,
    height: int,
    width: int,
    bins: int = 5,
    normalize: bool = True,
) -> np.ndarray:
    """Convert ``(x, y, t, p)`` events to a polarity-split voxel grid."""
    if bins <= 0:
        raise ValueError("bins must be positive")
    voxel = np.zeros((bins * 2, height, width), dtype=np.float32)
    events = _valid_events(events, height, width)
    if len(events) == 0:
        return voxel

    x = events[:, 0].astype(np.int64)
    y = events[:, 1].astype(np.int64)
    t_norm = _normalize_time(events[:, 2])
    bin_idx = np.clip((t_norm * bins).astype(np.int64), 0, bins - 1)
    polarity_offset = np.where(events[:, 3] >= 0, 0, bins)
    channels = polarity_offset + bin_idx

    np.add.at(voxel, (channels, y, x), 1.0)
    if normalize:
        max_per_channel = voxel.reshape(voxel.shape[0], -1).max(axis=1)
        for channel, max_value in enumerate(max_per_channel):
            if max_value > 0:
                voxel[channel] /= max_value
    return np.clip(voxel, 0.0, 1.0)


def events_to_time_surface(events: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return a one-channel surface holding latest normalized event time."""
    surface = np.zeros((1, height, width), dtype=np.float32)
    events = _valid_events(events, height, width)
    if len(events) == 0:
        return surface

    x = events[:, 0].astype(np.int64)
    y = events[:, 1].astype(np.int64)
    t_norm = _normalize_time(events[:, 2]).astype(np.float32)
    np.maximum.at(surface[0], (y, x), t_norm)
    return surface


def build_event_tensor(
    events: np.ndarray,
    height: int,
    width: int,
    bins: int = 5,
    include_time_surface: bool = True,
) -> np.ndarray:
    """Build the default detector input tensor."""
    voxel = events_to_voxel_grid(events, height=height, width=width, bins=bins)
    if not include_time_surface:
        return voxel
    time_surface = events_to_time_surface(events, height=height, width=width)
    return np.concatenate([voxel, time_surface], axis=0)


def _valid_events(events: np.ndarray, height: int, width: int) -> np.ndarray:
    arr = np.asarray(events, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("events must have shape (N, 4) in x/y/t/p order")
    keep = (
        np.isfinite(arr).all(axis=1)
        & (arr[:, 0] >= 0)
        & (arr[:, 0] < width)
        & (arr[:, 1] >= 0)
        & (arr[:, 1] < height)
    )
    return arr[keep]


def _normalize_time(timestamps: np.ndarray) -> np.ndarray:
    t = np.asarray(timestamps, dtype=np.float32)
    t_min = float(t.min())
    t_max = float(t.max())
    if t_max <= t_min:
        return np.zeros_like(t, dtype=np.float32)
    return (t - t_min) / (t_max - t_min)
