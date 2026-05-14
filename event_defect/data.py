from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np


_COLUMN_ALIASES = {
    "x": ("x", "xs", "coord_x"),
    "y": ("y", "ys", "coord_y"),
    "t": ("t", "ts", "time", "timestamp", "timestamps"),
    "p": ("p", "pol", "polarity", "polarities", "event_g", "g", "gray", "value"),
}


def load_event_h5(path: str | Path, column_order: str = "xytp") -> np.ndarray:
    """Load an event-camera h5 file into an ``(N, 4)`` x/y/t/p array.

    Supported layouts:
    - a single ``events`` dataset with four columns
    - separate datasets named like ``x``, ``y``, ``t`` and ``p``
    - nested groups that contain either of those layouts
    """
    path = Path(path)
    with h5py.File(path, "r") as h5:
        events = _find_event_matrix(h5)
        if events is not None:
            return _matrix_to_xytp(events, column_order)

        columns = _find_event_columns(h5)
        if columns is None:
            raise ValueError(f"No event stream found in {path}")

        x = np.asarray(columns["x"])
        y = np.asarray(columns["y"])
        t = np.asarray(columns["t"])
        p = np.asarray(columns["p"])
        if not (len(x) == len(y) == len(t) == len(p)):
            raise ValueError(f"Column lengths differ in {path}")
        return np.stack([x, y, t, p], axis=1).astype(np.float32, copy=False)


def _find_event_matrix(group: h5py.Group) -> np.ndarray | None:
    preferred = ("events", "event", "Event", "event_stream")
    for name in preferred:
        if name in group and isinstance(group[name], h5py.Dataset):
            arr = np.asarray(group[name])
            if arr.ndim == 2 and 4 in arr.shape:
                return arr

    for _, value in group.items():
        if isinstance(value, h5py.Group):
            found = _find_event_matrix(value)
            if found is not None:
                return found
        elif isinstance(value, h5py.Dataset):
            arr = np.asarray(value)
            if arr.ndim == 2 and arr.shape[1] == 4:
                return arr
    return None


def _find_event_columns(group: h5py.Group) -> dict[str, np.ndarray] | None:
    columns: dict[str, np.ndarray] = {}
    flat = _flatten_datasets(group)

    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            match = _lookup_dataset(flat, alias)
            if match is not None:
                columns[canonical] = np.asarray(match)
                break

    if set(columns) == {"x", "y", "t", "p"}:
        return columns
    return None


def _flatten_datasets(group: h5py.Group) -> dict[str, h5py.Dataset]:
    flat: dict[str, h5py.Dataset] = {}

    def visit(name: str, obj: h5py.Dataset) -> None:
        if isinstance(obj, h5py.Dataset):
            flat[name.lower()] = obj
            flat[Path(name).name.lower()] = obj

    group.visititems(visit)
    return flat


def _lookup_dataset(flat: dict[str, h5py.Dataset], alias: str) -> h5py.Dataset | None:
    alias = alias.lower()
    if alias in flat:
        return flat[alias]
    suffix = "/" + alias
    for name, dataset in flat.items():
        if name.endswith(suffix):
            return dataset
    return None


def _matrix_to_xytp(matrix: np.ndarray, column_order: str) -> np.ndarray:
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError("Event matrix must be two-dimensional")
    if arr.shape[0] == 4 and arr.shape[1] != 4:
        arr = arr.T
    if arr.shape[1] != 4:
        raise ValueError("Event matrix must have four columns")

    order = column_order.lower()
    if sorted(order) != ["p", "t", "x", "y"]:
        raise ValueError("column_order must contain x, y, t and p exactly once")

    mapping = {name: order.index(name) for name in "xytp"}
    out = np.stack(
        [arr[:, mapping["x"]], arr[:, mapping["y"]], arr[:, mapping["t"]], arr[:, mapping["p"]]],
        axis=1,
    )
    return out.astype(np.float32, copy=False)
