from __future__ import annotations

from typing import List, Tuple

import numpy as np


def normalize_series_to_polyline(
    values: np.ndarray, width: int, height: int, *, pad: int = 2
) -> List[Tuple[int, int]]:
    """Map 1D values into integer pixel polyline points for sparkline.

    - Pads drawing area by `pad` pixels on each side
    - Constant series → flat line centered vertically
    - Empty or invalid input → empty list
    """
    try:
        arr = np.asarray(values, dtype=float).ravel()
    except Exception:
        return []
    n = int(arr.size)
    if n <= 0 or width <= 2 * pad or height <= 2 * pad:
        return []

    x0, x1 = pad, max(pad + 1, width - pad - 1)
    y0, y1 = pad, max(pad + 1, height - pad - 1)

    # X positions spread evenly across available width
    if n == 1:
        xs = np.array([(x0 + x1) // 2], dtype=int)
    else:
        xs = np.linspace(x0, x1, n).astype(int)

    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return []

    if abs(vmax - vmin) < 1e-12:
        # Flat line in middle
        y_mid = int((y0 + y1) / 2)
        ys = np.full(n, y_mid, dtype=int)
    else:
        # Normalize to [0, 1]
        norm = (arr - vmin) / (vmax - vmin)
        # Flip Y (0 at bottom visually → top pixel)
        ys_float = y1 - norm * (y1 - y0)
        ys = np.clip(ys_float, y0, y1).astype(int)

    return list(zip(xs.tolist(), ys.tolist()))
