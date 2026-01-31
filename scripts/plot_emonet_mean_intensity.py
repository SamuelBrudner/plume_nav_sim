from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Emonet background-subtracted mean intensity per frame from a CSV. "
            "Writes a full-range plot and an early-time zoom plot."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/tmp/emonet_mean_intensity.csv"),
        help="CSV path (default: /tmp/emonet_mean_intensity.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
        help="Output directory for PNGs (default: plots/)",
    )
    parser.add_argument(
        "--baseline-n",
        type=int,
        default=5,
        help="Number of baseline frames (default: 5)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        help="Threshold in baseline standard deviations (default: 5.0)",
    )
    parser.add_argument(
        "--zoom-max-frame",
        type=int,
        default=2000,
        help="Max frame shown in zoom plot (default: 2000)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG DPI (default: 200)",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)

    frames: list[int] = []
    means: list[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        expected = {"frame", "mean_intensity_bgsub"}
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"CSV must contain columns {sorted(expected)}; got {reader.fieldnames}"
            )
        for row in reader:
            frames.append(int(row["frame"]))
            means.append(float(row["mean_intensity_bgsub"]))

    x = np.asarray(frames, dtype=int)
    y = np.asarray(means, dtype=float)
    return x, y


def _compute_threshold(y: np.ndarray, baseline_n: int, sigma: float) -> tuple[float, float, float]:
    if baseline_n <= 0:
        raise ValueError("baseline_n must be > 0")
    baseline_n = min(baseline_n, len(y))

    baseline = y[:baseline_n]
    baseline_mean = float(baseline.mean())
    baseline_std = float(baseline.std())
    threshold = baseline_mean + sigma * baseline_std
    return baseline_mean, baseline_std, threshold


def _save_plot(
    *,
    x: np.ndarray,
    y: np.ndarray,
    baseline_mean: float,
    threshold: float,
    sigma: float,
    out_path: Path,
    title: str,
    zoom_max_frame: int | None = None,
    dpi: int,
) -> None:
    plt.figure(figsize=(12, 4))

    if zoom_max_frame is None:
        plt.plot(x, y, lw=0.8)
    else:
        mask = x <= int(zoom_max_frame)
        plt.plot(x[mask], y[mask], lw=1.0)

    plt.axhline(baseline_mean, color="k", ls="--", lw=1, label="baseline mean")
    plt.axhline(
        threshold,
        color="r",
        ls="--",
        lw=1,
        label=f"baseline + {sigma}σ",
    )
    plt.title(title)
    plt.xlabel("frame")
    plt.ylabel("mean intensity")
    plt.legend(loc="upper right")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main() -> int:
    args = _parse_args()
    x, y = _read_csv(args.csv)

    baseline_mean, baseline_std, threshold = _compute_threshold(
        y=y,
        baseline_n=int(args.baseline_n),
        sigma=float(args.sigma),
    )

    out_dir: Path = args.out_dir
    out_full = out_dir / "emonet_mean_intensity.png"
    out_zoom = out_dir / "emonet_mean_intensity_zoom.png"

    _save_plot(
        x=x,
        y=y,
        baseline_mean=baseline_mean,
        threshold=threshold,
        sigma=float(args.sigma),
        out_path=out_full,
        title="Emonet: mean intensity per frame (background-subtracted, clipped)",
        zoom_max_frame=None,
        dpi=int(args.dpi),
    )
    _save_plot(
        x=x,
        y=y,
        baseline_mean=baseline_mean,
        threshold=threshold,
        sigma=float(args.sigma),
        out_path=out_zoom,
        title="Zoom: early frames",
        zoom_max_frame=int(args.zoom_max_frame),
        dpi=int(args.dpi),
    )

    print(f"baseline_mean={baseline_mean:.8f}")
    print(f"baseline_std={baseline_std:.8f}")
    print(f"threshold={threshold:.8f}")
    print(f"wrote {out_full}")
    print(f"wrote {out_zoom}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
