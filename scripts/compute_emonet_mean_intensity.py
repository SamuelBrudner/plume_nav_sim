from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-frame mean intensity after Emonet background subtraction "
            "(clip at 0), and write a CSV for plotting."
        )
    )
    parser.add_argument(
        "--mat",
        type=Path,
        required=True,
        help="Path to MATLAB v7.3/HDF5 frames .mat file",
    )
    parser.add_argument(
        "--frames-key",
        type=str,
        default="frames",
        help="HDF5 dataset key containing frames (default: frames)",
    )
    parser.add_argument(
        "--background-n",
        type=int,
        default=200,
        help="Number of initial frames used to estimate background (default: 200)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If >0, limit analysis to first N frames (default: 0 = all)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/tmp/emonet_mean_intensity.csv"),
        help="Output CSV path (default: /tmp/emonet_mean_intensity.csv)",
    )
    return parser.parse_args()


def _compute_background(*, dset: h5py.Dataset, n_bg: int) -> np.ndarray:
    bg: np.ndarray | None = None
    for i in range(n_bg):
        frame = dset[i]  # (X, Y)
        frame = np.transpose(frame, (1, 0)).astype(np.float32)  # -> (Y, X)
        if bg is None:
            bg = np.zeros_like(frame, dtype=np.float32)
        bg += frame
    if bg is None:
        raise RuntimeError("No frames read while computing background")
    return bg / float(n_bg)


def main() -> int:
    args = _parse_args()

    mat_path: Path = args.mat
    if not mat_path.exists():
        raise FileNotFoundError(mat_path)

    with h5py.File(mat_path, "r") as f:
        means = _extracted_from_main_9(f, args)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "mean_intensity_bgsub"])
        for i, m in enumerate(means.tolist()):
            w.writerow([i, m])

    print(f"wrote_csv={args.out_csv}")
    return 0


# TODO Rename this here and in `main`
def _extracted_from_main_9(f, args):
    dset = f[args.frames_key]
    n_total, n_x, n_y = (int(s) for s in dset.shape)  # (T, X, Y)

    n_frames = n_total
    if int(args.max_frames) > 0:
        n_frames = min(n_frames, int(args.max_frames))

    n_bg = min(int(args.background_n), n_frames)
    if n_bg <= 0:
        raise ValueError("background_n must be >= 1")

    print(f"frames_key={args.frames_key}")
    print(f"shape(T,X,Y)=({n_total},{n_x},{n_y}), dtype={dset.dtype}")
    print(f"analyzing_n_frames={n_frames}")
    print(f"background_n={n_bg}")

    bg = _compute_background(dset=dset, n_bg=n_bg)

    result = np.zeros(n_frames, dtype=np.float32)
    for t in range(n_frames):
        frame = dset[t]
        frame = np.transpose(frame, (1, 0)).astype(np.float32)
        frame = frame - bg
        np.clip(frame, 0.0, None, out=frame)
        result[t] = frame.mean()
        if t % 500 == 0:
            print(f"  computed {t}/{n_frames}")

    return result


if __name__ == "__main__":
    raise SystemExit(main())
