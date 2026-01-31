from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview an ingested movie plume (e.g., emonet_smoke_v1) with an exponential "
            "decay correction applied at render time, then write a GIF."
        )
    )

    parser.add_argument(
        "--movie-dataset-id",
        type=str,
        default="emonet_smoke_v1",
        help="Registry dataset id (default: emonet_smoke_v1)",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Optional cache root override (defaults to ~/.cache/plume_nav_sim/data_zoo)",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Auto-download dataset if missing (default: false)",
    )
    parser.add_argument(
        "--zarr",
        type=Path,
        default=None,
        help="Direct path to Zarr dataset root (overrides --movie-dataset-id)",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/emonet_smoke_decay_corrected.gif"),
        help="Output GIF path (default: /tmp/emonet_smoke_decay_corrected.gif)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="GIF fps (default: 10)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=600,
        help="Max frames to render into GIF (default: 600)",
    )

    parser.add_argument(
        "--fit-start",
        type=int,
        default=0,
        help="Start frame for exponential fit (default: 0)",
    )
    parser.add_argument(
        "--fit-end",
        type=int,
        default=0,
        help="End frame for exponential fit (0 means auto; default: 0)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Epsilon floor for log-fitting (default: 1e-6)",
    )
    parser.add_argument(
        "--max-gain",
        type=float,
        default=10.0,
        help="Cap for decay correction gain multiplier (default: 10.0)",
    )

    parser.add_argument(
        "--vmax-quantile",
        type=float,
        default=0.999,
        help="Quantile used for grayscale normalization (default: 0.999)",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=100,
        help="Number of frames sampled to estimate vmax (default: 100)",
    )
    parser.add_argument(
        "--samples-per-frame",
        type=int,
        default=50_000,
        help="Pixel samples per frame for vmax estimation (default: 50000)",
    )

    return parser.parse_args()


def _resolve_zarr_path(args: argparse.Namespace) -> Path:
    if args.zarr is not None:
        zarr_path = Path(args.zarr)
        if not zarr_path.exists():
            raise FileNotFoundError(zarr_path)
        return zarr_path

    from plume_nav_sim.data_zoo.download import ensure_dataset_available

    cache_root: Path | None = None
    if args.cache_root is not None:
        cache_root = Path(args.cache_root).expanduser()

    return ensure_dataset_available(
        str(args.movie_dataset_id),
        cache_root=cache_root,
        auto_download=bool(args.auto_download),
    )


def _open_concentration(zarr_path: Path):
    import zarr

    store = zarr.DirectoryStore(str(zarr_path))
    root = zarr.open_group(store, mode="r")
    if "concentration" not in root:
        raise KeyError(f"Zarr root missing 'concentration': {zarr_path}")
    return root, root["concentration"]


def _iter_time_chunks(n_frames: int, chunk_t: int) -> Iterable[tuple[int, int]]:
    for t0 in range(0, n_frames, chunk_t):
        t1 = min(t0 + chunk_t, n_frames)
        yield t0, t1


def _compute_means(conc, *, chunk_t: int) -> np.ndarray:
    n_frames = int(conc.shape[0])
    means = np.zeros(n_frames, dtype=np.float64)
    for t0, t1 in _iter_time_chunks(n_frames, chunk_t):
        chunk = conc[t0:t1]
        means[t0:t1] = chunk.mean(axis=(1, 2))
    return means


def _fit_exponential_decay(
    means: np.ndarray,
    *,
    fit_start: int,
    fit_end: int,
    eps: float,
) -> tuple[float, float]:
    n = int(means.shape[0])
    start = max(0, int(fit_start))
    end = int(fit_end)
    if end <= 0 or end > n:
        end = n

    y = means[start:end].astype(np.float64, copy=False)
    t = np.arange(start, end, dtype=np.float64)

    y_safe = np.maximum(y, float(eps))
    logy = np.log(y_safe)

    # logy ~ b + m*t
    m, b = np.polyfit(t, logy, deg=1)

    if not np.isfinite(m) or not np.isfinite(b):
        raise ValueError(f"Non-finite fit (m={m}, b={b})")

    return float(m), float(b)


def _compute_gain(*, m: float, t: np.ndarray, max_gain: float) -> np.ndarray:
    # If mean ~ exp(b + m t), then multiply by exp(-m t) to flatten.
    gain = np.exp(-m * t).astype(np.float32)
    if max_gain > 0:
        gain = np.minimum(gain, float(max_gain))
    return gain


def _estimate_vmax(
    conc,
    *,
    gain: np.ndarray,
    max_frames: int,
    sample_frames: int,
    samples_per_frame: int,
    quantile: float,
) -> float:
    n_frames = min(int(conc.shape[0]), int(max_frames))
    if n_frames <= 0:
        raise ValueError("No frames available")

    n_take = min(int(sample_frames), n_frames)
    rng = np.random.default_rng(42)
    frame_idx = np.sort(rng.choice(n_frames, size=n_take, replace=False))

    samples: list[np.ndarray] = []
    for i, t in enumerate(frame_idx.tolist()):
        frame = conc[int(t)].astype(np.float32, copy=False)
        frame = frame * float(gain[int(t)])
        flat = frame.reshape(-1)
        k = min(int(samples_per_frame), int(flat.size))
        if k <= 0:
            continue
        # Deterministic per-frame sampling for reproducibility.
        rng_i = np.random.default_rng(1000 + i)
        idx = rng_i.choice(flat.size, size=k, replace=False)
        samples.append(flat[idx])

    if not samples:
        raise ValueError("Failed to sample any pixels for vmax")

    all_samples = np.concatenate(samples)
    vmax = float(np.quantile(all_samples, float(quantile)))
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError(f"Invalid vmax={vmax}")
    return vmax


def _frame_to_rgb(frame: np.ndarray, *, vmax: float) -> np.ndarray:
    x = frame.astype(np.float32, copy=False) / float(vmax)
    x = np.clip(x, 0.0, 1.0)
    u8 = (x * 255.0).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def main() -> int:
    args = _parse_args()

    zarr_path = _resolve_zarr_path(args)
    root, conc = _open_concentration(zarr_path)

    n_frames = int(conc.shape[0])
    chunk_t = int(getattr(conc, "chunks", (100,))[0] or 100)

    means = _compute_means(conc, chunk_t=chunk_t)

    m, b = _fit_exponential_decay(
        means,
        fit_start=int(args.fit_start),
        fit_end=int(args.fit_end),
        eps=float(args.eps),
    )

    # Report a per-frame decay time constant in frames (tau = -1/m).
    tau_frames = float("inf")
    if m < 0:
        tau_frames = -1.0 / m

    max_frames = min(int(args.max_frames), n_frames)
    t = np.arange(n_frames, dtype=np.float32)
    gain = _compute_gain(m=m, t=t, max_gain=float(args.max_gain))

    vmax = _estimate_vmax(
        conc,
        gain=gain,
        max_frames=max_frames,
        sample_frames=int(args.sample_frames),
        samples_per_frame=int(args.samples_per_frame),
        quantile=float(args.vmax_quantile),
    )

    print(f"dataset={zarr_path}")
    print(f"n_frames={n_frames}")
    print(f"chunks_t={chunk_t}")
    print(f"fit: log(mean)=b+m*t with m={m:.6e}, b={b:.6f}")
    print(f"tau_frames={tau_frames:.2f}")
    fps = float(
        getattr(root.attrs, "get", lambda *_: None)("fps")
        or root.attrs.get("fps", 90.0)
    )
    print(f"tau_seconds={tau_frames / float(fps):.3f}")
    print(f"vmax_quantile={float(args.vmax_quantile)}")
    print(f"vmax={vmax:.6f}")

    frames_rgb: list[np.ndarray] = []
    for ti in range(max_frames):
        frame = conc[ti].astype(np.float32, copy=False)
        frame = frame * float(gain[ti])
        frames_rgb.append(_frame_to_rgb(frame, vmax=vmax))

    from plume_nav_sim.plume.video import save_video_frames

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_video_frames(frames_rgb, args.out, fps=int(args.fps))
    print(f"wrote_gif={args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
