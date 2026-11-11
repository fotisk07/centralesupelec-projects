from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

VALID_LABELS = [0, 1, 2, 3, 4]
SAMPLING_RATE = 100
EPOCH_DURATION_SECONDS = 30
NUM_CHANNELS = 5
SAMPLES_PER_CHANNEL = SAMPLING_RATE * EPOCH_DURATION_SECONDS


def _discover_record_ids(base_dir: Path, split: str) -> list[int]:
    """Return the ordered list of record IDs for a split."""
    manifest_path = base_dir / "records_manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        subset = manifest[manifest["split"] == split]
        return subset["record_id"].astype(int).tolist()

    signal_dir = base_dir / f"signals_{split}"
    record_ids: list[int] = []
    for path in sorted(signal_dir.glob("signal_*.npy")):
        try:
            record_ids.append(int(path.stem.split("_")[-1]))
        except ValueError as exc:
            raise ValueError(f"Unexpected signal filename: {path.name}") from exc
    return record_ids


def _load_signal_array(path: Path) -> np.ndarray:
    """Load and validate a signal array shaped (NUM_CHANNELS, samples)."""
    array = np.load(path).astype(np.float32)
    if array.ndim != 2 or array.shape[0] != NUM_CHANNELS:  # noqa: PLR2004
        raise ValueError(
            f"Signal array at {path} has shape {array.shape}, expected ({NUM_CHANNELS}, N_samples).",
        )
    if array.shape[1] % SAMPLES_PER_CHANNEL != 0:
        raise ValueError(
            f"Signal array at {path} has {array.shape[1]} samples, not divisible by {SAMPLES_PER_CHANNEL}.",
        )
    return array


def _load_target_array(path: Path) -> np.ndarray:
    """Load the per-epoch labels (1D, int16)."""
    array = np.load(path).astype(np.int16)
    if array.ndim == 2 and array.shape[0] == NUM_CHANNELS:  # noqa: PLR2004
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"Target array at {path} must be 1D, found shape {array.shape}.")
    return array


def iter_signals_and_targets(base_dir: Path, split: str) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (record_id, signal_array, target_array) tuples for the chosen split."""
    record_ids = _discover_record_ids(base_dir, split)
    if not record_ids:
        raise FileNotFoundError(f"No records found for split={split} under {base_dir}.")

    signal_dir = base_dir / f"signals_{split}"
    target_dir = base_dir / f"targets_{split}"

    for record_id in record_ids:
        signal_path = signal_dir / f"signal_{record_id}.npy"
        target_path = target_dir / f"target_{record_id}.npy"
        if not signal_path.exists() or not target_path.exists():
            raise FileNotFoundError(f"Missing signal/target pair for record {record_id} in split {split}.")
        signal_array = _load_signal_array(signal_path)
        target_array = _load_target_array(target_path)
        yield record_id, signal_array, target_array


def iter_signals_only(base_dir: Path, split: str) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (record_id, signal_array) tuples without requiring targets."""
    record_ids = _discover_record_ids(base_dir, split)
    if not record_ids:
        raise FileNotFoundError(f"No records found for split={split} under {base_dir}.")

    signal_dir = base_dir / f"signals_{split}"

    for record_id in record_ids:
        signal_path = signal_dir / f"signal_{record_id}.npy"
        if not signal_path.exists():
            raise FileNotFoundError(f"Missing signal for record {record_id} in split {split}.")
        signal_array = _load_signal_array(signal_path)
        yield record_id, signal_array
