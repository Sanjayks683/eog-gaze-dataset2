"""
src/data/datasets.py
=====================
Windowing, regression targets, fixation flags and processed-data caching.

Produces ML-ready (X, y_angle, metadata) arrays from preprocessed Trials.
"""

from __future__ import annotations

import os
import pickle
import warnings
from typing import List, Tuple

import numpy as np

from src.config import CFG, window_samples, stride_samples
from src.data.schema import Trial


def _window_channel_arrays(trial: Trial) -> Tuple[List[str], List[np.ndarray]]:
    """
    Channels stacked into model input windows, per CFG.data.window_channel_mode:
      "bipolar" → [H, V]
      "all_eog" → [H, V] + sorted EOG_* channels (ControlSignal and other
                  non-EOG channels are never included).
    """
    names = ["H", "V"]
    arrays = [trial.channels["H"], trial.channels["V"]]
    if getattr(CFG.data, "window_channel_mode", "bipolar") == "all_eog":
        eog_keys = sorted(k for k in trial.channels if k.startswith("EOG_"))
        names = names + eog_keys
        arrays = arrays + [trial.channels[k] for k in eog_keys]
    return names, arrays


def _fixation_flags(trial: Trial, starts: np.ndarray, win: int) -> np.ndarray:
    """
    Flag windows that fall on a settled fixation, the only samples Barbara et al.
    (BSPC 2023) score gaze error on: inside a single ControlSignal 1/2 interval (one
    cue position), starting at least CFG.segmentation.fixation_settle_ms after that
    interval began. All False without targets or ControlSignal.
    """
    starts = np.asarray(starts, dtype=np.int64)
    if trial.target_angle is None or "ControlSignal" not in trial.channels or len(starts) == 0:
        return np.zeros(len(starts), dtype=bool)
    cs = np.asarray(trial.channels["ControlSignal"])
    n = min(len(trial.target_angle), len(cs))
    cs = cs[:n]
    onset = np.r_[True, cs[1:] != cs[:-1]]
    interval = np.cumsum(onset) - 1
    interval_start = np.flatnonzero(onset)[interval]
    ends = np.minimum(starts + win - 1, n - 1)
    settle = int(round(CFG.segmentation.fixation_settle_ms * trial.fs / 1000.0))
    return (np.isin(cs[starts], (1, 2)) & (interval[starts] == interval[ends])
            & (starts - interval_start[starts] >= settle))


def make_regression_targets(trials: List[Trial]) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Slide windows over every trial and pair each with its mean target gaze angle.

    Returns
    -------
    X       : (n_windows, n_channels, window_len) float32
    y_angle : (n_windows, 2) float32 — mean H and V angle per window (degrees)
    metadata: list of dicts (subject_id, trial_id, window_start, fs, channel_names, is_fixation)
    """
    X_list, y_list, meta_list = [], [], []

    for trial in trials:
        if not trial.has_bipolar():
            warnings.warn(f"Recording {trial.subject_id}/{trial.trial_id} has no H/V channels — skipping.")
            continue
        if trial.target_angle is None:
            warnings.warn(f"Recording {trial.subject_id}/{trial.trial_id} has no target angles — skipping.")
            continue

        ch_names, ch_arrays = _window_channel_arrays(trial)
        n = min(len(a) for a in ch_arrays)
        win = window_samples(trial.fs)
        stride = stride_samples(trial.fs)
        if n < win:
            continue

        angle_arr = trial.target_angle[:n]
        starts = np.arange(0, n - win + 1, stride)
        fixation = _fixation_flags(trial, starts, win)

        for start, is_fixation in zip(starts.tolist(), fixation.tolist()):
            end = start + win
            X_list.append(np.stack([a[start:end] for a in ch_arrays], axis=0).astype(np.float32))
            y_list.append(np.mean(angle_arr[start:end], axis=0).astype(np.float32))
            meta_list.append({
                "subject_id": trial.subject_id,
                "trial_id": trial.trial_id,
                "dataset_source": trial.dataset_source,
                "window_start": start,
                "fs": trial.fs,
                "channel_names": ch_names,
                "is_fixation": is_fixation,
            })

    n_ch = len(X_list[0]) if X_list else 2
    X = np.stack(X_list, axis=0) if X_list else np.empty((0, n_ch, 0), dtype=np.float32)
    y = np.array(y_list, dtype=np.float32) if y_list else np.empty((0, 2), dtype=np.float32)
    return X, y, meta_list


def save_processed(X: np.ndarray, y: np.ndarray, metadata: list, tag: str, processed_dir: str = None) -> str:
    """Save processed arrays to disk. Returns the saved file path."""
    if processed_dir is None:
        processed_dir = CFG.paths.data_processed
    os.makedirs(processed_dir, exist_ok=True)
    path = os.path.join(processed_dir, f"{tag}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"X": X, "y": y, "metadata": metadata}, f, protocol=4)
    print(f"Saved processed data: {path}  (X={X.shape}, y={y.shape})")
    return path


def load_processed(tag: str, processed_dir: str = None) -> Tuple[np.ndarray, np.ndarray, list]:
    """Load processed arrays from disk."""
    if processed_dir is None:
        processed_dir = CFG.paths.data_processed
    path = os.path.join(processed_dir, f"{tag}.pkl")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Processed file not found: {path}. Run `python main.py --phase preprocess` first.")
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["X"], d["y"], d["metadata"]
