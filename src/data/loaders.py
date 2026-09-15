"""
src/data/loaders.py
===================
Raw loader for EyeCon Dataset 2 (Monopolar, Stationary head).

Download Dataset_Stationary.zip from https://www.um.edu.mt/cbc/ourprojects/eyecon/eogdataset/
and extract it into data/raw/Dataset_Stationary/, which then holds one folder per
subject (S1 ... S10) with EOG.mat, ControlSignal.mat and TargetGA.mat.

load_trials() returns one Trial per subject with the raw monopolar channels
(EOG_0 ... EOG_3), the ControlSignal and the per-sample target gaze angles.
"""

from __future__ import annotations

import glob
import os
import re
import warnings
from typing import Dict, List, Optional

import numpy as np
import scipy.io as sio

from src.config import CFG
from src.data.schema import Trial, make_empty_event_timestamps

DOWNLOAD_URL = "https://www.um.edu.mt/cbc/ourprojects/eyecon/eogdataset/"


def _check_dir(path: str) -> None:
    """Raise FileNotFoundError if the dataset directory is missing or empty."""
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"\n[Dataset 2] Directory not found: {path}\n"
            f"Download Dataset_Stationary.zip from {DOWNLOAD_URL} and extract it there."
        )
    if not [f for f in os.listdir(path) if not f.startswith(".")]:
        raise FileNotFoundError(
            f"\n[Dataset 2] Directory exists but is EMPTY: {path}\n"
            f"Extract Dataset_Stationary.zip (from {DOWNLOAD_URL}) into this folder."
        )


def _load_mat_safe(filepath: str) -> dict:
    """Load .mat file, trying scipy first, then h5py for v7.3 files."""
    try:
        return sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        try:
            import h5py
            data = {}
            with h5py.File(filepath, "r") as f:
                def _visit(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        data[name.replace("/", "_")] = np.array(obj)
                f.visititems(_visit)
            return data
        except ImportError:
            raise ImportError(
                "This .mat file uses HDF5 format (MATLAB v7.3). "
                "Install h5py: pip install h5py"
            )


def _discover_subject_structure(root: str) -> List[str]:
    """Sorted subject subdirectories (S1/, S2/, ...), searched recursively."""
    candidates = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "__MACOSX" in dirpath:
            continue
        if re.match(r"(?i)^(sub|s\d+|subject)", os.path.basename(dirpath)):
            candidates.append(dirpath)
    return sorted(set(candidates))


def _parse_events_into(val, event_timestamps: Dict[str, int]) -> None:
    """Try to parse an event/label array into the event_timestamps dict."""
    try:
        arr = np.atleast_1d(np.array(val)).flatten()
        int_arr = arr[~np.isnan(arr.astype(float))].astype(int)
        keys_in_order = [
            "saccade1_onset", "saccade1_end",
            "saccade2_onset", "saccade2_end",
            "blink_onset", "blink_end",
        ]
        for i, k in enumerate(keys_in_order):
            if i < len(int_arr):
                event_timestamps[k] = int(int_arr[i])
    except Exception:
        pass


def _parse_mat_trial(mat_data: dict, subject_id: str, trial_id: str, fs: float) -> Trial:
    """
    Convert the merged .mat contents of one subject into a Trial. Channel arrays,
    target angles and event markers are recognised by their variable names.
    """
    channels: Dict[str, np.ndarray] = {}
    event_timestamps = make_empty_event_timestamps()
    target_angle: Optional[np.ndarray] = None
    detected_fs: float = fs

    EOG_CHANNEL_PATTERNS = re.compile(
        r"(?i)(eog|heog|veog|hori|vert|left|right|up|down|chan|ch\d+|signal|data)", re.IGNORECASE
    )
    FS_PATTERNS = re.compile(r"(?i)(fs|srate|samplingrate|sampling_rate|freq)")
    ANGLE_PATTERNS = re.compile(r"(?i)(angle|gaze|target|degree|deg)")
    EVENT_PATTERNS = re.compile(r"(?i)(event|onset|trigger|marker|timestamp|label)")

    for key, val in mat_data.items():
        if key.startswith("__"):
            continue

        if FS_PATTERNS.search(key) and np.isscalar(val):
            detected_fs = float(val)
            continue

        if ANGLE_PATTERNS.search(key) and isinstance(val, np.ndarray) and val.ndim >= 1:
            arr = np.array(val, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] in (1, 2) and arr.shape[1] > 10:
                arr = arr.T
            if arr.shape[-1] in (1, 2) or (arr.ndim == 1 and len(arr) > 10):
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]
                if arr.shape[1] == 1:
                    arr = np.hstack([arr, np.zeros_like(arr)])
                target_angle = arr
            continue

        if EVENT_PATTERNS.search(key):
            _parse_events_into(val, event_timestamps)
            continue

        if EOG_CHANNEL_PATTERNS.search(key) and isinstance(val, np.ndarray):
            if val.ndim == 1 and len(val) > 10:
                channels[key] = np.array(val, dtype=np.float64)
            elif val.ndim == 2:
                for i in range(val.shape[0] if val.shape[0] < val.shape[1] else val.shape[1]):
                    if val.shape[0] < val.shape[1]:
                        channels[f"{key}_{i}"] = val[i].astype(np.float64)
                    else:
                        channels[f"{key}_{i}"] = val[:, i].astype(np.float64)
            continue

        if isinstance(val, np.ndarray) and val.ndim == 1 and len(val) > 100:
            channels[f"ch_{key}"] = np.array(val, dtype=np.float64)

    return Trial(
        subject_id=subject_id,
        trial_id=trial_id,
        dataset_source="dataset2",
        montage_type="monopolar",
        fs=detected_fs,
        channels=channels,
        event_timestamps=event_timestamps,
        target_angle=target_angle,
        metadata={"source_file": trial_id},
    )


def load_dataset2(max_subjects: Optional[int] = None) -> List[Trial]:
    """One Trial per subject from data/raw/Dataset_Stationary/."""
    root = CFG.paths.dataset_dir
    _check_dir(root)
    subject_dirs = _discover_subject_structure(root)
    if not subject_dirs:
        raise FileNotFoundError(f"No subject folders (S1, S2, ...) found under {root}.")

    trials: List[Trial] = []
    for subj_dir in (subject_dirs[:max_subjects] if max_subjects else subject_dirs):
        subject_id = os.path.basename(subj_dir)
        combined = {}
        for fpath in sorted(glob.glob(os.path.join(subj_dir, "**", "*.mat"), recursive=True)):
            try:
                combined.update(_load_mat_safe(fpath))
            except Exception as e:
                warnings.warn(f"Skipping {fpath}: {e}")
        if not combined:
            continue
        try:
            trial = _parse_mat_trial(combined, subject_id, f"{subject_id}_trial", CFG.data.fs_hz)
            if "ControlSignal" in combined:
                trial.channels["ControlSignal"] = np.array(combined["ControlSignal"]).flatten()
            if "TargetGA" in combined:
                trial.metadata["TargetGA"] = np.array(combined["TargetGA"])
            trial.validate()
            trials.append(trial)
        except Exception as e:
            warnings.warn(f"Failed to parse the recording of {subject_id}: {e}")

    print(f"[Dataset 2] Loaded {len(trials)} recordings from {root}")
    return trials


def load_trials() -> List[Trial]:
    """All Dataset 2 recordings (the pipeline's entry point)."""
    return load_dataset2()
