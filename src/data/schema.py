"""
src/data/schema.py
==================
The Trial dataclass used throughout the pipeline: one Dataset 2 recording.

Fields
------
subject_id      : str   — subject identifier, e.g. "S1"
trial_id        : str   — recording identifier, e.g. "S1_trial"
dataset_source  : str   — "dataset2"
montage_type    : str   — "monopolar"
fs              : float — sampling rate in Hz
channels        : dict  — channel name → numpy ndarray (n_samples,)
event_timestamps: dict  — see REQUIRED_EVENT_KEYS; -1 when unknown
target_angle    : numpy.ndarray or None — (n_samples, 2): [:, 0] = H angle (deg), [:, 1] = V angle (deg)
metadata        : dict  — free-form extras
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


REQUIRED_EVENT_KEYS = [
    "saccade1_onset",
    "saccade1_end",
    "saccade2_onset",
    "saccade2_end",
    "blink_onset",
    "blink_end",
]


def make_empty_event_timestamps() -> Dict[str, int]:
    """Return a dict with all required event keys set to -1 (unknown)."""
    return {k: -1 for k in REQUIRED_EVENT_KEYS}


@dataclass
class Trial:
    subject_id: str
    trial_id: str
    dataset_source: str
    montage_type: str
    fs: float

    channels: Dict[str, np.ndarray] = field(default_factory=dict)

    event_timestamps: Dict[str, int] = field(default_factory=make_empty_event_timestamps)

    target_angle: Optional[np.ndarray] = None

    metadata: Dict = field(default_factory=dict)

    def validate(self) -> None:
        """Raise ValueError if the Trial is structurally invalid (called by the loader)."""
        if not self.channels:
            raise ValueError(f"Trial {self.subject_id}/{self.trial_id}: channels dict is empty")
        if self.fs <= 0:
            raise ValueError(f"Trial {self.subject_id}/{self.trial_id}: fs={self.fs} must be > 0")
        for name, arr in self.channels.items():
            if arr is None or len(arr) == 0:
                raise ValueError(
                    f"Trial {self.subject_id}/{self.trial_id}: channel '{name}' is empty"
                )
        self._validate_event_timestamps()

    def _validate_event_timestamps(self) -> None:
        """
        Check that provided (non-sentinel) event timestamps are monotonically
        non-decreasing. Sentinel value -1 means 'not available' and is skipped.
        """
        known = [(k, v) for k, v in self.event_timestamps.items() if v != -1]
        order = {k: i for i, k in enumerate(REQUIRED_EVENT_KEYS)}
        known_sorted = sorted(known, key=lambda kv: order.get(kv[0], 99))
        values = [v for _, v in known_sorted]
        for i in range(1, len(values)):
            if values[i] < values[i - 1]:
                raise ValueError(
                    f"Trial {self.subject_id}/{self.trial_id}: "
                    f"event_timestamps are not monotonically non-decreasing: {known_sorted}"
                )

    def n_samples(self) -> int:
        """Return the length of the first channel array."""
        if not self.channels:
            return 0
        return len(next(iter(self.channels.values())))

    def duration_s(self) -> float:
        """Trial duration in seconds."""
        return self.n_samples() / self.fs if self.fs > 0 else 0.0

    def has_bipolar(self) -> bool:
        """True if channels contain 'H' and 'V' keys (after bipolarisation)."""
        return "H" in self.channels and "V" in self.channels

    def __repr__(self) -> str:
        ch_names = list(self.channels.keys())
        return (
            f"Trial(subject={self.subject_id!r}, trial={self.trial_id!r}, "
            f"dataset={self.dataset_source!r}, montage={self.montage_type!r}, "
            f"fs={self.fs} Hz, channels={ch_names}, "
            f"n_samples={self.n_samples()}, duration={self.duration_s():.2f}s)"
        )
