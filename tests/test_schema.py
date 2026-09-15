"""
tests/test_schema.py
====================
The Trial dataclass, and the Dataset 2 loader (skipped when the data is not downloaded).
"""

import numpy as np
import pytest

from src.data.schema import Trial, make_empty_event_timestamps


def test_dataset2_loads_a_valid_recording():
    from src.data.loaders import load_dataset2

    try:
        trials = load_dataset2(max_subjects=1)
    except FileNotFoundError as e:
        pytest.skip(f"Dataset 2 not downloaded: {e}")

    assert len(trials) == 1
    trial = trials[0]
    trial.validate()
    assert trial.dataset_source == "dataset2" and trial.montage_type == "monopolar" and trial.fs == 256.0
    assert {"EOG_0", "EOG_1", "EOG_2", "EOG_3", "ControlSignal"} <= set(trial.channels)
    assert trial.target_angle is not None and trial.target_angle.shape == (trial.n_samples(), 2)


def test_trial_repr_does_not_crash():
    t = Trial(subject_id="S01", trial_id="T001", dataset_source="dataset2", montage_type="monopolar",
              fs=256.0, channels={"H": np.zeros(500), "V": np.zeros(500)})
    assert "S01" in repr(t)


def test_trial_n_samples_and_duration():
    t = Trial(subject_id="S01", trial_id="T001", dataset_source="dataset2", montage_type="monopolar",
              fs=500.0, channels={"H": np.zeros(1000), "V": np.zeros(1000)})
    assert t.n_samples() == 1000
    assert abs(t.duration_s() - 2.0) < 1e-6


def test_trial_has_bipolar():
    t = Trial(subject_id="S01", trial_id="T001", dataset_source="dataset2", montage_type="monopolar",
              fs=256.0, channels={"EOG_0": np.zeros(500), "EOG_1": np.zeros(500)})
    assert not t.has_bipolar()
    t.channels["H"] = np.zeros(500)
    t.channels["V"] = np.zeros(500)
    assert t.has_bipolar()


def test_validate_rejects_event_timestamps_that_go_backwards():
    ts = make_empty_event_timestamps()
    ts["saccade1_onset"], ts["saccade1_end"] = 200, 100
    t = Trial(subject_id="S01", trial_id="T001", dataset_source="dataset2", montage_type="monopolar",
              fs=256.0, channels={"H": np.zeros(500)}, event_timestamps=ts)
    with pytest.raises(ValueError, match="monotonically"):
        t.validate()
