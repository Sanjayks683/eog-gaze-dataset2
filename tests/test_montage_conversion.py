"""
tests/test_montage_conversion.py
================================
Bipolar H and V from Dataset 2's four monopolar electrodes (src/data/unify.py).
"""

import numpy as np
import pytest

from src.data.schema import Trial
from src.data.unify import MONOPOLAR_MAP, bipolarize, to_common_schema


def _trial(n: int = 500, **values) -> Trial:
    channels = {k: np.full(n, float(values.get(k, 0.0))) for k in ("EOG_0", "EOG_1", "EOG_2", "EOG_3")}
    return Trial(subject_id="S1", trial_id="S1_trial", dataset_source="dataset2", montage_type="monopolar",
                 fs=256.0, channels=channels)


def test_verified_map_gives_h_as_minus_eog2_minus_eog3_and_v_as_eog0_minus_eog1():
    trial = to_common_schema(_trial(EOG_0=1.2, EOG_1=-0.8, EOG_2=2.5, EOG_3=-1.5))

    np.testing.assert_allclose(trial.channels["H"], -(2.5 - (-1.5)))
    np.testing.assert_allclose(trial.channels["V"], 1.2 - (-0.8))
    assert all(k in trial.channels for k in ("EOG_0", "EOG_1", "EOG_2", "EOG_3"))


def test_bipolar_channels_keep_the_recording_length():
    trial = to_common_schema(_trial(n=731))
    assert len(trial.channels["H"]) == len(trial.channels["V"]) == 731


def test_a_map_without_flips_gives_right_minus_left_and_up_minus_down():
    mapping = dict(MONOPOLAR_MAP, flip_h=False, flip_v=False)
    trial = bipolarize(_trial(EOG_0=4.0, EOG_1=1.0, EOG_2=3.0, EOG_3=1.0), mapping)
    np.testing.assert_allclose(trial.channels["H"], 2.0)
    np.testing.assert_allclose(trial.channels["V"], 3.0)


def test_missing_electrode_raises():
    trial = _trial()
    del trial.channels["EOG_3"]
    with pytest.raises(KeyError, match="EOG_3"):
        to_common_schema(trial)
