"""
src/data/unify.py
=================
Bipolar horizontal (H) and vertical (V) channels from Dataset 2's four monopolar
electrodes:

    H = −(EOG_2 − EOG_3)        V = EOG_0 − EOG_1

The electrode pairs and signs were verified against the recorded target angles with
scripts/verify_montage.py: a rightward saccade gives a positive H deflection and an
upward saccade a positive V deflection. The monopolar channels are kept alongside
H and V.
"""

from __future__ import annotations

from src.data.schema import Trial

MONOPOLAR_MAP = dict(
    right_key="EOG_2",
    left_key="EOG_3",
    up_key="EOG_0",
    down_key="EOG_1",
    flip_h=True,
    flip_v=False,
)


def bipolarize(trial: Trial, mapping: dict = None) -> Trial:
    """Add channels 'H' and 'V' to the trial (in place) and return it."""
    m = mapping or MONOPOLAR_MAP
    for k in (m["right_key"], m["left_key"], m["up_key"], m["down_key"]):
        if k not in trial.channels:
            raise KeyError(
                f"Channel key {k!r} not found in recording {trial.subject_id}/{trial.trial_id}. "
                f"Available channels: {list(trial.channels)}."
            )
    h = trial.channels[m["right_key"]] - trial.channels[m["left_key"]]
    v = trial.channels[m["up_key"]] - trial.channels[m["down_key"]]
    trial.channels["H"] = -h if m["flip_h"] else h
    trial.channels["V"] = -v if m["flip_v"] else v
    return trial


def to_common_schema(trial: Trial) -> Trial:
    """The pipeline's unification step: bipolar H and V for a Dataset 2 recording."""
    return bipolarize(trial)
