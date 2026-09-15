"""
tests/test_pipeline_e2e.py
==========================
The whole regression pipeline on synthetic Dataset-2-like recordings: bipolarisation,
drift removal, windowing, fixation flags, engineered and context features, cross-subject
folds, the mean-angle baseline and XGBoost, and the results table.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import CFG, _CONFIG_SECTIONS
from src.data.datasets import make_regression_targets
from src.data.schema import Trial
from src.data.unify import to_common_schema
from src.evaluation.compare_to_paper import build_master_table
from src.features.context import make_context_features
from src.features.engineered import extract_features_batch, get_feature_names
from src.models.classical_ml import run_regression_cv, save_results
from src.preprocessing.normalize import preprocess_trials
from src.training.cv_splits import get_folds, get_subject_ids_from_metadata

FS = 256.0


@pytest.fixture()
def restore_cfg():
    snapshot = {section: dict(vars(getattr(CFG, section))) for section in _CONFIG_SECTIONS}
    yield CFG
    for section, values in snapshot.items():
        sub = getattr(CFG, section)
        for k, v in values.items():
            setattr(sub, k, v)


def _recording(subject: str, seed: int, n_trials: int = 40) -> Trial:
    """Cue at P1 for 1 s, P2 for 1 s, blink interval 2 s; EOG = gain x gaze + drift + noise."""
    rng = np.random.RandomState(seed)
    sec = int(FS)
    cs, target = [], []
    for _ in range(n_trials):
        p1, p2 = rng.uniform(-20, 20, 2), rng.uniform(-20, 20, 2)
        cs += [1] * sec + [2] * sec + [3] * 2 * sec
        target += [p1] * sec + [p2] * 3 * sec
    target = np.array(target)
    n = len(cs)
    drift = np.linspace(0, 30, n)
    h = 10 * target[:, 0] + drift + rng.randn(n)
    v = 8 * target[:, 1] - drift + rng.randn(n)
    # H = -(EOG_2 - EOG_3) = h and V = EOG_0 - EOG_1 = v under the verified montage
    channels = {"EOG_0": v / 2, "EOG_1": -v / 2, "EOG_2": -h / 2, "EOG_3": h / 2, "ControlSignal": np.array(cs)}
    return Trial(subject_id=subject, trial_id=f"{subject}_trial", dataset_source="dataset2",
                 montage_type="monopolar", fs=FS, channels=channels, target_angle=target)


def test_regression_pipeline_runs_end_to_end_on_synthetic_recordings(tmp_path, restore_cfg):
    pp = CFG.preprocessing
    pp.drift_removal_method, pp.baseline_window_sec, pp.baseline_causal = "robust_line", 20.0, True
    pp.context_baselines, pp.context_lags_sec, pp.context_range_windows_sec = [["robust_mean", 10.0]], [1.0], [20.0]
    CFG.data.window_channel_mode = "all_eog"
    CFG.cv.regression_train_windows, CFG.cv.xgb_device = "weighted", "cpu"

    trials = [to_common_schema(_recording(f"S{i}", i)) for i in range(4)]
    raw = [{k: v.copy() for k, v in t.channels.items()} for t in trials]
    trials = preprocess_trials(trials)
    X, y, meta = make_regression_targets(trials)

    names = meta[0]["channel_names"]
    assert X.shape[1] == 6 and names == ["H", "V", "EOG_0", "EOG_1", "EOG_2", "EOG_3"]
    assert 0.1 < np.mean([m["is_fixation"] for m in meta]) < 0.9
    context = make_context_features(trials, raw, meta)
    features = np.hstack([extract_features_batch(X, fs=FS, channel_names=names), context])
    assert features.shape[1] == len(get_feature_names(6, names)) + context.shape[1]

    folds = get_folds(get_subject_ids_from_metadata(meta), strategy="group_kfold", k=2)
    mean = run_regression_cv(features, y, folds, model_name="mean", metadata=meta)
    xgb = run_regression_cv(features, y, folds, model_name="xgb", metadata=meta)
    assert xgb["fixation_mae_h_deg"] < mean["fixation_mae_h_deg"]
    assert xgb["fixation_mae_v_deg"] < mean["fixation_mae_v_deg"]

    save_results(mean, "baseline_reg_mean", str(tmp_path))
    save_results(xgb, "classical_reg_xgb", str(tmp_path))
    rows = build_master_table(results_dir=str(tmp_path))
    assert rows[1]["method"] == "XGBoost" and rows[1]["fixation_mae_h_deg"] == xgb["fixation_mae_h_deg"]
