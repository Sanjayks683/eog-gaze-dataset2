"""
tests/test_robustness_scripts.py
================================
Pieces of the nested cross-validation and statistics scripts that can be checked
without the dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.nested_cv import evaluate, run_nested
from scripts.subject_statistics import compare


def test_nested_evaluate_scores_only_windows_with_a_prediction():
    y = np.zeros((4, 2))
    pred = np.array([[1.0, 1.0], [np.nan, np.nan], [3.0, 3.0], [1.0, 1.0]])
    out = evaluate(y, pred, np.array(["A", "A", "B", "B"]), np.array([True, True, True, False]))

    assert out["fixation_mae_h_deg"] == pytest.approx(2.0)
    assert out["rmse_h_deg"] == pytest.approx(np.sqrt(11 / 3))
    assert out["score_fixation"] == pytest.approx(2.0)


def test_run_nested_picks_the_candidate_that_wins_on_inner_folds():
    rng = np.random.RandomState(0)
    subjects = np.repeat([f"S{i}" for i in range(10)], 20)
    y = rng.randn(200, 2)
    fixation = np.ones(200, dtype=bool)

    def fit_predict(error, train_idx, test_idx, out):
        out[test_idx] = y[test_idx] + error

    selected, fixed, folds, _ = run_nested([0.5, 0.1, 1.0], fit_predict, y, subjects, fixation, name=str)
    assert all(fold["selected"] == {"fixation": "0.1", "rmse": "0.1"} for fold in folds)
    assert np.allclose(selected["fixation"], y + 0.1)
    assert np.allclose(fixed[1.0], y + 1.0)


def test_statistics_compare_pairs_subjects():
    reference = {f"S{i}": [2.0 + i, 3.0 + i] for i in range(10)}
    model = {s: [h - (0.5 + 0.1 * i), v + 0.2] for i, (s, (h, v)) in enumerate(reference.items())}
    out = compare(model, reference, "model", "reference")

    assert out["improvement_deg"] == pytest.approx([0.95, -0.2])
    lo, hi = out["improvement_ci95_deg"]["h"]
    assert 0.5 <= lo < 0.95 < hi <= 1.4
    assert out["subjects_improved"] == {"h": 10, "v": 0}
    assert out["wilcoxon_p"]["h"] < 0.01
