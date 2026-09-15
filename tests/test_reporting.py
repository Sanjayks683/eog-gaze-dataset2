"""
tests/test_reporting.py
=======================
Results tables with the published Dataset 2 numbers, and the XGBoost device switch.
"""

from __future__ import annotations

import json

import pytest

from src.config import CFG, _CONFIG_SECTIONS


@pytest.fixture()
def restore_cfg():
    snapshot = {section: dict(vars(getattr(CFG, section))) for section in _CONFIG_SECTIONS}
    yield CFG
    for section, values in snapshot.items():
        sub = getattr(CFG, section)
        for k, v in values.items():
            setattr(sub, k, v)


def _result(**extra):
    return {"pooled_rmse_h_deg": 6.0, "pooled_rmse_v_deg": 5.0, "pooled_mae_h_deg": 4.0, "pooled_mae_v_deg": 3.0,
            "fixation_mae_h_deg": 4.4, "fixation_mae_v_deg": 3.9, **extra}


def test_results_table_lists_both_models_and_the_published_numbers(tmp_path):
    from src.evaluation.compare_to_paper import build_master_table, save_master_table_csv

    (tmp_path / "baseline_reg_mean.json").write_text(json.dumps(_result(fixation_mae_h_deg=12.6)))
    (tmp_path / "classical_reg_xgb.json").write_text(
        json.dumps(_result(train_windows="weighted", nonfixation_weight=0.2)))

    rows = build_master_table(results_dir=str(tmp_path))

    assert [r["method"] for r in rows[:2]] == ["Baseline: train-fold mean angle", "XGBoost"]
    assert rows[0]["fixation_mae_h_deg"] == 12.6 and "weighted 0.2" in rows[1]["notes"]
    published = {r["method"]: r for r in rows if r["method"].startswith("Published")}
    assert len(published) == 4
    assert (published["Published: Barbara_2023_DKF_long"]["fixation_mae_h_deg"],
            published["Published: Barbara_2023_DKF_long"]["fixation_mae_v_deg"]) == (5.23, 6.59)
    assert all(r["rmse_h_deg"] is None for r in published.values())

    header = open(save_master_table_csv(rows, str(tmp_path)), encoding="utf-8").readline().strip()
    assert header == "method,rmse_h_deg,rmse_v_deg,mae_h_deg,mae_v_deg,fixation_mae_h_deg,fixation_mae_v_deg,notes"


def test_known_start_table_lists_the_four_published_rows():
    from src.evaluation.known_start import known_start_rows

    keys = ("mae_h_deg", "mae_v_deg", "mae_h_sd_deg", "mae_v_sd_deg",
            "excluded_mae_h_deg", "excluded_mae_v_deg", "excluded_fraction")
    fits = {k: {key: 1.0 for key in keys} for k in ("short_saccades", "short_level", "long_saccades")}
    rows = known_start_rows({"same_subject": fits, "unseen_subject": fits})

    assert len([r for r in rows if r["method"].startswith("Known start")]) == 6
    published = [r for r in rows if r["method"].startswith("Published")]
    assert len(published) == 4 and {r["excluded_mae_h_deg"] for r in published} == {1.64, 1.51, 5.23, 5.82}


def test_xgb_device_is_passed_only_when_it_is_not_the_cpu(restore_cfg):
    from src.models.classical_ml import build_xgb_regressor

    CFG.cv.xgb_device = "cpu"
    assert build_xgb_regressor().estimator.get_params().get("device") is None
    CFG.cv.xgb_device = "cuda"
    assert build_xgb_regressor().estimator.get_params()["device"] == "cuda"
