"""
src/models/classical_ml.py
===========================
Gaze regression:
  XGBoost — one XGBRegressor per axis (horizontal, vertical)
  Trivial floor — always predict the training folds' mean angle; every model must beat it

Results are saved to JSON, never only printed.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.config import CFG
from src.evaluation.metrics import compute_fixation_metrics


def build_xgb_regressor():
    """XGBRegressor per axis (via MultiOutputRegressor)."""
    # The device is only passed when it is not the CPU, so CPU runs keep exactly the
    # arguments the committed results were produced with.
    device = {} if CFG.cv.xgb_device == "cpu" else {"device": CFG.cv.xgb_device}
    xgb = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=CFG.cv.random_seed, n_jobs=-1, verbosity=0, **device
    )
    return MultiOutputRegressor(xgb, n_jobs=1)


def fit_predict_regression_fold(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str = "xgb",
    fixation: Optional[np.ndarray] = None,
    train_windows: str = "all",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit one regressor ("xgb" or "mean") on train_idx and predict test_idx, applying
    the training-window choice ("all", "fixation" or "weighted"; see run_regression_cv).

    Returns (y_pred, y_true, test_idx) for the test windows with finite targets.
    """
    if train_windows == "fixation":
        train_idx = train_idx[fixation[train_idx]]
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    w_tr = (np.where(fixation[train_idx], 1.0, CFG.cv.regression_nonfixation_weight)
            if train_windows == "weighted" else None)

    valid_tr = ~np.isnan(y_tr).any(axis=1)
    if not valid_tr.all():
        X_tr, y_tr = X_tr[valid_tr], y_tr[valid_tr]
        w_tr = None if w_tr is None else w_tr[valid_tr]
    valid_te = ~np.isnan(y_te).any(axis=1)
    if not valid_te.all():
        X_te, y_te = X_te[valid_te], y_te[valid_te]

    if model_name == "xgb":
        model = build_xgb_regressor()
    elif model_name == "mean":
        model = DummyRegressor(strategy="mean")
    else:
        raise ValueError(f"Unknown regressor: {model_name}")

    model.fit(X_tr, y_tr, **({} if w_tr is None else {"sample_weight": w_tr}))
    return model.predict(X_te), y_te, test_idx[valid_te]


def run_regression_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    model_name: str = "xgb",
    metadata: Optional[List[dict]] = None,
) -> Dict:
    """
    Run cross-validated regression and return results dict.

    With per-window metadata the results also carry the fixation-window MAE
    (see compute_fixation_metrics). CFG.cv.regression_train_windows picks the
    training windows: "all", "fixation" (fixation windows only), or "weighted"
    (all windows, non-fixation ones weighted by CFG.cv.regression_nonfixation_weight).
    """
    train_windows = CFG.cv.regression_train_windows
    if train_windows not in ("all", "fixation", "weighted"):
        raise ValueError("CFG.cv.regression_train_windows must be 'all', 'fixation' or "
                         f"'weighted', got {train_windows!r}")
    if train_windows != "all" and metadata is None:
        raise ValueError(f"regression_train_windows={train_windows!r} needs per-window metadata")
    fixation = (np.array([bool(m.get("is_fixation", False)) for m in metadata])
                if metadata is not None else None)

    results = {"model": model_name, "train_windows": train_windows, "folds": []}
    if train_windows == "weighted":
        results["nonfixation_weight"] = CFG.cv.regression_nonfixation_weight
    all_preds, all_true, all_meta = [], [], []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        y_pred, y_te, kept_idx = fit_predict_regression_fold(
            X, y, train_idx, test_idx, model_name, fixation, train_windows)
        if metadata is not None:
            all_meta.extend(metadata[i] for i in kept_idx)

        results["folds"].append({
            "fold": fold_i,
            "rmse_h_deg": float(np.sqrt(mean_squared_error(y_te[:, 0], y_pred[:, 0]))),
            "rmse_v_deg": float(np.sqrt(mean_squared_error(y_te[:, 1], y_pred[:, 1]))),
            "mae_h_deg": float(mean_absolute_error(y_te[:, 0], y_pred[:, 0])),
            "mae_v_deg": float(mean_absolute_error(y_te[:, 1], y_pred[:, 1])),
            "r2_h": float(r2_score(y_te[:, 0], y_pred[:, 0])),
            "r2_v": float(r2_score(y_te[:, 1], y_pred[:, 1])),
        })
        all_preds.append(y_pred)
        all_true.append(y_te)

    all_preds = np.vstack(all_preds)
    all_true = np.vstack(all_true)
    results["pooled_rmse_h_deg"] = float(np.sqrt(mean_squared_error(all_true[:, 0], all_preds[:, 0])))
    results["pooled_rmse_v_deg"] = float(np.sqrt(mean_squared_error(all_true[:, 1], all_preds[:, 1])))
    results["pooled_mae_h_deg"] = float(mean_absolute_error(all_true[:, 0], all_preds[:, 0]))
    results["pooled_mae_v_deg"] = float(mean_absolute_error(all_true[:, 1], all_preds[:, 1]))
    results["pooled_r2_h"] = float(r2_score(all_true[:, 0], all_preds[:, 0]))
    results["pooled_r2_v"] = float(r2_score(all_true[:, 1], all_preds[:, 1]))
    if metadata is not None:
        results.update(compute_fixation_metrics(all_true, all_preds, all_meta))
    return results


def save_results(results: Dict, name: str, results_dir: str = None) -> str:
    """Save results dict as JSON. Returns file path."""
    if results_dir is None:
        results_dir = CFG.paths.results
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {path}")
    return path
