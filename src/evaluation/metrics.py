"""
src/evaluation/metrics.py
==========================
Gaze-error metrics.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """RMSE, MAE and R² for H and V separately, and the combined RMSE."""
    return {
        "rmse_h_deg": float(np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))),
        "rmse_v_deg": float(np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))),
        "mae_h_deg": float(mean_absolute_error(y_true[:, 0], y_pred[:, 0])),
        "mae_v_deg": float(mean_absolute_error(y_true[:, 1], y_pred[:, 1])),
        "r2_h": float(r2_score(y_true[:, 0], y_pred[:, 0])),
        "r2_v": float(r2_score(y_true[:, 1], y_pred[:, 1])),
        "rmse_combined_deg": float(np.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))),
    }


def compute_fixation_metrics(y_true: np.ndarray, y_pred: np.ndarray, metadata: List[dict]) -> Dict:
    """
    Gaze error scored the way Barbara et al. (BSPC 2023) report it: mean absolute
    error over fixation windows only (metadata "is_fixation"), computed per subject
    and then averaged across subjects (± SD across subjects).

    Returns an empty dict when the metadata has no fixation flags or no fixations.
    """
    fixation = np.array([bool(m.get("is_fixation", False)) for m in metadata])
    if len(fixation) != len(y_true) or not fixation.any():
        return {}
    subjects = np.array([m["subject_id"] for m in metadata])
    abs_err = np.abs(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float))
    subject_ids = np.unique(subjects[fixation])
    per_subject = np.array([abs_err[fixation & (subjects == s)].mean(axis=0) for s in subject_ids])
    return {
        "fixation_mae_h_deg": float(per_subject[:, 0].mean()),
        "fixation_mae_v_deg": float(per_subject[:, 1].mean()),
        "fixation_mae_h_sd_deg": float(per_subject[:, 0].std()),
        "fixation_mae_v_sd_deg": float(per_subject[:, 1].std()),
        "n_fixation_windows": int(fixation.sum()),
        "fixation_mae_per_subject": {str(s): [float(h), float(v)] for s, (h, v) in zip(subject_ids, per_subject)},
    }
