"""
src/evaluation/compare_to_paper.py
====================================
Results table for one config: the trivial mean-angle baseline, XGBoost, and the
published Dataset 2 results of Barbara et al. (BSPC 86, 2023).

Saved as master_results_table.csv in the config's results folder. Units: degrees.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

from src.config import CFG

# Barbara et al., BSPC vol.86 (2023), Tables 2-3. The paper reports MAE over fixation
# samples only (Appendix E.2), mean ± SD across subjects, with every model parameter
# fitted on the SAME subject (3 contiguous subsets: fit / tune / test, all 6 role
# permutations), each segment starting from the known gaze, and outlier segments
# excluded (6.85% short, 5.83% long). There is no RMSE in the paper.
PAPER_RESULTS = {
    "Barbara_2023_DKF_short": {
        "description": "Multiple-model dual Kalman filter, 1 s saccade / 2 s blink segments",
        "fixation_mae_h_deg": 1.64, "fixation_mae_v_deg": 1.97,
        "notes": "BSPC 86 (2023) Table 2: ±0.82 / ±0.34; within-subject; known start; 6.85% outliers excluded",
    },
    "Barbara_2023_DKF_long": {
        "description": "Multiple-model dual Kalman filter, 32 s segments (8 trials)",
        "fixation_mae_h_deg": 5.23, "fixation_mae_v_deg": 6.59,
        "notes": "BSPC 86 (2023) Table 3: ±2.00 / ±3.10; within-subject; known start; 5.83% outliers excluded",
    },
    "Barbara_2019_differencing_short": {
        "description": "Signal differencing + 2-channel linear regression [BSPC 47, 2019], as run in BSPC 86",
        "fixation_mae_h_deg": 1.51, "fixation_mae_v_deg": 1.95,
        "notes": "BSPC 86 (2023) Table 2 state of the art: ±0.55 / ±0.29; same protocol as DKF short",
    },
    "Barbara_2019_differencing_long": {
        "description": "Signal differencing + 2-channel linear regression [BSPC 47, 2019], as run in BSPC 86",
        "fixation_mae_h_deg": 5.82, "fixation_mae_v_deg": 8.04,
        "notes": "BSPC 86 (2023) Table 3 state of the art: ±2.70 / ±2.96; same protocol as DKF long",
    },
}

_ERROR_KEYS = ("rmse_h_deg", "rmse_v_deg", "mae_h_deg", "mae_v_deg",
               "fixation_mae_h_deg", "fixation_mae_v_deg")


def load_result_safe(name: str, results_dir: str) -> Optional[Dict]:
    path = os.path.join(results_dir, f"{name}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _error_columns(d: Optional[Dict]) -> Dict:
    """Error columns of a result JSON (all-window pooled RMSE/MAE, fixation MAE)."""
    d = d or {}
    return {
        "rmse_h_deg": d.get("pooled_rmse_h_deg"), "rmse_v_deg": d.get("pooled_rmse_v_deg"),
        "mae_h_deg": d.get("pooled_mae_h_deg"), "mae_v_deg": d.get("pooled_mae_v_deg"),
        "fixation_mae_h_deg": d.get("fixation_mae_h_deg"),
        "fixation_mae_v_deg": d.get("fixation_mae_v_deg"),
    }


def build_master_table(results_dir: str = None) -> List[Dict]:
    """
    Rows with keys: method, rmse_h_deg, rmse_v_deg, mae_h_deg, mae_v_deg,
    fixation_mae_h_deg, fixation_mae_v_deg, notes. RMSE/MAE are pooled over all test
    windows; fixation MAE is the per-subject mean over fixation windows, the metric
    the published results use.
    """
    if results_dir is None:
        results_dir = CFG.paths.results

    rows = []
    d = load_result_safe("baseline_reg_mean", results_dir)
    if d:
        rows.append({"method": "Baseline: train-fold mean angle", **_error_columns(d),
                     "notes": "Always predicts the training folds' mean H/V angle"})

    d = load_result_safe("classical_reg_xgb", results_dir)
    if d:
        train_windows = d.get("train_windows", "all")
        notes = {
            "all": "trained on all windows",
            "fixation": "trained on fixation windows only",
            "weighted": f"trained on all windows, non-fixation windows weighted {d.get('nonfixation_weight')}",
        }.get(train_windows, f"trained on {train_windows} windows")
        rows.append({"method": "XGBoost", **_error_columns(d), "notes": notes})

    for paper_key, paper_data in PAPER_RESULTS.items():
        rows.append({
            "method": f"Published: {paper_key}",
            **{k: paper_data.get(k) for k in _ERROR_KEYS},
            "notes": paper_data["notes"],
        })
    return rows


def print_master_table(rows: List[Dict]) -> None:
    """Pretty-print the results table."""
    print("\n" + "=" * 110)
    print(f"{'Method':<40} {'RMSE H°':>8} {'RMSE V°':>8} {'MAE H°':>8} {'MAE V°':>8} {'FixMAE H°':>9} {'FixMAE V°':>9}")
    print("=" * 110)
    for row in rows:
        fmt = lambda v: f"{v:.3f}" if v is not None else "  N/A "
        print(f"{row['method']:<40} {fmt(row['rmse_h_deg']):>8} {fmt(row['rmse_v_deg']):>8} "
              f"{fmt(row['mae_h_deg']):>8} {fmt(row['mae_v_deg']):>8} "
              f"{fmt(row['fixation_mae_h_deg']):>9} {fmt(row['fixation_mae_v_deg']):>9}")
        if row.get("notes"):
            print(f"  -> {row['notes']}")
    print("=" * 110)


def save_master_table_csv(rows: List[Dict], results_dir: str = None) -> str:
    """Save the results table as CSV."""
    if results_dir is None:
        results_dir = CFG.paths.results
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "master_results_table.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", *_ERROR_KEYS, "notes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results table saved: {path}")
    return path


def run_comparison(results_dir: str = None) -> None:
    """Build, print and save the results table."""
    rows = build_master_table(results_dir)
    print_master_table(rows)
    save_master_table_csv(rows, results_dir)
