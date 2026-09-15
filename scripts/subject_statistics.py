"""
scripts/subject_statistics.py
==============================
Per-subject significance of the context and range features.

For each track (real-time, offline), the nested-CV result (settings selected by fixation
MAE, see scripts/nested_cv.py) is compared with the 60 s baseline using engineered
window features only, trained on all windows, on each subject's fixation MAE: a
two-sided Wilcoxon signed-rank test, and a 95% bootstrap confidence interval of the
mean improvement (10,000 resamples of subjects).

    python scripts/subject_statistics.py

Reads reports/nested_cv/<track>/nested_cv_results.json; writes
reports/statistics/statistics.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG  # noqa: E402

REFERENCE = "window=60s features=engineered train=all"
N_BOOT = 10_000


def compare(model: dict, reference: dict, model_name: str, reference_name: str, seed: int = 0) -> dict:
    """Paired per-subject comparison of fixation MAE (lower is better)."""
    subjects = sorted(model)
    m = np.array([model[s] for s in subjects])
    r = np.array([reference[s] for s in subjects])
    rng = np.random.RandomState(seed)
    resample = rng.randint(0, len(m), size=(N_BOOT, len(m)))  # same subjects for model and reference
    boot, boot_diff = m[resample].mean(axis=1), (r - m)[resample].mean(axis=1)

    def ci(samples):
        return {"h": np.percentile(samples[:, 0], [2.5, 97.5]).round(3).tolist(),
                "v": np.percentile(samples[:, 1], [2.5, 97.5]).round(3).tolist()}

    out = {"model": model_name, "reference": reference_name, "n_subjects": len(subjects),
           "model_mean_deg": m.mean(axis=0).round(3).tolist(), "reference_mean_deg": r.mean(axis=0).round(3).tolist(),
           "model_ci95_deg": ci(boot),
           "improvement_deg": (r - m).mean(axis=0).round(3).tolist(),
           "improvement_ci95_deg": ci(boot_diff),
           "subjects_improved": {"h": int((m[:, 0] < r[:, 0]).sum()), "v": int((m[:, 1] < r[:, 1]).sum())},
           "per_subject_deg": {s: {"model": [round(float(x), 4) for x in model[s]],
                                   "reference": [round(float(x), 4) for x in reference[s]]} for s in subjects},
           "wilcoxon_p": {}}
    for axis, (a, b) in {"h": (m[:, 0], r[:, 0]), "v": (m[:, 1], r[:, 1]),
                         "mean_hv": (m.mean(axis=1), r.mean(axis=1))}.items():
        out["wilcoxon_p"][axis] = float(wilcoxon(a, b, alternative="two-sided").pvalue)
    return out


def main() -> None:
    root = CFG.paths.project_root
    results = {}
    for track in ("realtime", "offline"):
        path = os.path.join(root, "reports", "nested_cv", track, "nested_cv_results.json")
        if not os.path.isfile(path):
            print(f"missing {path}; skipping the {track} track")
            continue
        r = json.load(open(path))
        results[track] = compare(r["nested_selected"]["fixation"]["fixation_mae_per_subject"],
                                 r["fixed_candidates"][REFERENCE]["fixation_mae_per_subject"],
                                 f"nested-selected ({track})", f"{REFERENCE} ({track})")
    out_dir = os.path.join(root, "reports", "statistics")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "statistics.json"), "w") as f:
        json.dump(results, f, indent=2)
    for track, c in results.items():
        print(f"{track}: {c['model']} {c['model_mean_deg']} vs {c['reference']} {c['reference_mean_deg']} | "
              f"improvement {c['improvement_deg']} (95% CI H {c['improvement_ci95_deg']['h']}, "
              f"V {c['improvement_ci95_deg']['v']}) | Wilcoxon p H {c['wilcoxon_p']['h']:.4f}, "
              f"V {c['wilcoxon_p']['v']:.4f} | improved {c['subjects_improved']}/{c['n_subjects']}")
    print(f"Saved {os.path.join(out_dir, 'statistics.json')}")


if __name__ == "__main__":
    main()
