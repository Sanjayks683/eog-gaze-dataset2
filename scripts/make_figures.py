"""
scripts/make_figures.py
=======================
Result figures from the saved JSON files (nothing is retrained):

  final_results.png            fixation MAE of the real-time and offline pipelines against
                               the baselines and the nested-CV result
  known_start.png              known-start errors against the published methods
  per_subject_fixation_mae.png each subject's fixation MAE with and without context + range features
  nested_cv_candidates.png     every nested-CV combination scored on the outer test folds

    python scripts/make_figures.py

Writes reports/figures/; a figure whose inputs are missing is skipped.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.known_start import PAPER_KNOWN_START  # noqa: E402

REPORTS = os.path.join(ROOT, "reports")
OUT_DIR = os.path.join(REPORTS, "figures")
AXES = (("h", "horizontal"), ("v", "vertical"))
TRACKS = (("realtime", "Real-time"), ("offline", "Offline"))
REFERENCE = "window=60s features=engineered train=all"


def _load(*parts):
    path = os.path.join(REPORTS, *parts)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _error_bars(means, sds) -> np.ndarray:
    """(2, n) error bars that stop at zero (an SD larger than the mean would cross it)."""
    means, sds = np.nan_to_num(np.asarray(means, dtype=float)), np.nan_to_num(np.asarray(sds, dtype=float))
    return np.vstack([np.minimum(sds, means), sds])


def _save(fig, name: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {os.path.join(OUT_DIR, name)}")


def final_results() -> None:
    series = [
        ("Always predict the mean angle", lambda t: _load(f"{t}_weighted", "baseline_reg_mean.json")),
        ("60 s baseline, engineered features only",
         lambda t: (_load("nested_cv", t, "nested_cv_results.json") or {}).get("fixed_candidates", {}).get(REFERENCE)),
        ("+ context + range, weighted training", lambda t: _load(f"{t}_weighted", "classical_reg_xgb.json")),
        ("+ context + range, fixation windows only", lambda t: _load(f"{t}_fixation", "classical_reg_xgb.json")),
        ("Nested cross-validation", lambda t: (_load("nested_cv", t, "nested_cv_results.json") or {})
         .get("nested_selected", {}).get("fixation")),
    ]
    results = {t: [get(t) for _, get in series] for t, _ in TRACKS}
    if not any(r for rows in results.values() for r in rows):
        raise FileNotFoundError("no results")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x, width = np.arange(len(TRACKS)), 0.8 / len(series)
    for ax, (key, name) in zip(axes, AXES):
        for i, (label, _) in enumerate(series):
            rows = [results[t][i] or {} for t, _ in TRACKS]
            means = [r.get(f"fixation_mae_{key}_deg", np.nan) for r in rows]
            sds = [r.get(f"fixation_mae_{key}_sd_deg", 0.0) for r in rows]
            ax.bar(x + (i - (len(series) - 1) / 2) * width, means, width, yerr=_error_bars(means, sds),
                   capsize=2, label=label)
        ax.set_xticks(x, [label for _, label in TRACKS])
        ax.set_ylabel(f"{name} fixation MAE (deg)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Dataset 2, 5-fold cross-subject (mean ± SD across subjects)")
    fig.tight_layout()
    _save(fig, "final_results.png")


def known_start() -> None:
    ours = _load("known_start", "known_start_protocol.json")
    fused = _load("known_start_fusion", "known_start_protocol.json")
    if not ours:
        raise FileNotFoundError("no known-start results")
    segments = (("short", "short 1-2 s"), ("long", "long 32 s"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = ("This pipeline: detected saccades", "Fused with cross-subject XGBoost",
              "Published: dual Kalman filter", "Published: signal differencing")
    x, width = np.arange(len(segments)), 0.2
    for ax, (key, name) in zip(axes, AXES):
        values = {label: [] for label in labels}
        sds = {label: [] for label in labels}
        for length, segment in segments:
            for label, source, estimator in ((labels[0], ours, "saccades"), (labels[1], fused, "fused")):
                r = (source or {}).get("same_subject", {}).get(f"{length}_{estimator}")
                values[label].append(r[f"excluded_mae_{key}_deg"] if r else np.nan)
                sds[label].append(r[f"excluded_mae_{key}_sd_deg"] if r else 0.0)
            for label, kind in ((labels[2], "Kalman"), (labels[3], "differencing")):
                row = next(p for p in PAPER_KNOWN_START if p[1] == segment and kind in p[0])
                values[label].append(row[2] if key == "h" else row[3])
                sds[label].append(0.0)
        for i, label in enumerate(labels):
            ax.bar(x + (i - 1.5) * width, values[label], width, yerr=_error_bars(values[label], sds[label]),
                   capsize=2, label=label, hatch="//" if label.startswith("Published") else None)
        ax.set_xticks(x, [segment for _, segment in segments])
        ax.set_ylabel(f"{name} fixation MAE (deg)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Known-start task, same subject, outlier segments dropped")
    fig.tight_layout()
    _save(fig, "known_start.png")


def per_subject_fixation_mae() -> None:
    stats = _load("statistics", "statistics.json")
    panels = [(t, label, stats[t]) for t, label in TRACKS if stats and t in stats]
    if not panels:
        raise FileNotFoundError("no statistics.json")
    fig, axes = plt.subplots(2, len(panels), figsize=(3.5 * len(panels), 6.5), squeeze=False)
    for j, (track, track_label, comparison) in enumerate(panels):
        per_subject = comparison["per_subject_deg"]
        for i, (axis, name) in enumerate(AXES):
            ax = axes[i, j]
            reference = np.array([per_subject[s]["reference"][i] for s in per_subject])
            model = np.array([per_subject[s]["model"][i] for s in per_subject])
            for a, b in zip(reference, model):
                ax.plot([0, 1], [a, b], marker="o", ms=3, alpha=0.7, color="tab:green" if b < a else "tab:red")
            ax.plot([0, 1], [reference.mean(), model.mean()], marker="s", lw=2.5, color="black")
            ax.set_xticks([0, 1], ["engineered\nfeatures only", "nested CV"])
            ax.set_xlim(-0.3, 1.3)
            ax.text(0.5, 0.98, _p(comparison["wilcoxon_p"][axis]), transform=ax.transAxes,
                    ha="center", va="top", fontsize=8)
            if i == 0:
                ax.set_title(track_label, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{name} fixation MAE (deg)")
    fig.suptitle("Per-subject fixation MAE (green = improved, black = mean, Wilcoxon signed-rank p)")
    fig.tight_layout()
    _save(fig, "per_subject_fixation_mae.png")


def nested_cv_candidates() -> None:
    tracks = [(t, label, _load("nested_cv", t, "nested_cv_results.json")) for t, label in TRACKS]
    tracks = [(t, label, r) for t, label, r in tracks if r]
    if not tracks:
        raise FileNotFoundError("no nested-CV results")
    windows = ("30", "60", "120")
    rows = [(f, tw) for f in ("engineered", "context", "range") for tw in ("all", "weighted", "fixation")]
    name = lambda w, f, tw: f"window={w}s features={f} train={tw}"
    fig, axes = plt.subplots(1, len(tracks), figsize=(6.5 * len(tracks), 5), squeeze=False)
    for ax, (track, label, results) in zip(axes[0], tracks):
        grid = np.array([[results["fixed_candidates"][name(w, f, tw)]["score_fixation"] for w in windows]
                         for f, tw in rows])
        picks = Counter(fold["selected"]["fixation"] for fold in results["folds"])
        image = ax.imshow(grid, cmap="viridis_r", aspect="auto")
        for i, (f, tw) in enumerate(rows):
            for j, w in enumerate(windows):
                n = picks.get(name(w, f, tw), 0)
                ax.text(j, i, f"{grid[i, j]:.2f}" + (f"\npicked {n}/{len(results['folds'])}" if n else ""),
                        ha="center", va="center", fontsize=7, color="white" if grid[i, j] > grid.mean() else "black")
        ax.set_xticks(range(len(windows)), [f"{w} s" for w in windows])
        ax.set_xlabel("drift-baseline window")
        ax.set_yticks(range(len(rows)), [f"{'+ context + range' if f == 'range' else '+ context' if f == 'context' else f}, "
                                         f"{tw} windows" for f, tw in rows])
        ax.set_title(f"{label}: mean H/V fixation MAE (deg)", fontsize=10)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle("Nested cross-validation: every combination scored on the outer test folds, "
                 "and how often the inner loop picked it")
    fig.tight_layout()
    _save(fig, "nested_cv_candidates.png")


def main() -> None:
    for figure in (final_results, known_start, per_subject_fixation_mae, nested_cv_candidates):
        try:
            figure()
        except (FileNotFoundError, KeyError, StopIteration) as e:
            print(f"skipped {figure.__name__}: {e}")


if __name__ == "__main__":
    main()
