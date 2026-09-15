"""
scripts/verify_montage.py
=========================
Verify the bipolar electrode pairs and signs used in src/data/unify.py.

Correlates the difference of every electrode pair with the recorded target gaze angles
(H and V) for every subject, Fisher-z averages the per-subject correlations, and prints
the best pairs per axis next to the pair and sign in MONOPOLAR_MAP.

    python scripts/verify_montage.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loaders import load_dataset2  # noqa: E402
from src.data.unify import MONOPOLAR_MAP  # noqa: E402


def fisher_mean(corrs):
    """Average correlations via Fisher z-transform (robust to outliers)."""
    return float(np.mean(np.arctanh(np.clip(corrs, -0.999, 0.999))))


def main(top_k: int = 3) -> None:
    trials = load_dataset2()
    eog_keys = sorted([k for k in trials[0].channels if k.startswith("EOG_")], key=lambda s: int(s.split("_")[1]))
    print(f"=== Dataset 2: {len(trials)} recordings, {len(eog_keys)} electrodes ===")

    rH, rV = defaultdict(list), defaultdict(list)
    for trial in trials:
        if trial.target_angle is None:
            continue
        n = min(len(trial.channels[eog_keys[0]]), trial.target_angle.shape[0])
        tH, tV = trial.target_angle[:n, 0], trial.target_angle[:n, 1]
        for i, j in combinations(range(len(eog_keys)), 2):
            d = trial.channels[eog_keys[i]][:n] - trial.channels[eog_keys[j]][:n]
            rH[(i, j)].append(np.corrcoef(d, tH)[0, 1])
            rV[(i, j)].append(np.corrcoef(d, tV)[0, 1])

    for axis, rr in (("H", rH), ("V", rV)):
        ranked = sorted(((abs(fisher_mean(v)), fisher_mean(v), np.median(v), len(v), k) for k, v in rr.items()),
                        reverse=True)
        print(f"  Best {axis} pairs:")
        for _, mz, med, n_subj, (i, j) in ranked[:top_k]:
            print(f"    {eog_keys[i]}-{eog_keys[j]}: fisher_mean_z={mz:+.3f} median_r={med:+.3f} ({n_subj} subjects)")

    m = MONOPOLAR_MAP
    print(f"  MONOPOLAR_MAP: H = {'-' if m['flip_h'] else '+'}({m['right_key']}-{m['left_key']}), "
          f"V = {'-' if m['flip_v'] else '+'}({m['up_key']}-{m['down_key']})")
    print("A negative correlation for a pair means the map needs that pair with flip set.")


if __name__ == "__main__":
    main()
