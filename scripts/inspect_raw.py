"""
scripts/inspect_raw.py
=======================
Check the Dataset 2 download: load the first subject's recording, print its channels,
sampling rate and duration, and plot every channel to reports/figures/raw_trial_dataset2.png.

    python scripts/inspect_raw.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import CFG  # noqa: E402
from src.data.loaders import load_dataset2  # noqa: E402


def plot_trial(trial, save_path: str) -> None:
    """Plot all channels of a recording and save the figure."""
    ch_names = list(trial.channels.keys())
    t = np.arange(trial.n_samples()) / trial.fs
    fig, axes = plt.subplots(len(ch_names), 1, figsize=(14, 2.5 * len(ch_names)), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(f"Dataset 2  |  Subject: {trial.subject_id}  |  fs={trial.fs} Hz  |  "
                 f"Duration={trial.duration_s():.2f}s", fontsize=11)
    for ax, ch in zip(axes, ch_names):
        ax.plot(t, trial.channels[ch], lw=0.8)
        ax.set_ylabel(ch, fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path}")


def main():
    os.makedirs(CFG.paths.reports_figures, exist_ok=True)
    try:
        trials = load_dataset2(max_subjects=1)
    except FileNotFoundError as e:
        print(f"[SKIP] Dataset 2 not downloaded yet.\n{e}")
        return
    trial = trials[0]
    print(f"  fs (Hz):          {trial.fs}")
    print(f"  Channels:         {list(trial.channels)}")
    print(f"  Samples:          {trial.n_samples()}  ({trial.duration_s():.1f} s)")
    print(f"  Target angles:    {None if trial.target_angle is None else trial.target_angle.shape}")
    plot_trial(trial, os.path.join(CFG.paths.reports_figures, "raw_trial_dataset2.png"))


if __name__ == "__main__":
    main()
