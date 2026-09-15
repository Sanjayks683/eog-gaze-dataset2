"""
main.py
=======
Command-line runner for the Dataset 2 gaze-estimation pipeline.

    python main.py --config configs/realtime_weighted.yaml --phase all
    python main.py --config configs/realtime_weighted.yaml --phase preprocess
    python main.py --config configs/realtime_weighted.yaml --phase train
    python main.py --config configs/realtime_weighted.yaml --phase compare
    python main.py --config configs/known_start.yaml --phase known_start
    python main.py --phase inspect

Phases:
  inspect      plot one raw recording and print its channels (checks the download)
  preprocess   load, bipolarise, remove drift, filter, z-score, window, context features, CV folds
  train        5-fold cross-subject XGBoost and the mean-angle baseline; results to JSON
  compare      results table with the published Dataset 2 numbers
  known_start  the published known-start protocol (a separate, easier task)
  all          preprocess, train and compare
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CFG, load_config_from_yaml


def run_inspect():
    from scripts.inspect_raw import main as inspect_main
    inspect_main()


def run_preprocess():
    print("\n" + "=" * 60)
    print("  Preprocess: load, filter, window, context features, folds")
    print("=" * 60)
    from src.data.datasets import make_regression_targets, save_processed
    from src.data.loaders import load_trials
    from src.data.unify import to_common_schema
    from src.features.context import make_context_features
    from src.preprocessing.normalize import preprocess_trials
    from src.training.cv_splits import get_folds, get_subject_ids_from_metadata, save_folds

    print("[1/3] Loading Dataset 2 and forming bipolar H / V ...")
    trials = [to_common_schema(t) for t in load_trials()]
    use_context = bool(CFG.preprocessing.context_baselines or CFG.preprocessing.context_lags_sec)
    raw_signals = [{k: v.copy() for k, v in t.channels.items()} for t in trials] if use_context else None

    print("[2/3] Drift removal, filtering and per-subject z-scoring ...")
    trials = preprocess_trials(trials)

    print("[3/3] Windowing and saving processed arrays ...")
    X, y, meta = make_regression_targets(trials)
    save_processed(X, y, meta, "regression")
    if use_context:
        save_processed(make_context_features(trials, raw_signals, meta), y, meta, "context")
    save_folds(get_folds(get_subject_ids_from_metadata(meta)), meta)
    print("Preprocessing completed successfully.")


def run_train():
    print("\n" + "=" * 60)
    print("  Train: 5-fold cross-subject regression")
    print("=" * 60)
    import numpy as np
    from src.data.datasets import load_processed
    from src.features.engineered import extract_features_batch
    from src.models.classical_ml import run_regression_cv, save_results
    from src.training.cv_splits import load_folds

    X, y, meta = load_processed("regression")
    folds, _ = load_folds()
    fs_values = sorted({m["fs"] for m in meta})
    if len(fs_values) != 1:
        raise ValueError(f"Expected one sampling rate, got {fs_values}")

    print("Extracting engineered window features ...")
    features = extract_features_batch(X, fs=fs_values[0], channel_names=meta[0]["channel_names"])
    if CFG.preprocessing.context_baselines or CFG.preprocessing.context_lags_sec:
        X_ctx, _, _ = load_processed("context")
        print(f"Adding {X_ctx.shape[1]} context features")
        features = np.hstack([features, X_ctx])

    print("\nBaseline: train-fold mean angle")
    save_results(run_regression_cv(features, y, folds, model_name="mean", metadata=meta), "baseline_reg_mean")
    print(f"\nXGBoost (training windows: {CFG.cv.regression_train_windows})")
    save_results(run_regression_cv(features, y, folds, model_name="xgb", metadata=meta), "classical_reg_xgb")


def run_compare():
    from src.evaluation.compare_to_paper import run_comparison
    run_comparison()


def run_known_start():
    print("\n" + "=" * 60)
    print("  Known-start protocol (separate from the main results)")
    print("=" * 60)
    from src.data.loaders import load_trials
    from src.data.unify import to_common_schema
    from src.evaluation.known_start import run_known_start as evaluate

    evaluate([to_common_schema(t) for t in load_trials()])


def main():
    parser = argparse.ArgumentParser(description="EOG gaze estimation on EyeCon Dataset 2")
    parser.add_argument("--phase", choices=["inspect", "preprocess", "train", "compare", "known_start", "all"],
                        default="all", help="Pipeline phase to run")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML config in configs/")
    args = parser.parse_args()

    if args.config:
        load_config_from_yaml(args.config)

    if args.phase == "inspect":
        run_inspect()
    elif args.phase == "preprocess":
        run_preprocess()
    elif args.phase == "train":
        run_train()
    elif args.phase == "compare":
        run_compare()
    elif args.phase == "known_start":
        run_known_start()
    elif args.phase == "all":
        try:
            run_preprocess()
            run_train()
            run_compare()
        except FileNotFoundError as e:
            print(f"\n[INFO] Pipeline stopped: raw data missing. See the download instructions in README.md.\n  {e}")


if __name__ == "__main__":
    main()
