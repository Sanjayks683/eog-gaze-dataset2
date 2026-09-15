"""
src/config.py
=============
Single source of truth for every tunable parameter of the Dataset 2 pipeline.
The YAML files in configs/ override these defaults (load_config_from_yaml);
nothing downstream should hard-code a value that belongs here.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml
import os


@dataclass
class PreprocessingConfig:
    highpass_cutoff_hz: float = 0.2
    lowpass_cutoff_hz: float = 30.0
    filter_order: int = 4
    notch_hz: Optional[float] = 50.0
    notch_quality_factor: float = 30.0
    blink_threshold_std: float = 4.5
    blink_min_duration_ms: float = 40.0
    # Drift removal: "robust_line" (real-time configs, past-only) and "robust_mean"
    # (offline configs, centred) are the ones the results use; "highpass",
    # "polynomial_detrend" and "moving_median" are also implemented.
    drift_removal_method: str = "highpass"
    polynomial_detrend_order: int = 2
    median_baseline_sec: float = 30.0
    median_baseline_causal: bool = False
    baseline_window_sec: float = 30.0
    baseline_causal: bool = False
    baseline_clip_mad: float = 3.0
    # Context features for the regressor (empty = off): window means of other baseline
    # estimates minus the main one, as [method, window_sec] pairs (same causal setting
    # as the main baseline), and of the main signal lagged by context_lags_sec seconds.
    context_baselines: List[list] = field(default_factory=list)
    context_lags_sec: List[float] = field(default_factory=list)
    # Rolling-quantile range features over these windows (empty = off), one set per
    # [low, high] quantile pair; see src/features/context.py.
    context_range_windows_sec: List[float] = field(default_factory=list)
    context_range_quantiles: List[list] = field(default_factory=lambda: [[0.10, 0.90], [0.05, 0.95]])


@dataclass
class SegmentationConfig:
    window_ms: float = 300.0
    stride_ms: float = 150.0
    # A window is a "fixation" window when it lies inside one ControlSignal 1/2
    # interval and starts this long after the interval (the cue) began. Saccades on
    # Dataset 2 peak ~200 ms and settle by ~400 ms after the cue.
    fixation_settle_ms: float = 400.0


@dataclass
class CVConfig:
    strategy: str = "group_kfold"
    k: int = 5
    random_seed: int = 42
    # XGBoost device: "cpu" (the committed results) or "cuda". GPU training is much
    # faster but its results differ slightly from CPU ones.
    xgb_device: str = "cpu"
    # Which training windows the regressor is fit on: "all", "fixation" (fixation
    # windows only) or "weighted" (all windows, non-fixation ones weighted by
    # regression_nonfixation_weight). Test windows are never filtered.
    regression_train_windows: str = "all"
    regression_nonfixation_weight: float = 0.2


@dataclass
class DataConfig:
    # Sampling rate from the Dataset 2 Data Description PDF (g.tec g.USBamp).
    fs_hz: float = 256.0
    # "bipolar" -> windows hold H and V; "all_eog" -> H, V and the four monopolar channels.
    window_channel_mode: str = "bipolar"


class PathConfig:
    """
    Filesystem layout. `folds_file` is a property derived from `data_processed`
    so that overriding data_processed (YAML/tests) always redirects the folds file
    with it.
    """

    def __init__(self):
        self.project_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.data_raw: str = os.path.join(self.project_root, "data", "raw")
        self.dataset_dir: str = os.path.join(self.data_raw, "Dataset_Stationary")
        self.data_processed: str = os.path.join(self.project_root, "data", "processed")
        self.reports_figures: str = os.path.join(self.project_root, "reports", "figures")
        self.results: str = os.path.join(self.project_root, "reports")

    @property
    def folds_file(self) -> str:
        return os.path.join(self.data_processed, "cv_folds.pkl")


@dataclass
class KnownStartConfig:
    """Separate known-start evaluation (src/evaluation/known_start.py)."""
    lowpass_hz: float = 20.0
    speed_onset_sd: float = 8.0       # event must reach this speed (robust velocity SDs)
    speed_offset_sd: float = 4.0      # event extends while speed stays above this
    merge_ms: float = 150.0           # merge events closer than this (blink strokes, corrective saccades)
    min_event_ms: float = 8.0
    level_window_ms: float = 100.0    # median level window before / after an event
    after_gap_ms: float = 50.0        # skip this long after an event before measuring its end level
    # Blink = vertical excursion > blink_excursion_ratio x net vertical displacement AND the
    # returned part > blink_return_scale x median |vertical displacement| of events shorter
    # than saccade_max_ms. On Dataset 2 this flags 90% of blink-interval events and 1% of
    # cue-driven saccades (blinks last ~190-400 ms, saccades ~50-110 ms).
    blink_excursion_ratio: float = 3.0
    blink_return_scale: float = 1.5
    saccade_max_ms: float = 150.0
    blink_pair_ms: float = 500.0      # two events this close whose vertical displacements cancel = blink
    blink_pair_residual: float = 0.3
    # Ground-truth labels (paper App. E.1), from sample-to-sample EOG differences
    gt_fixation_sd: float = 5.0       # |difference| above this many robust SDs on either channel = moving
    gt_merge_ms: float = 20.0         # fixation gaps shorter than this inside a movement count as movement
    gt_min_saccade_ms: float = 10.0   # shorter movement runs are ignored when judging subject mistakes
    gt_blink_ratio: float = 0.5       # blink spike threshold = this x median largest |V difference| per blink interval
    gt_blink_second_ratio: float = 0.5  # the opposite closing spike must exceed this x the threshold
    gt_blink_max_ms: float = 500.0
    # Windows and subject mistakes (paper Sec. 4.5.3)
    response_min_ms: float = 80.0     # response saccade must start this long after the cue ...
    response_max_ms: float = 600.0    # ... and no later than this
    premature_ms: float = 150.0       # a saccade starting this close to the window end is premature
    blink_margin_ms: float = 150.0    # saccade-labelled strokes this close to a labelled blink belong to it
    min_scored_ms: float = 50.0       # windows need at least this many fixation samples to be scored
    # Outlier segments: H or V error beyond Q3 + outlier_iqr_factor x IQR (Tukey's far-out
    # fence), per subject and segment kind. The paper only says "substantially high" error.
    outlier_iqr_factor: float = 3.0
    n_subsets: int = 3
    short_saccade_windows: int = 66   # per subset, as in the paper
    short_blink_windows: int = 33
    long_segments: int = 8            # per subset, each of long_trials consecutive 4 s trials
    long_trials: int = 8


@dataclass
class Config:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    known_start: KnownStartConfig = field(default_factory=KnownStartConfig)


CFG = Config()

_CONFIG_SECTIONS = ("preprocessing", "segmentation", "cv", "data", "paths", "known_start")


def window_samples(fs: float) -> int:
    """Convert window_ms to integer sample count given sampling rate fs."""
    return int(round(CFG.segmentation.window_ms * fs / 1000.0))


def stride_samples(fs: float) -> int:
    """Convert stride_ms to integer sample count given sampling rate fs."""
    return int(round(CFG.segmentation.stride_ms * fs / 1000.0))


def blink_min_samples(fs: float) -> int:
    """Convert blink_min_duration_ms to integer sample count given fs."""
    return int(round(CFG.preprocessing.blink_min_duration_ms * fs / 1000.0))


def load_config_from_yaml(yaml_path: str) -> Config:
    """
    Override the GLOBAL CFG defaults from a YAML file (in place) and return it.
    Only keys present in the YAML override; all others remain at their current
    value. Every downstream module reads the global CFG, so mutating it here is
    what actually applies the overrides.
    """
    with open(yaml_path, "r") as f:
        overrides = yaml.safe_load(f) or {}

    if not isinstance(overrides, dict):
        raise ValueError(f"Config file {yaml_path!r} must contain a YAML mapping.")

    for section, values in overrides.items():
        if section not in _CONFIG_SECTIONS:
            raise ValueError(
                f"Unknown config section {section!r} in {yaml_path}. "
                f"Valid sections: {', '.join(_CONFIG_SECTIONS)}."
            )
        if not isinstance(values, dict):
            raise ValueError(f"Section {section!r} in {yaml_path} must be a mapping.")
        sub_cfg = getattr(CFG, section)
        for k, v in values.items():
            if not hasattr(sub_cfg, k):
                raise ValueError(
                    f"Unknown config key {section}.{k} in {yaml_path}. "
                    "Check src/config.py for valid field names."
                )
            if section == "paths" and isinstance(v, str) and not os.path.isabs(v):
                v = os.path.normpath(os.path.join(CFG.paths.project_root, v))
            setattr(sub_cfg, k, v)
    return CFG
