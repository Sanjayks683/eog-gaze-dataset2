"""
tests/test_config.py
====================
The global config singleton and YAML overrides.
"""

from __future__ import annotations

import glob
import os

import pytest
import yaml

from src.config import CFG, Config, _CONFIG_SECTIONS, load_config_from_yaml


@pytest.fixture()
def restore_cfg():
    """Snapshot the global CFG and restore it after each test."""
    snapshot = {section: dict(vars(getattr(CFG, section))) for section in _CONFIG_SECTIONS}
    yield CFG
    for section, values in snapshot.items():
        sub = getattr(CFG, section)
        for k, v in values.items():
            setattr(sub, k, v)


def _write_yaml(tmp_path, data) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return str(path)


def test_yaml_overrides_update_global_cfg(tmp_path, restore_cfg):
    """load_config_from_yaml must mutate the GLOBAL CFG (what the pipeline reads)."""
    yaml_path = _write_yaml(tmp_path, {
        "preprocessing": {"highpass_cutoff_hz": 0.3, "baseline_window_sec": 45.0},
        "cv": {"k": 3},
        "data": {"fs_hz": 500.0},
    })

    returned = load_config_from_yaml(yaml_path)

    assert returned is CFG
    assert CFG.preprocessing.highpass_cutoff_hz == 0.3
    assert CFG.preprocessing.baseline_window_sec == 45.0
    assert CFG.cv.k == 3
    assert CFG.data.fs_hz == 500.0
    assert CFG.segmentation.window_ms == Config().segmentation.window_ms


def test_yaml_unknown_section_rejected(tmp_path, restore_cfg):
    yaml_path = _write_yaml(tmp_path, {"not_a_section": {"x": 1}})
    with pytest.raises(ValueError, match="Unknown config section"):
        load_config_from_yaml(yaml_path)


def test_yaml_unknown_key_rejected(tmp_path, restore_cfg):
    yaml_path = _write_yaml(tmp_path, {"preprocessing": {"not_a_real_key": 1}})
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config_from_yaml(yaml_path)


def test_sampling_rate_is_the_documented_256_hz():
    """Dataset 2's Data Description PDF gives 256 Hz; the PDF cannot be text-parsed."""
    assert Config().data.fs_hz == 256.0


def test_yaml_empty_file_keeps_defaults(tmp_path, restore_cfg):
    yaml_path = _write_yaml(tmp_path, {})
    before = CFG.preprocessing.lowpass_cutoff_hz
    load_config_from_yaml(yaml_path)
    assert CFG.preprocessing.lowpass_cutoff_hz == before


def test_every_shipped_config_loads(restore_cfg):
    configs = sorted(glob.glob(os.path.join(CFG.paths.project_root, "configs", "*.yaml")))
    assert len(configs) == 5
    for path in configs:
        load_config_from_yaml(path)
        assert CFG.paths.results.startswith(os.path.join(CFG.paths.project_root, "reports"))
