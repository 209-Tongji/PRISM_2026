from pathlib import Path

import pytest

from sdpo_vlm.configuration import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_all_release_configs_resolve():
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        config = load_config(str(path))
        assert config["data"]["split_seed"] == 42
        assert config["lora"]["r"] == 32
        assert "_base" not in config


def test_environment_values_override_inherited_paths(monkeypatch):
    monkeypatch.setenv("PRISM_MODEL", "/models/qwen")
    monkeypatch.setenv("PRISM_SFT_ADAPTER", "/models/sft/student")
    config = load_config(str(ROOT / "configs/no_auxiliary.yaml"))
    assert config["model"]["base_path"] == "/models/qwen"
    assert config["model"]["resume_adapter"] == "/models/sft/student"
    assert config["sdpo"]["dpo_aux_alpha"] == 0
    assert config["sdpo"]["reward_weighted"] is True


def test_required_environment_and_cycles_fail_early(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text("value: ${PRISM_TEST_REQUIRED}\n")
    monkeypatch.delenv("PRISM_TEST_REQUIRED", raising=False)
    with pytest.raises(ValueError, match="PRISM_TEST_REQUIRED"):
        load_config(str(path))
    path.write_text("_base: config.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(str(path))
