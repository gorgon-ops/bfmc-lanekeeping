"""ruff does not check that an import still resolves, so a package rename can leave a script dead.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parent.parent / "scripts").glob("*.py"))


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_there_are_scripts_to_check():
    assert SCRIPTS, "no scripts found; has the directory moved?"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_script_imports(path):
    load(path)


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.stem)
def test_script_parses_its_own_help(path):
    module = load(path)
    with pytest.raises(SystemExit) as exit_info:
        module.build_parser().parse_args(["--help"])
    assert exit_info.value.code == 0


def test_calibrate_warp_emits_a_loadable_config(tmp_path):
    from bfmc.config import AppConfig

    module = load(Path(__file__).resolve().parent.parent / "scripts" / "calibrate_warp.py")
    block = module.as_yaml(AppConfig().warp, 318)

    path = tmp_path / "calibrated.yaml"
    path.write_text(block, encoding="utf-8")

    config = AppConfig.load(path)
    assert config.tracking.lane_width_px == 318


def test_calibrate_warp_handles_never_seeing_both_lanes(tmp_path):
    from bfmc.config import AppConfig

    module = load(Path(__file__).resolve().parent.parent / "scripts" / "calibrate_warp.py")
    block = module.as_yaml(AppConfig().warp, None)

    path = tmp_path / "partial.yaml"
    path.write_text(block, encoding="utf-8")
    assert AppConfig.load(path) is not None


def test_simulate_knows_its_tracks():
    module = load(Path(__file__).resolve().parent.parent / "scripts" / "simulate.py")
    assert {"oval", "tight", "hairpin", "figure8"} <= set(module.TRACKS)
    for factory in module.TRACKS.values():
        assert len(factory().points) > 10


def test_simulate_comparisons_use_real_config_keys():
    from bfmc.config import AppConfig

    module = load(Path(__file__).resolve().parent.parent / "scripts" / "simulate.py")
    for overrides in module.COMPARISONS.values():
        config = AppConfig()
        for key, value in overrides.items():
            config.set(key, value)   # raises KeyError on a stale key


def test_simulate_runs_a_short_lap(tmp_path):
    module = load(Path(__file__).resolve().parent.parent / "scripts" / "simulate.py")
    assert module.main(["--steps", "25"]) == 0
