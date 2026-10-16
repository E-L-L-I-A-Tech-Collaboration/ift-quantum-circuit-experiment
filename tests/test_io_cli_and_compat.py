from __future__ import annotations

from dataclasses import replace
import subprocess
import sys

import numpy as np

from src.experiments.adaptive_manifold import Config, run_experiment
from src.experiments.io import reload_saved_run, save_run


def test_output_serialization_reload_roundtrip(tmp_path) -> None:
    result = run_experiment(Config(nx=4, ny=4, steps=20, seed=11, snapshot_steps=[0, 10, 20]))
    run_dir = save_run(result, tmp_path, data_provenance="classical_simulation")
    loaded = reload_saved_run(run_dir)

    assert loaded["final_state"].shape == result["state"].shape
    assert np.allclose(loaded["final_state"], result["state"])
    assert np.array_equal(loaded["initial_faces"], np.asarray(result["initial_faces"], dtype=int))


def test_snapshot_steps_generated() -> None:
    cfg = Config(nx=4, ny=4, steps=30, snapshot_steps=[0, 7, 30], seed=21)
    result = run_experiment(cfg)
    steps = [entry["step"] for entry in result["snapshots"]]
    assert steps == [0, 7, 30]


def test_short_smoke_run_no_nan_inf() -> None:
    result = run_experiment(Config(nx=4, ny=4, steps=40, seed=2))
    assert np.isfinite(result["state"]).all()
    assert np.isfinite(result["history_obj"]).all()
    assert np.isfinite(result["history_sync"]).all()


def test_root_compatibility_scripts_execute() -> None:
    first = subprocess.run(
        [sys.executable, "ift_quantum_manifold_prototype.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    second = subprocess.run(
        [sys.executable, "classiq_lorenz_step_prototype.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0
    assert second.returncode in (0, 1)
    if second.returncode != 0:
        assert "Install classiq" in (second.stderr + second.stdout)


def test_optional_quantum_imports_do_not_break_core_imports() -> None:
    import src
    import src.experiments
    import src.quantum

    assert src is not None
    assert src.experiments is not None
    assert src.quantum is not None
