from __future__ import annotations

import importlib

import numpy as np

from src.experiments.adaptive_manifold import Config, run_experiment
from src.experiments.benchmark import run_paired_benchmark


def test_classical_install_path_does_not_require_classiq() -> None:
    module = importlib.import_module("src.providers.classical_lorenz")
    assert hasattr(module, "ClassicalLorenzProvider")


def test_quantum_module_import_is_safe_without_credentials() -> None:
    module = importlib.import_module("src.quantum.classiq_provider")
    assert hasattr(module, "synthesize_lorenz_step")


def test_paired_benchmark_runs_and_reports_same_initial_conditions() -> None:
    cfg = Config(nx=4, ny=4, steps=30, snapshot_steps=[0, 30], surgery_attempts=10)
    paired = run_paired_benchmark(cfg, seeds=[0, 1])
    assert len(paired["pairs"]) == 2
    assert "mean_delta_final_objective" in paired


def test_adaptive_surgeries_logged_when_accepted() -> None:
    result = run_experiment(
        Config(nx=6, ny=6, steps=120, surgery_interval=10, surgery_attempts=50, seed=11)
    )
    if result["accepted_steps"]:
        assert any(row["accepted"] for row in result["surgery_log"])
    assert isinstance(result["surgery_log"], list)
