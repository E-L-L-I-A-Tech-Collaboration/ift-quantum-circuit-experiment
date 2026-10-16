from __future__ import annotations

from dataclasses import asdict, replace
from statistics import mean
from typing import Iterable

from .adaptive_manifold import Config, run_experiment


def run_benchmark(config: Config | None = None, seeds: Iterable[int] = range(5)) -> dict:
    base_config = config or Config()
    runs = []

    for seed in seeds:
        result = run_experiment(replace(base_config, seed=seed))
        runs.append(
            {
                "seed": seed,
                "accepted_flips": len(result["accepted_steps"]),
                "final_objective": float(result["history_obj"][-1]),
                "final_sync": float(result["history_sync"][-1]),
            }
        )

    return {
        "config": asdict(base_config),
        "runs": runs,
        "mean_accepted_flips": mean(run["accepted_flips"] for run in runs),
        "mean_final_objective": mean(run["final_objective"] for run in runs),
        "mean_final_sync": mean(run["final_sync"] for run in runs),
    }


def run_paired_benchmark(config: Config | None = None, seeds: Iterable[int] = range(5)) -> dict:
    base_config = config or Config()
    pairs = []

    for seed in seeds:
        adaptive_config = replace(base_config, seed=seed, topology_mode="adaptive")
        fixed_config = replace(base_config, seed=seed, topology_mode="fixed")

        adaptive = run_experiment(adaptive_config)
        fixed = run_experiment(fixed_config)

        pair = {
            "seed": seed,
            "adaptive": adaptive["summary"],
            "fixed": fixed["summary"],
            "delta_final_objective": adaptive["summary"]["final_objective"]
            - fixed["summary"]["final_objective"],
            "delta_final_sync": adaptive["summary"]["final_synchronization"]
            - fixed["summary"]["final_synchronization"],
        }
        pairs.append(pair)

    return {
        "config": asdict(base_config),
        "pairs": pairs,
        "mean_delta_final_objective": mean(pair["delta_final_objective"] for pair in pairs),
        "mean_delta_final_sync": mean(pair["delta_final_sync"] for pair in pairs),
    }