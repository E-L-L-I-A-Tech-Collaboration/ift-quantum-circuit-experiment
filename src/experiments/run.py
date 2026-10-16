from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from src.experiments.adaptive_manifold import Config, run_experiment
from src.experiments.benchmark import run_paired_benchmark
from src.experiments.io import config_hash, save_run, write_json
from src.providers import ClassicalLorenzProvider, ClassiqStateProvider, RecordedStateProvider
from src.visualization.plotting import plot_mesh, plot_series


def _load_config(path: str | None) -> Config:
    if path is None:
        return Config()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Config(**payload)


def _resolve_provider(name: str, recorded_path: str | None):
    lowered = name.lower()
    if lowered == "classical":
        return ClassicalLorenzProvider(), "classical_simulation"
    if lowered == "recorded":
        if not recorded_path:
            raise ValueError("--recorded-path is required with --provider recorded")
        return RecordedStateProvider(recorded_path), "recorded_replay"
    if lowered == "classiq":
        return ClassiqStateProvider(), "quantum_local_arithmetic_prototype"
    raise ValueError(f"Unsupported provider: {name}")


def _save_single_run_figures(result: dict, run_dir: Path, prefix: str = "") -> None:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Keep color scale consistent across mesh snapshots.
    vmin = float(np.min(result["state"][:, 0]))
    vmax = float(np.max(result["state"][:, 0]))

    fig = plot_mesh(
        result["points"],
        result["initial_faces"],
        result["initial_state"][:, 0],
        "Initial manifold",
    )
    fig.axes[0].collections[0].set_clim(vmin=vmin, vmax=vmax)
    fig.savefig(fig_dir / f"{prefix}initial_manifold.png", dpi=180)
    plt.close(fig)

    fig = plot_mesh(
        result["points"],
        result["faces"],
        result["state"][:, 0],
        "Evolved manifold",
    )
    fig.axes[0].collections[0].set_clim(vmin=vmin, vmax=vmax)
    fig.savefig(fig_dir / f"{prefix}evolved_manifold.png", dpi=180)
    plt.close(fig)

    fig = plot_series(
        result["history_t"],
        result["history_obj"],
        "time",
        "objective",
        "Objective history",
        str(fig_dir / f"{prefix}objective_history.png"),
    )
    plt.close(fig)

    fig = plot_series(
        result["history_t"],
        result["history_sync"],
        "time",
        "synchronization",
        "Synchronization history",
        str(fig_dir / f"{prefix}synchronization_history.png"),
    )
    plt.close(fig)

    accepted = [row for row in result["surgery_log"] if row["accepted"]]
    if accepted:
        steps = [row["step"] for row in accepted]
        deltas = [row["objective_delta"] for row in accepted]
        plt.figure(figsize=(7, 4))
        plt.plot(steps, np.cumsum(np.ones(len(steps))), marker="o")
        plt.xlabel("step")
        plt.ylabel("cumulative accepted surgeries")
        plt.title("Accepted surgeries over time")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{prefix}accepted_surgeries_over_time.png", dpi=180)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.bar(range(len(deltas)), deltas)
        plt.xlabel("accepted surgery index")
        plt.ylabel("objective delta")
        plt.title("Objective delta per accepted surgery")
        plt.tight_layout()
        plt.savefig(fig_dir / f"{prefix}objective_delta_per_accepted_surgery.png", dpi=180)
        plt.close()

    snapshots = result.get("snapshots", [])
    if snapshots:
        ncols = min(3, len(snapshots))
        nrows = int(np.ceil(len(snapshots) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        for index, snapshot in enumerate(snapshots):
            r, c = divmod(index, ncols)
            ax = axes[r][c]
            faces = np.asarray(snapshot["faces"], dtype=int)
            state = np.asarray(snapshot["state"], dtype=float)
            mesh = ax.tripcolor(
                result["points"][:, 0],
                result["points"][:, 1],
                faces,
                state[:, 0],
                shading="gouraud",
                vmin=vmin,
                vmax=vmax,
            )
            ax.triplot(result["points"][:, 0], result["points"][:, 1], faces, linewidth=0.35)
            ax.set_aspect("equal")
            ax.set_title(f"Step {snapshot['step']}")
        for index in range(len(snapshots), nrows * ncols):
            r, c = divmod(index, ncols)
            axes[r][c].axis("off")
        fig.colorbar(mesh, ax=axes.ravel().tolist(), label="x-state")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{prefix}manifold_snapshots.png", dpi=180)
        plt.close(fig)


def _plot_adaptive_vs_fixed(adaptive_result: dict, fixed_result: dict, run_dir: Path) -> None:
    fig_dir = run_dir / "figures"

    plt.figure(figsize=(7, 4))
    plt.plot(adaptive_result["history_t"], adaptive_result["history_obj"], label="adaptive")
    plt.plot(fixed_result["history_t"], fixed_result["history_obj"], label="fixed")
    plt.xlabel("time")
    plt.ylabel("objective")
    plt.title("Adaptive vs fixed objective")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "adaptive_vs_fixed_objective.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(adaptive_result["history_t"], adaptive_result["history_sync"], label="adaptive")
    plt.plot(fixed_result["history_t"], fixed_result["history_sync"], label="fixed")
    plt.xlabel("time")
    plt.ylabel("synchronization")
    plt.title("Adaptive vs fixed synchronization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "adaptive_vs_fixed_synchronization.png", dpi=180)
    plt.close()

    labels = ["final_objective", "final_synchronization", "accepted_surgeries"]
    adaptive_values = [adaptive_result["summary"][label] for label in labels]
    fixed_values = [fixed_result["summary"][label] for label in labels]

    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, adaptive_values, width, label="adaptive")
    plt.bar(x + width / 2, fixed_values, width, label="fixed")
    plt.xticks(x, labels)
    plt.title("Adaptive vs fixed final metric summary")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "adaptive_vs_fixed_final_metric_summary.png", dpi=180)
    plt.close()


def _run_single(config: Config, provider_name: str, provider, output_root: Path) -> Path:
    del provider_name
    result = run_experiment(config, provider=provider)
    run_dir = save_run(result, output_root, data_provenance=result["provider_metadata"].get("mode", "classical_simulation"))
    _save_single_run_figures(result, run_dir)
    return run_dir


def _paired_runs(config: Config, provider_name: str, provider, output_root: Path, seeds: Iterable[int]) -> Path:
    del provider_name
    benchmark_root = output_root / "paired"
    benchmark_root.mkdir(parents=True, exist_ok=True)

    pairs_summary = []
    for seed in seeds:
        adaptive_result = run_experiment(replace(config, seed=seed, topology_mode="adaptive"), provider=provider)
        fixed_result = run_experiment(replace(config, seed=seed, topology_mode="fixed"), provider=provider)

        adaptive_suffix = config_hash(adaptive_result["config"])
        fixed_suffix = config_hash(fixed_result["config"])

        adaptive_run_dir = save_run(
            adaptive_result,
            benchmark_root,
            run_id=f"seed_{seed}_adaptive_{adaptive_suffix}",
            data_provenance=adaptive_result["provider_metadata"].get("mode", "classical_simulation"),
        )
        fixed_run_dir = save_run(
            fixed_result,
            benchmark_root,
            run_id=f"seed_{seed}_fixed_{fixed_suffix}",
            data_provenance=fixed_result["provider_metadata"].get("mode", "classical_simulation"),
        )

        _save_single_run_figures(adaptive_result, adaptive_run_dir)
        _save_single_run_figures(fixed_result, fixed_run_dir)
        _plot_adaptive_vs_fixed(adaptive_result, fixed_result, adaptive_run_dir)

        same_initial_faces = np.array_equal(
            np.asarray(adaptive_result["initial_faces"], dtype=int),
            np.asarray(fixed_result["initial_faces"], dtype=int),
        )
        same_initial_state = np.allclose(adaptive_result["initial_state"], fixed_result["initial_state"])

        pairs_summary.append(
            {
                "seed": seed,
                "adaptive_run": str(adaptive_run_dir),
                "fixed_run": str(fixed_run_dir),
                "same_initial_faces": bool(same_initial_faces),
                "same_initial_state": bool(same_initial_state),
                "adaptive_summary": adaptive_result["summary"],
                "fixed_summary": fixed_result["summary"],
            }
        )

    write_json(benchmark_root / "paired_summary.json", {"pairs": pairs_summary})
    return benchmark_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive/fixed manifold experiments")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config")
    parser.add_argument("--mode", choices=["adaptive", "fixed"], default=None)
    parser.add_argument("--provider", choices=["classical", "recorded", "classiq"], default="classical")
    parser.add_argument("--recorded-path", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default="results/runs")
    parser.add_argument("--benchmark", choices=["paired"], default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)

    if args.steps is not None:
        config = replace(config, steps=args.steps)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.mode is not None:
        config = replace(config, topology_mode=args.mode)

    provider, _provenance = _resolve_provider(args.provider, args.recorded_path)
    config = replace(config, provider=args.provider)

    output_root = Path(args.output)

    if args.benchmark == "paired":
        benchmark_dir = _paired_runs(config, args.provider, provider, output_root, args.seeds)
        paired_summary = run_paired_benchmark(config=config, seeds=args.seeds)
        write_json(benchmark_dir / "paired_aggregate.json", paired_summary)
        print(f"Paired benchmark completed: {benchmark_dir}")
        return

    run_dir = _run_single(config, args.provider, provider, output_root)
    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()
