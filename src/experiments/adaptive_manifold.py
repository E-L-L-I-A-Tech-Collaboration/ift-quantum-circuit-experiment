from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import random
import time
from typing import Any, Literal

import numpy as np
from scipy import linalg

from src.dynamics.coupling import field_mismatch_energy
from src.manifold.geometry import Face, triangle_quality
from src.manifold.invariants import compute_invariants
from src.manifold.mesh import adjacency_from_faces, build_rectangular_triangulation
from src.providers import ClassicalLorenzProvider, StateProvider
from src.topology.persistent import persistent_signature, snapshot_topology
from src.topology.surgery import SurgeryProposalRecord, attempt_state_driven_surgery


@dataclass
class Config:
    nx: int = 8
    ny: int = 8
    steps: int = 1200
    dt: float = 0.005
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    coupling: float = 0.12
    surgery_interval: int = 20
    surgery_attempts: int = 80
    seed: int = 11
    topology_mode: Literal["fixed", "adaptive"] = "adaptive"
    provider: str = "classical"
    detailed_surgery_logging: bool = False
    transient_cutoff: int = 200
    snapshot_steps: list[int] = field(default_factory=lambda: [0, 100, 300, 600, 1200])


def objective(
    points: np.ndarray,
    faces: list[Face] | tuple[Face, ...],
    state: np.ndarray,
    geometry_weight: float = 0.30,
    field_weight: float = 1.00,
) -> float:
    field_energy = field_mismatch_energy(state, faces)
    qualities = [triangle_quality(points, face) for face in faces]
    geometry_penalty = float(np.mean([(1.0 - quality) ** 2 for quality in qualities]))
    return field_weight * field_energy + geometry_weight * geometry_penalty


def _sync_metric(state: np.ndarray) -> float:
    center = state.mean(axis=0)
    return float(np.mean(np.linalg.norm(state - center, axis=1)))


def _graph_metrics(adjacency: list[set[int]]) -> dict[str, float | int | None]:
    n = len(adjacency)
    degrees = np.asarray([len(neighbors) for neighbors in adjacency], dtype=float)
    mean_degree = float(np.mean(degrees)) if n else 0.0

    diameter: int | None = None
    if n > 0:
        # Compute graph diameter with BFS from each source; suitable for this mesh scale.
        max_distance = 0
        disconnected = False
        for source in range(n):
            distances = [-1] * n
            distances[source] = 0
            queue = [source]
            head = 0
            while head < len(queue):
                node = queue[head]
                head += 1
                for neighbor in adjacency[node]:
                    if distances[neighbor] == -1:
                        distances[neighbor] = distances[node] + 1
                        queue.append(neighbor)
            if any(distance < 0 for distance in distances):
                disconnected = True
                break
            max_distance = max(max_distance, max(distances))
        if not disconnected:
            diameter = int(max_distance)

    spectral_gap: float | None = None
    if n > 1:
        matrix = np.zeros((n, n), dtype=float)
        for i, neighbors in enumerate(adjacency):
            matrix[i, i] = len(neighbors)
            for j in neighbors:
                matrix[i, j] = -1.0
        eigenvalues = np.sort(linalg.eigvalsh(matrix))
        if len(eigenvalues) >= 2:
            spectral_gap = float(max(0.0, eigenvalues[1]))

    return {
        "mean_vertex_degree": mean_degree,
        "graph_diameter": diameter,
        "laplacian_spectral_gap": spectral_gap,
    }


def _mesh_quality_metrics(points: np.ndarray, faces: list[Face]) -> dict[str, float]:
    qualities = np.asarray([triangle_quality(points, face) for face in faces], dtype=float)
    if qualities.size == 0:
        return {"mean_triangle_quality": 0.0, "min_triangle_quality": 0.0}
    return {
        "mean_triangle_quality": float(np.mean(qualities)),
        "min_triangle_quality": float(np.min(qualities)),
    }


def _build_snapshot(
    step: int,
    dt: float,
    points: np.ndarray,
    faces: list[Face],
    state: np.ndarray,
    objective_value: float,
    sync_value: float,
) -> dict[str, Any]:
    adjacency = adjacency_from_faces(len(points), faces)
    topo = snapshot_topology(step, len(points), faces)
    graph = _graph_metrics(adjacency)
    mesh_quality = _mesh_quality_metrics(points, faces)
    return {
        "step": int(step),
        "time": float(step * dt),
        "state": np.array(state, dtype=float),
        "faces": np.asarray(faces, dtype=int),
        "topology": {
            "invariants": asdict(topo.invariants),
            "betti": asdict(topo.betti),
        },
        "objective": float(objective_value),
        "synchronization": float(sync_value),
        "graph": graph,
        "mesh_quality": mesh_quality,
    }


def run_experiment(config: Config = Config(), provider: StateProvider | None = None) -> dict:
    if config.steps < 1:
        raise ValueError("steps must be >= 1")
    if config.topology_mode not in {"fixed", "adaptive"}:
        raise ValueError("topology_mode must be either 'fixed' or 'adaptive'")

    start_time = time.perf_counter()
    np_rng = np.random.default_rng(config.seed)
    py_rng = random.Random(config.seed)

    points, faces = build_rectangular_triangulation(config.nx, config.ny, seed=config.seed)
    initial_faces = list(faces)
    baseline = compute_invariants(len(points), faces)
    topology_history = [snapshot_topology(0, len(points), faces)]

    resolved_provider = provider or ClassicalLorenzProvider()
    if config.provider and config.provider != "classical" and provider is None:
        raise ValueError(
            "Non-classical provider requested via config.provider but no provider instance was supplied."
        )

    state = resolved_provider.initialize(len(points), config.seed, config)
    if state.shape != (len(points), 3):
        raise ValueError(
            f"Provider returned state with shape {state.shape}; expected {(len(points), 3)}."
        )
    if not np.isfinite(state).all():
        raise ValueError("Provider returned non-finite values during initialization.")

    initial_state = np.array(state, dtype=float)
    snapshot_steps = sorted(set([0, config.steps, *config.snapshot_steps]))

    history_t: list[float] = []
    history_obj: list[float] = []
    history_sync: list[float] = []
    accepted_steps: list[tuple[int, float, float]] = []
    surgery_log: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []

    for step in range(config.steps + 1):
        adjacency = adjacency_from_faces(len(points), faces)
        current_objective = objective(points, faces, state)
        current_sync = _sync_metric(state)
        history_t.append(step * config.dt)
        history_obj.append(current_objective)
        history_sync.append(current_sync)

        if step in snapshot_steps:
            snapshots.append(
                _build_snapshot(
                    step,
                    config.dt,
                    points,
                    faces,
                    state,
                    current_objective,
                    current_sync,
                )
            )

        if (
            config.topology_mode == "adaptive"
            and step > 0
            and step % config.surgery_interval == 0
        ):
            objective_fn = lambda candidate_faces: objective(points, candidate_faces, state)
            faces, accepted, old_objective, new_objective, proposal_logs = attempt_state_driven_surgery(
                points,
                faces,
                baseline,
                config.surgery_attempts,
                py_rng,
                objective_fn,
            )

            for proposal in proposal_logs:
                record: SurgeryProposalRecord = proposal
                record_dict = record.to_dict()
                record_dict["step"] = int(step)
                record_dict["candidate_edge"] = list(record_dict["candidate_edge"])
                if config.detailed_surgery_logging or record.accepted:
                    surgery_log.append(record_dict)

            if accepted:
                accepted_steps.append((step, old_objective, new_objective))
                topology_history.append(snapshot_topology(step, len(points), faces))

        if step < config.steps:
            state = resolved_provider.evolve(
                state,
                adjacency,
                config,
                step,
            )
            if state.shape != (len(points), 3):
                raise ValueError(
                    f"Provider returned state with shape {state.shape}; expected {(len(points), 3)}."
                )
            if not np.isfinite(state).all():
                raise ValueError(f"Provider produced non-finite values at step {step}.")

    if config.topology_mode == "fixed" and faces != initial_faces:
        raise RuntimeError("Fixed topology mode violated: final faces differ from initial faces.")

    final_invariants = compute_invariants(len(points), faces)
    adjacency_final = adjacency_from_faces(len(points), faces)
    final_obj = float(history_obj[-1])
    transient_index = max(0, min(len(history_obj) - 1, config.transient_cutoff))
    post_obj = np.asarray(history_obj[transient_index:], dtype=float)
    post_sync = np.asarray(history_sync[transient_index:], dtype=float)
    rejection_counts = Counter(
        row["rejection_reason"]
        for row in surgery_log
        if not row["accepted"] and row["rejection_reason"]
    )

    runtime_seconds = float(time.perf_counter() - start_time)
    graph_metrics = _graph_metrics(adjacency_final)
    mesh_quality = _mesh_quality_metrics(points, faces)

    summary = {
        "seed": config.seed,
        "provider": resolved_provider.metadata().get("provider", config.provider),
        "topology_mode": config.topology_mode,
        "accepted_surgeries": len(accepted_steps),
        "proposed_surgeries": len(surgery_log),
        "rejection_counts": dict(rejection_counts),
        "initial_objective": float(history_obj[0]),
        "final_objective": final_obj,
        "minimum_objective": float(np.min(history_obj)),
        "mean_objective_after_transient": float(np.mean(post_obj)),
        "final_synchronization": float(history_sync[-1]),
        "mean_synchronization_after_transient": float(np.mean(post_sync)),
        "runtime_seconds": runtime_seconds,
        "vertices": final_invariants.vertices,
        "edges": final_invariants.edges,
        "faces": final_invariants.faces,
        "euler_characteristic": final_invariants.euler_characteristic,
        "connected_components": final_invariants.connected_components,
        "boundary_loops": final_invariants.boundary_loops,
        "non_manifold_edges": final_invariants.nonmanifold_edges,
        "orientable": final_invariants.orientable,
        **graph_metrics,
        **mesh_quality,
    }

    return {
        "points": points,
        "initial_faces": initial_faces,
        "initial_state": initial_state,
        "faces": faces,
        "state": state,
        "baseline": baseline,
        "final_invariants": final_invariants,
        "history_t": np.asarray(history_t),
        "history_obj": np.asarray(history_obj),
        "history_sync": np.asarray(history_sync),
        "accepted_steps": accepted_steps,
        "surgery_log": surgery_log,
        "snapshots": snapshots,
        "summary": summary,
        "provider_metadata": resolved_provider.metadata(),
        "config": config,
        "topology_history": topology_history,
        "persistent_signature": persistent_signature(topology_history),
        "runtime_seconds": runtime_seconds,
        "graph_metrics": graph_metrics,
        "mesh_quality": mesh_quality,
        "adjacency": adjacency_final,
    }


def main() -> None:
    import matplotlib.pyplot as plt

    from src.visualization.plotting import plot_mesh, plot_series

    result = run_experiment(Config())
    print("Initial invariants:")
    print(result["baseline"])
    print("\nFinal invariants:")
    print(result["final_invariants"])
    print(f"\nAccepted edge flips: {len(result['accepted_steps'])}")
    print("Persistent topology signature:")
    print(result["persistent_signature"])
    if result["accepted_steps"]:
        print("First five accepted surgeries (step, objective before, objective after):")
        for row in result["accepted_steps"][:5]:
            print(row)

    plot_mesh(
        result["points"],
        result["initial_faces"],
        result["initial_state"][:, 0],
        "Initial interaction manifold (initial x-state)",
    )
    plt.savefig("initial_manifold.png", dpi=180)
    plt.show()

    plot_mesh(
        result["points"],
        result["faces"],
        result["state"][:, 0],
        "Evolved interaction manifold (final x-state shown)",
    )
    plt.savefig("evolved_manifold.png", dpi=180)
    plt.show()

    plot_series(
        result["history_t"],
        result["history_obj"],
        xlabel="time",
        ylabel="field + geometry objective",
        title="Co-evolution objective",
        output_path="objective_history.png",
    )
    plt.show()

    plot_series(
        result["history_t"],
        result["history_sync"],
        xlabel="time",
        ylabel="mean distance from network state centroid",
        title="Network synchronization / dispersion",
        output_path="synchronization_history.png",
    )
    plt.show()