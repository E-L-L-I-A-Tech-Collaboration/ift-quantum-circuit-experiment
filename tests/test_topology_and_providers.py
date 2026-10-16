from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.experiments.adaptive_manifold import Config, run_experiment
from src.manifold.invariants import compute_invariants
from src.manifold.mesh import build_rectangular_triangulation, edge_to_faces
from src.providers import ClassicalLorenzProvider, RecordedStateProvider
from src.topology.surgery import attempt_state_driven_surgery, propose_edge_flip


def test_initial_rectangular_triangulation_invariants() -> None:
    points, faces = build_rectangular_triangulation(4, 4, seed=0)
    invariants = compute_invariants(len(points), faces)
    assert invariants.vertices == 16
    assert invariants.faces == 18
    assert invariants.euler_characteristic == 1
    assert invariants.connected_components == 1
    assert invariants.boundary_loops == 1
    assert invariants.nonmanifold_edges == 0
    assert invariants.orientable


def test_valid_interior_flip_preserves_core_invariants() -> None:
    points, faces = build_rectangular_triangulation(4, 4, seed=1)
    baseline = compute_invariants(len(points), faces)

    interior_edges = [edge for edge, owners in edge_to_faces(faces).items() if len(owners) == 2]
    assert interior_edges
    candidate = None
    for edge in interior_edges:
        candidate = propose_edge_flip(points, faces, edge)
        if candidate is not None:
            break
    assert candidate is not None

    after = compute_invariants(len(points), candidate)
    assert after.euler_characteristic == baseline.euler_characteristic
    assert after.connected_components == baseline.connected_components
    assert after.boundary_loops == baseline.boundary_loops
    assert after.orientable == baseline.orientable
    assert after.nonmanifold_edges == 0


def test_invalid_flip_boundary_edge_rejected() -> None:
    points, faces = build_rectangular_triangulation(4, 4, seed=2)
    boundary_edge = next(edge for edge, owners in edge_to_faces(faces).items() if len(owners) == 1)
    candidate = propose_edge_flip(points, faces, boundary_edge)
    assert candidate is None


def test_fixed_mode_preserves_exact_faces() -> None:
    result = run_experiment(Config(nx=4, ny=4, steps=40, topology_mode="fixed", seed=5))
    assert result["initial_faces"] == result["faces"]


def test_adaptive_mode_accepts_improving_flip_controlled_fixture() -> None:
    points = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.1, 1.0], [0.0, 1.0]], dtype=float)
    faces = [(0, 1, 2), (0, 2, 3)]
    baseline = compute_invariants(len(points), faces)

    def objective_fn(candidate_faces):
        has_good_diagonal = any(set(face) == {0, 1, 3} for face in candidate_faces)
        return 0.0 if has_good_diagonal else 1.0

    rng = np.random.default_rng(0)
    updated_faces, accepted, old_obj, new_obj, logs = attempt_state_driven_surgery(
        points,
        list(faces),
        baseline,
        attempts=1,
        rng=rng,
        objective_fn=objective_fn,
    )

    assert accepted
    assert new_obj < old_obj
    assert len(logs) >= 1
    assert any(record.accepted for record in logs)
    assert updated_faces != faces


def test_identical_seeds_reproduce_initial_mesh_and_state() -> None:
    cfg = Config(nx=5, ny=5, steps=10, seed=7)
    run_a = run_experiment(cfg)
    run_b = run_experiment(cfg)

    assert np.allclose(run_a["points"], run_b["points"])
    assert run_a["initial_faces"] == run_b["initial_faces"]
    assert np.allclose(run_a["initial_state"], run_b["initial_state"])


def test_classical_provider_shape_and_finite() -> None:
    provider = ClassicalLorenzProvider()
    state = provider.initialize(vertex_count=10, seed=3, config=Config(steps=2))
    assert state.shape == (10, 3)
    assert np.isfinite(state).all()


def test_recorded_provider_rejects_malformed_dimensions(tmp_path) -> None:
    malformed = tmp_path / "bad.npz"
    np.savez(malformed, states=np.zeros((3, 4)))
    with pytest.raises(ValueError, match="shape"):
        RecordedStateProvider(malformed)


def test_recorded_provider_rejects_non_finite_values(tmp_path) -> None:
    malformed = tmp_path / "bad_inf.npz"
    states = np.zeros((3, 4, 3), dtype=float)
    states[0, 0, 0] = np.inf
    np.savez(malformed, states=states)
    with pytest.raises(ValueError, match="non-finite"):
        RecordedStateProvider(malformed)


def test_fixed_and_adaptive_start_identically_for_same_seed() -> None:
    base = Config(nx=5, ny=5, steps=20, seed=13)
    fixed = run_experiment(replace(base, topology_mode="fixed"))
    adaptive = run_experiment(replace(base, topology_mode="adaptive"))

    assert np.allclose(fixed["points"], adaptive["points"])
    assert fixed["initial_faces"] == adaptive["initial_faces"]
    assert np.allclose(fixed["initial_state"], adaptive["initial_state"])
