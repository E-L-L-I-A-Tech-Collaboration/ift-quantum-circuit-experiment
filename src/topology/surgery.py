from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from src.manifold.geometry import Edge, Face, canonical_edge, orient_ccw, oriented_area
from src.manifold.invariants import Invariants, compute_invariants
from src.manifold.mesh import edge_to_faces


@dataclass(frozen=True)
class SurgeryProposalRecord:
    step: int
    candidate_edge: Edge
    accepted: bool
    rejection_reason: str
    objective_before: float
    candidate_objective: float
    objective_delta: float
    topology_before: Dict[str, object]
    topology_after: Dict[str, object]
    triangle_quality_before: float
    triangle_quality_after: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def opposite_vertex(face: Face, edge: Edge) -> int:
    for vertex in face:
        if vertex not in edge:
            return vertex
    raise ValueError("No opposite vertex found")


def propose_edge_flip(points: np.ndarray, faces: Sequence[Face], edge: Edge) -> List[Face] | None:
    edge_owners = edge_to_faces(faces)
    owners = edge_owners.get(edge, [])
    if len(owners) != 2:
        return None

    first_index, second_index = owners
    first_face, second_face = faces[first_index], faces[second_index]
    u, v = edge
    a = opposite_vertex(first_face, edge)
    b = opposite_vertex(second_face, edge)

    if a == b or canonical_edge(a, b) in edge_owners:
        return None

    new_first = orient_ccw(points, (a, b, u))
    new_second = orient_ccw(points, (b, a, v))
    if abs(oriented_area(points, new_first)) < 1e-10:
        return None
    if abs(oriented_area(points, new_second)) < 1e-10:
        return None

    candidate = list(faces)
    candidate[first_index] = new_first
    candidate[second_index] = new_second
    return candidate


def topology_preserved(candidate: Invariants, baseline: Invariants) -> bool:
    return (
        candidate.euler_characteristic == baseline.euler_characteristic
        and candidate.connected_components == baseline.connected_components
        and candidate.boundary_loops == baseline.boundary_loops
        and candidate.nonmanifold_edges == 0
        and candidate.orientable == baseline.orientable
    )


def _mean_triangle_quality(points: np.ndarray, faces: Sequence[Face]) -> float:
    from src.manifold.geometry import triangle_quality

    if not faces:
        return 0.0
    qualities = [triangle_quality(points, face) for face in faces]
    return float(np.mean(qualities))


def attempt_state_driven_surgery(
    points: np.ndarray,
    faces: List[Face],
    baseline_invariants: Invariants,
    attempts: int,
    rng,
    objective_fn: Callable[[Sequence[Face]], float],
) -> Tuple[List[Face], bool, float, float, List[SurgeryProposalRecord]]:
    current_objective = objective_fn(faces)
    topology_before = compute_invariants(len(points), faces)
    quality_before = _mean_triangle_quality(points, faces)
    interior_edges = [edge for edge, owners in edge_to_faces(faces).items() if len(owners) == 2]
    rng.shuffle(interior_edges)
    logs: List[SurgeryProposalRecord] = []

    for edge in interior_edges[:attempts]:
        rejection_reason = ""
        candidate_objective = current_objective
        topology_after = topology_before
        quality_after = quality_before

        edge_owners = edge_to_faces(faces)
        owners = edge_owners.get(edge, [])
        if len(owners) != 2:
            rejection_reason = "boundary_edge"
            logs.append(
                SurgeryProposalRecord(
                    step=-1,
                    candidate_edge=edge,
                    accepted=False,
                    rejection_reason=rejection_reason,
                    objective_before=current_objective,
                    candidate_objective=candidate_objective,
                    objective_delta=0.0,
                    topology_before=asdict(topology_before),
                    topology_after=asdict(topology_before),
                    triangle_quality_before=quality_before,
                    triangle_quality_after=quality_before,
                )
            )
            continue

        candidate = propose_edge_flip(points, faces, edge)
        if candidate is None:
            first_index, second_index = owners
            first_face, second_face = faces[first_index], faces[second_index]
            u, v = edge
            a = opposite_vertex(first_face, edge)
            b = opposite_vertex(second_face, edge)
            if a == b:
                rejection_reason = "invalid_quadrilateral"
            elif canonical_edge(a, b) in edge_owners:
                rejection_reason = "duplicate_diagonal"
            else:
                proposed_first = orient_ccw(points, (a, b, u))
                proposed_second = orient_ccw(points, (b, a, v))
                if abs(oriented_area(points, proposed_first)) < 1e-10 or abs(
                    oriented_area(points, proposed_second)
                ) < 1e-10:
                    rejection_reason = "degenerate_triangle"
                else:
                    rejection_reason = "invalid_quadrilateral"
            logs.append(
                SurgeryProposalRecord(
                    step=-1,
                    candidate_edge=edge,
                    accepted=False,
                    rejection_reason=rejection_reason,
                    objective_before=current_objective,
                    candidate_objective=candidate_objective,
                    objective_delta=0.0,
                    topology_before=asdict(topology_before),
                    topology_after=asdict(topology_before),
                    triangle_quality_before=quality_before,
                    triangle_quality_after=quality_before,
                )
            )
            continue

        candidate_invariants = compute_invariants(len(points), candidate)
        topology_after = candidate_invariants
        quality_after = _mean_triangle_quality(points, candidate)

        if candidate_invariants.nonmanifold_edges > 0:
            rejection_reason = "non_manifold_result"
        elif candidate_invariants.orientable != baseline_invariants.orientable:
            rejection_reason = "orientability_violation"
        elif not topology_preserved(candidate_invariants, baseline_invariants):
            rejection_reason = "topology_violation"

        if rejection_reason:
            logs.append(
                SurgeryProposalRecord(
                    step=-1,
                    candidate_edge=edge,
                    accepted=False,
                    rejection_reason=rejection_reason,
                    objective_before=current_objective,
                    candidate_objective=candidate_objective,
                    objective_delta=0.0,
                    topology_before=asdict(topology_before),
                    topology_after=asdict(candidate_invariants),
                    triangle_quality_before=quality_before,
                    triangle_quality_after=quality_after,
                )
            )
            continue

        candidate_objective = objective_fn(candidate)
        if candidate_objective + 1e-12 < current_objective:
            logs.append(
                SurgeryProposalRecord(
                    step=-1,
                    candidate_edge=edge,
                    accepted=True,
                    rejection_reason="",
                    objective_before=current_objective,
                    candidate_objective=candidate_objective,
                    objective_delta=candidate_objective - current_objective,
                    topology_before=asdict(topology_before),
                    topology_after=asdict(candidate_invariants),
                    triangle_quality_before=quality_before,
                    triangle_quality_after=quality_after,
                )
            )
            return candidate, True, current_objective, candidate_objective, logs

        logs.append(
            SurgeryProposalRecord(
                step=-1,
                candidate_edge=edge,
                accepted=False,
                rejection_reason="objective_not_improved",
                objective_before=current_objective,
                candidate_objective=candidate_objective,
                objective_delta=candidate_objective - current_objective,
                topology_before=asdict(topology_before),
                topology_after=asdict(candidate_invariants),
                triangle_quality_before=quality_before,
                triangle_quality_after=quality_after,
            )
        )

    return faces, False, current_objective, current_objective, logs