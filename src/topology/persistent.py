from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

from src.manifold.geometry import Face
from src.manifold.invariants import Invariants, compute_invariants

from .betti import BettiNumbers, estimate_betti_numbers


@dataclass(frozen=True)
class TopologySnapshot:
    step: int
    invariants: Invariants
    betti: BettiNumbers


def snapshot_topology(step: int, n_vertices: int, faces: Sequence[Face]) -> TopologySnapshot:
    invariants = compute_invariants(n_vertices, faces)
    betti = estimate_betti_numbers(invariants)
    return TopologySnapshot(step=step, invariants=invariants, betti=betti)


def persistent_signature(history: Iterable[TopologySnapshot]) -> Tuple[Tuple[int, int, int], ...]:
    return tuple((entry.step, entry.betti.beta0, entry.betti.beta1) for entry in history)


def topology_unchanged(reference: TopologySnapshot, candidate: TopologySnapshot) -> bool:
    return reference.invariants == candidate.invariants and reference.betti == candidate.betti