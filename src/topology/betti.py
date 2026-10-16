from __future__ import annotations

from dataclasses import dataclass

from src.manifold.invariants import Invariants


@dataclass(frozen=True)
class BettiNumbers:
    beta0: int
    beta1: int
    beta2: int


def estimate_betti_numbers(invariants: Invariants) -> BettiNumbers:
    if invariants.nonmanifold_edges:
        raise ValueError("Betti number estimation requires a manifold mesh.")
    if not invariants.orientable:
        raise ValueError("Betti number estimation requires orientable surfaces.")

    beta0 = invariants.connected_components
    beta2 = 0 if invariants.boundary_loops > 0 else invariants.connected_components
    beta1 = beta0 + beta2 - invariants.euler_characteristic
    if beta1 < 0:
        raise ValueError("Invariants do not describe a valid orientable surface.")
    return BettiNumbers(beta0=beta0, beta1=beta1, beta2=beta2)