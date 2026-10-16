from __future__ import annotations

from typing import Sequence, Set

import numpy as np

from .coupling import diffusive_coupling


def lorenz_network_rhs(
    state: np.ndarray,
    adjacency: Sequence[Set[int]],
    sigma: float,
    rho: float,
    beta: float,
    coupling: float,
) -> np.ndarray:
    x, y, z = state.T
    local = np.column_stack(
        [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ]
    )
    return local + coupling * diffusive_coupling(state, adjacency)


def rk4_step(
    state: np.ndarray,
    adjacency: Sequence[Set[int]],
    dt: float,
    sigma: float,
    rho: float,
    beta: float,
    coupling: float,
) -> np.ndarray:
    rhs = lambda sample: lorenz_network_rhs(sample, adjacency, sigma, rho, beta, coupling)
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)