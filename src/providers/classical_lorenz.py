from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Sequence, Set

import numpy as np

from src.dynamics.lorenz import rk4_step

from .base import ProviderContext, StateProvider


class ClassicalLorenzProvider(StateProvider):
    """Default state provider using coupled Lorenz RK4 dynamics."""

    def initialize(self, vertex_count: int, seed: int, config: Any) -> np.ndarray:
        rng = np.random.default_rng(seed)
        state = rng.normal(0.0, 0.08, size=(vertex_count, 3))
        state[:, 0] += 0.1
        return state

    def evolve(
        self,
        state: np.ndarray,
        adjacency: Sequence[Set[int]],
        config: Any,
        step: int,
    ) -> np.ndarray:
        return rk4_step(
            state,
            adjacency,
            config.dt,
            config.sigma,
            config.rho,
            config.beta,
            config.coupling,
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "provider": "classical_lorenz",
            "mode": "classical_simulation",
            "description": "Coupled Lorenz RK4 integration over mesh adjacency.",
            "context_schema": list(asdict(ProviderContext(0.0, 0.0, 0.0, 0.0, 0.0)).keys()),
        }
