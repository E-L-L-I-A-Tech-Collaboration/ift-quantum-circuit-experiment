from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LorenzStepSpec:
    sigma: float = 10.0
    beta: float = 8.0 / 3.0
    rho: float = 28.0
    dt: float = 0.01


def simulate_lorenz_euler(state: np.ndarray, spec: LorenzStepSpec) -> np.ndarray:
    x, y, z = state
    dx = spec.sigma * (y - x)
    dy = x * (spec.rho - z) - y
    dz = x * y - spec.beta * z
    return np.asarray([x + spec.dt * dx, y + spec.dt * dy, z + spec.dt * dz], dtype=float)


class SimulatorProvider:
    def __init__(self, spec: LorenzStepSpec | None = None):
        self.spec = spec or LorenzStepSpec()

    def local_step(self, state: np.ndarray) -> np.ndarray:
        return simulate_lorenz_euler(state, self.spec)