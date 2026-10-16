from __future__ import annotations

from typing import Any

import numpy as np

from .simulator_provider import LorenzStepSpec


class HardwareProvider:
    def __init__(self, spec: LorenzStepSpec | None = None):
        self.spec = spec or LorenzStepSpec()

    def submit_lorenz_step(self, state: np.ndarray) -> Any:
        raise NotImplementedError("Hardware execution is not implemented in this prototype.")