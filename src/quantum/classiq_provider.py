from __future__ import annotations

from typing import Any, Tuple

from .simulator_provider import LorenzStepSpec


DEFAULT_FP_SIZE = 8
DEFAULT_FP_FRAC = 4


def synthesize_lorenz_step(
    spec: LorenzStepSpec | None = None,
    initial_state: Tuple[float, float, float] = (0.125, 0.0, 0.0),
    fp_size: int = DEFAULT_FP_SIZE,
    fp_frac: int = DEFAULT_FP_FRAC,
) -> Any:
    try:
        from classiq import CReal, Output, QNum, SIGNED, allocate, qfunc, show, synthesize
    except ImportError as exc:
        raise RuntimeError("Install classiq to synthesize the Lorenz step provider.") from exc

    step_spec = spec or LorenzStepSpec()

    @qfunc
    def lorenz_euler_step(
        x: QNum[fp_size, SIGNED, fp_frac],
        y: QNum[fp_size, SIGNED, fp_frac],
        z: QNum[fp_size, SIGNED, fp_frac],
        sigma: CReal,
        beta: CReal,
        rho: CReal,
        dt: CReal,
    ) -> None:
        x += dt * sigma * (y - x)
        y += dt * (x * (rho - z) - y)
        z += dt * (x * y - beta * z)

    @qfunc
    def program(
        x_out: Output[QNum[fp_size, SIGNED, fp_frac]],
        y_out: Output[QNum[fp_size, SIGNED, fp_frac]],
        z_out: Output[QNum[fp_size, SIGNED, fp_frac]],
    ) -> None:
        allocate(x_out)
        allocate(y_out)
        allocate(z_out)

        x_out |= initial_state[0]
        y_out |= initial_state[1]
        z_out |= initial_state[2]

        lorenz_euler_step(
            x_out,
            y_out,
            z_out,
            sigma=step_spec.sigma,
            beta=step_spec.beta,
            rho=step_spec.rho,
            dt=step_spec.dt,
        )

    quantum_program = synthesize(program)
    show(quantum_program)
    return quantum_program


def main() -> None:
    synthesize_lorenz_step()


if __name__ == "__main__":
    main()