# Data Provenance

Each run records:

- provider and topology mode,
- configuration hash,
- random seed,
- Python and package versions,
- operating system details,
- git commit SHA,
- timestamps,
- data provenance mode:
  - `classical_simulation`
  - `recorded_replay`
  - `recorded_quantum_replay`
  - `quantum_local_arithmetic_prototype`

## Important Distinctions

- Classical adaptive manifold runs are not quantum hardware execution.
- Recorded replay is only as trustworthy as recorded metadata.
- Classiq local arithmetic synthesis validates a primitive and does not execute the full adaptive-manifold feedback loop.
