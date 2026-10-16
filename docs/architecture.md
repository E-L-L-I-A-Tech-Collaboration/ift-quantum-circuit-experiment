# Architecture

## Pipeline

State provider
-> local state evolution
-> topological diagnostics
-> validated surgery
-> updated adjacency
-> next state evolution

## Modules

- `src/providers`: Provider protocol and concrete providers.
- `src/experiments/adaptive_manifold.py`: Main loop for fixed and adaptive topology modes.
- `src/topology/surgery.py`: Candidate edge flips, topology preservation checks, structured surgery records.
- `src/experiments/io.py`: Reproducible run serialization and reload utilities.

## Provider Contract

Each provider must expose:

- `initialize(vertex_count, seed, config)` returning finite `(vertex_count, 3)` arrays.
- `evolve(state, adjacency, config, step)` returning finite `(vertex_count, 3)` arrays.
- `metadata()` returning provenance and provider metadata.

Future quantum output can replace the classical provider without changing manifold topology logic.
