# Experiment Protocol

## Baseline Parameters

Recommended baseline:

- `nx=8`, `ny=8`
- `steps=1200`
- `dt=0.005`
- `sigma=10.0`, `rho=28.0`, `beta=8.0/3.0`
- `coupling=0.12`
- `surgery_interval=20`
- `surgery_attempts=80`
- `seed=11`
- `snapshot_steps=[0,100,300,600,1200]`

## Required Runs

1. Smoke run.
2. Full adaptive baseline.
3. Matched fixed control.
4. Paired multi-seed benchmark.

Fixed and adaptive controls must share identical initial mesh and state for each seed.

## Validation Checks

- No NaN/Inf states.
- Topology invariants preserved where required.
- Fixed mode keeps exact initial face list.
- Adaptive mode logs accepted surgeries.
- Snapshot files exist at configured steps.
- Serialized outputs reload successfully.
