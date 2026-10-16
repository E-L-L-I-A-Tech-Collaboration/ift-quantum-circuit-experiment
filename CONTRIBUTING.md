# Contributing

## Principles

- Preserve scientific honesty in naming and claims.
- Keep classical and optional quantum tracks clearly separated.
- Add tests for all behavior changes.
- Maintain reproducibility of generated results.

## Development Setup

```bash
pip install -e .[dev]
pytest -q
```

## Optional Quantum Tooling

```bash
pip install -e .[quantum]
```

Never require Classiq for core package import, classical runs, or CI.

## Pull Requests

- Include summary of changed files and behavior.
- Include exact commands used for testing.
- If experiments were run, include paths to generated artifacts.
- Avoid committing ad hoc large run directories; prefer curated references only.
