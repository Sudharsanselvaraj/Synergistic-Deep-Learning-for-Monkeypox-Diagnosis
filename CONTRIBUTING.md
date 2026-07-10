# Contributing

Thanks for your interest in Tri-Net v2!

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"       # add ".[metal]" on Apple Silicon
pre-commit install
```

## Guidelines

- **Reusable code** lives in `src/trinet/`; **experiment scripts** live in `experiments/`.
  Nothing experiment-specific belongs in the package.
- **Never fabricate metrics.** Every reported number must be regenerable from code on the
  held-out test set. Keep the train/val/test split leakage-free.
- **Generated artifacts** (`outputs/`, `data/`) are git-ignored — do not commit checkpoints,
  logs, or large binaries.
- Run `ruff` and `pytest` before opening a PR (`pre-commit` runs them automatically).

## Pull requests

1. Fork and create a feature branch.
2. Add tests for new behavior under `tests/`.
3. Ensure `pytest` passes and the CLI still runs (`trinet --help`).
4. Use clear, conventional commit messages (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`).
