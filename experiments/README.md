# Experiments

Runnable experiment scripts that use the `trinet` library. Reusable code lives in
`src/trinet/`; this directory holds one-off studies, ablations, and analyses.

| Script | Purpose |
|---|---|
| `binary_screening.py` | Binary Mpox-vs-rest evaluation (sensitivity/specificity + Wilson CIs) |

Most standard runs are available directly through the CLI — see `trinet --help`
(e.g. `trinet fusion`, `trinet cross-val`, `trinet diversity`).
