# Project Structure

The repository is organized so the root reads like a GitHub project rather than
a scratch workspace.

```text
.
├── .github/workflows/tests.yml
├── .gitignore
├── README.md
├── pyproject.toml
├── requirements.txt
├── event_defect/
├── scripts/
├── tests/
├── docs/
├── experiments/
├── external/
├── outputs/
└── legacy/
```

## Tracked Source Areas

- `event_defect/`: reusable Python package.
- `scripts/`: command-line workflows for data preparation, training,
  evaluation and visualization.
- `tests/`: automated pytest coverage for package and scripts.
- `docs/`: research notes, dataset notes, reproduction instructions and
  experiment summaries.
- `legacy/`: archived original step scripts kept for traceability.

## Local/Generated Areas

- `external/`: downloaded datasets or cloned external repositories.
- `experiments/`: training outputs, metrics, checkpoints and generated event
  streams.
- `outputs/`: one-off visualizations and demos.
- `.torch/`: local pretrained model cache.

These local areas are intentionally ignored where they contain large binary
artifacts. Lightweight reports and metrics can remain in `experiments/` for
documentation.

## Root Directory Policy

The root should stay limited to:

- project metadata and configuration,
- the package directory,
- script/test/doc directories,
- high-level README.

New notebooks, scratch scripts and temporary outputs should go under `outputs/`
or `experiments/`, not the root.
