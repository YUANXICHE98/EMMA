# ScienceWorld Benchmark

This directory wires ScienceWorld into the same benchmark entry surface as the other MemRL environments.

## Intended usage

Run ScienceWorld with the project-local virtual environment:

```bash
environment/.venv-scienceworld/bin/python \
  environment/scienceworld/run_memrl_scienceworld.py \
  --condition full \
  --episodes 1 \
  --start-index 0
```

## Notes

- Memory remains benchmark-scoped. ScienceWorld writes its own memory files under `environment/scienceworld/results/`.
- This adapter depends on the `scienceworld` Python package and a local Java runtime.
- Task/variation selection is controlled through `environment/scienceworld/config.yaml`.
