# InterCode-SQL Local Benchmark

This directory contains the InterCode-SQL adapter for the shared MemRL brain.

## What is implemented

- Local Python environment at `environment/.conda-intercode`
- Official InterCode SQL dataset and Docker image usage via `intercode-bench`
- The shared MemRL brain now lives in `environment/benchmark_core/`
- This benchmark is exposed through a thin adapter layer plus the shared runner `environment/run_memrl_benchmark.py`

## Current shared-brain status

The benchmark interface is now aligned across benchmarks, but the current benchmark-agnostic backend only supports the subset already implemented in the legacy MemRL modules:

- `full`
- `no_memory`
- `no_FailureMemory`

The higher-level brain ablations still exist in the shared registry, but they intentionally raise explicit errors until the shared brain backend really implements them.

## First smoke test

```bash
environment/.conda-intercode/bin/python \
  environment/run_memrl_benchmark.py \
  --benchmark intercode_sql \
  --condition full \
  --episodes 1 \
  --start-index 0
```

## Notes

- The SQL Docker image is built from the official package assets and uses the built-in `sql_queries.csv` dataset.
- Results are written under `environment/intercode_sql/results/`.
- For paper-primary runs, use a frontier SQL-capable reasoning solver. GPT-4-0125-preview is a legacy baseline; Qwen/DeepSeek-class models are cost or scaling rows unless explicitly reported.
