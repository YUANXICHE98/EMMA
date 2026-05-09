# EMMA: Evolutionary Metacognitive Memory Architecture

This repository contains a research implementation of EMMA, a memory-augmented agent architecture for frozen large language models. EMMA stores structured experience outside the model, retrieves relevant memories before each decision, updates memory values from environment feedback, and keeps benchmark adapters separate from the shared memory/RL core.

## What Is Included

- `environment/benchmark_core/`: shared EMMA runner, memory loop, value update, and hypergraph-style memory substrate used by benchmark adapters.
- `environment/*/adapter.py`: benchmark-local adapters for ALFWorld, ScienceWorld, InterCode-SQL, HLE, BigCodeBench, WebArena, OSWorld, SWE-bench, and LifelongAgentBench scaffolds.
- `exp/modules/`: legacy module implementation for LLM calls, embedding, retrieval, memory persistence, and RL optimization.
- `environment/hle/open_source_audit.py`: lightweight audit tool for HLE artifacts.
- `docs/REPRODUCIBILITY.md`: expected reproducibility scope, protocol notes, and why exact paper numbers are not promised by the default smoke runs.
- `docs/MODEL_PROTOCOL.md`: benchmark-specific model roles for paper-primary runs, baselines, scaling, and release smoke tests.
- `OPEN_SOURCE_AUDIT.md`: release checklist and HLE-specific audit notes.

## Design Boundary

EMMA is the shared memory mechanism. Benchmark code is responsible only for translating a benchmark episode into the shared interface:

- task description
- observation/state text
- action or answer
- environment verdict/reward
- adapter-local metadata needed for auditing

Benchmark-specific verifier fields, especially in HLE, are adapter-local evidence. They are not part of the shared EMMA algorithm and should not be interpreted as hidden task controllers.

## Quickstart

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r environment/hle/requirements.txt
```

Set an OpenAI-compatible endpoint:

```bash
export EMMA_OPENAI_API_KEY=...
export EMMA_OPENAI_BASE_URL=https://api.openai.com/v1
export EMMA_OPENAI_MODEL=gpt-4o-mini

export EMMA_EMBEDDING_API_KEY="$EMMA_OPENAI_API_KEY"
export EMMA_EMBEDDING_BASE_URL="$EMMA_OPENAI_BASE_URL"
export EMMA_EMBEDDING_MODEL=text-embedding-3-large
```

The quickstart model is a low-cost smoke default. It is not the paper-primary solver. For paper-quality runs, choose the benchmark-specific model role in `docs/MODEL_PROTOCOL.md`.

Run an HLE smoke test on the public text-only subset:

```bash
python environment/run_memrl_benchmark.py \
  --benchmark hle \
  --condition full \
  --episodes 5 \
  --start-index 0 \
  --results-dir environment/hle/results/smoke
```

Run protocol sanity checks before drawing conclusions from HLE outputs:

```bash
python environment/hle/protocol_sanity_check.py \
  --cases environment/hle/protocol_sanity_cases.json

python environment/hle/numeric_boundary_sanity_check.py
```

SWE-bench executes model-generated patches through the official harness. Run it only inside an isolated container or sandbox. The adapter refuses host execution unless `EMMA_SWEBENCH_ALLOW_HOST_HARNESS=1` is set.

## Reproducibility Scope

The default scripts are intended to reproduce the mechanism and protocol checks, not the exact table numbers from the paper. Exact benchmark scores depend on:

- model family and release date
- provider route and API protocol
- dataset access and filtering
- judge mode
- budget, retry, timeout, and sampling settings

This is expected for API-served LLM benchmark experiments. See `docs/REPRODUCIBILITY.md` for the recommended reporting format.

Model roles are explicit in this release: `gpt-5.2` is the default paper-primary family for general/reasoning benchmarks, `gpt-5.2-codex` is the paper-primary family for code benchmarks, `gpt-4-0125-preview` is the legacy baseline, Llama/Qwen are scaling models, and DeepSeek/GPT-4o-mini-class models are cost or smoke settings unless explicitly reported.

## Provider Configuration

The code uses `EMMA_*` environment variables first and keeps `MEMRL_*` as compatibility aliases. API keys should never be written to config files.

Useful variables:

- `EMMA_OPENAI_API_KEY`
- `EMMA_OPENAI_BASE_URL`
- `EMMA_OPENAI_MODEL`
- `EMMA_EMBEDDING_API_KEY`
- `EMMA_EMBEDDING_BASE_URL`
- `EMMA_EMBEDDING_MODEL`
- `EMMA_HLE_JUDGE_MODE`
- `EMMA_HLE_JUDGE_MODEL`

## Release Hygiene

Result folders, logs, virtual environments, local notes, and credentials are intentionally ignored. Before publishing result artifacts, run the audit checks described in `OPEN_SOURCE_AUDIT.md`.

Evaluator-only fields such as HLE gold answers are kept out of public per-episode metadata emitted by the shared runner.

## License

MIT. See `LICENSE`.
