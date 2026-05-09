# HLE Adapter

This adapter treats Humanity's Last Exam as a closed-ended QA benchmark under the shared EMMA runner.

## Design boundary

- one episode = one HLE question
- one action = one final answer
- no benchmark-specific controller
- text-only mode is the default
- HLE is supplementary breadth evidence, not the main memory-loop benchmark
- benchmark-local verifier feedback may be emitted by this adapter, but it should be presented as adapter-local audit evidence, not as shared EMMA core logic

## Dataset boundary

- default public smoke dataset: `koiwave/hle-short`
- official benchmark source: `cais/hle`
- official `cais/hle` is gated and explicitly asks users not to re-share the dataset
- current adapter supports three sources:
  - Hugging Face dataset name via `runner.dataset_name` or `EMMA_HLE_DATASET_NAME`
  - local `.parquet` / `.json` / `.jsonl` via `runner.dataset_path` or `EMMA_HLE_DATASET_PATH`
  - official `cais/hle` if the local runtime already has access credentials configured

The default is intentionally a public text-only subset so Docker smoke runs do not fail immediately on the official gate.

## Model role

HLE is solver-ceiling sensitive. Use `gpt-5.2` or another strongest available frontier reasoning model for paper-primary HLE runs. Keep `gpt-4-0125-preview` as a legacy comparable baseline. GPT-4o-mini-class, Qwen-32B-class, and DeepSeek-V3-class solvers are appropriate for smoke, ceiling, or cost-efficiency analysis only unless explicitly reported as their own table row.

## Open-source audit

Before publishing HLE artifacts, run:

```bash
python environment/hle/open_source_audit.py environment/hle/results/RELEASE_CANDIDATE \
  --output environment/hle/results/RELEASE_CANDIDATE/open_source_audit.json
```

The audit checks JSON/JSONL results for secret-like strings and exact gold-answer strings appearing in model-visible fields such as prompts, memory context, actions, observations, and rendered memory text.

Manual review is still required for representative trajectories. In particular, fields such as `proof_obligation`, `recompute_operator`, and `next_reasoning_move` should be described as HLE adapter-local verifier policy when they appear in HLE runs.

## Local run

```bash
export EMMA_OPENAI_API_KEY=...
export EMMA_EMBEDDING_API_KEY=...
export EMMA_OPENAI_BASE_URL=https://api.openai.com/v1
export EMMA_EMBEDDING_BASE_URL=https://api.openai.com/v1

python environment/run_memrl_benchmark.py \
  --benchmark hle \
  --condition full \
  --episodes 5 \
  --start-index 0 \
  --results-dir environment/hle/results/smoke
```

## Safe Docker helpers

Two wrapper scripts are included so the solver model is explicit and expensive solvers are blocked unless you opt in:

- `environment/hle/run_hle_docker_smoke.sh`: single-run smoke test
- `environment/hle/run_hle_docker_matrix.sh`: 4-run mini matrix (`full/no_memory` x `local/llm_judge`)

Minimal smoke example:

```bash
export EMMA_OPENAI_API_KEY=...
export EMMA_EMBEDDING_API_KEY=...
export EMMA_OPENAI_BASE_URL=https://api.openai.com/v1
export EMMA_EMBEDDING_BASE_URL=https://api.openai.com/v1
export EMMA_OPENAI_MODEL=gpt-4o-mini
export EMMA_HLE_JUDGE_MODE=local_exact_match

bash environment/hle/run_hle_docker_smoke.sh
```

`gpt-4o-mini` here is a smoke default, not the paper-primary HLE solver.

Mini matrix example:

```bash
export EMMA_OPENAI_API_KEY=...
export EMMA_EMBEDDING_API_KEY=...
export EMMA_OPENAI_BASE_URL=https://api.openai.com/v1
export EMMA_EMBEDDING_BASE_URL=https://api.openai.com/v1
export EMMA_OPENAI_MODEL=gpt-4o-mini
export EMMA_HLE_JUDGE_MODEL=gpt-4o-2024-08-06

bash environment/hle/run_hle_docker_matrix.sh
```

The matrix helper is intended for protocol checks. For paper-quality HLE runs, set `EMMA_OPENAI_MODEL` to the chosen frontier solver and report the solver, judge, protocol, and route separately.

If you intentionally want an expensive solver such as Gemini or GPT-5, you must add:

```bash
export EMMA_ALLOW_EXPENSIVE_SOLVER=1
```

Without that flag, the script exits before any benchmark call is made.

## Model Preflight

Before using a new secondary solver through an OpenAI-compatible route, test whether the current provider can actually serve it:

```bash
export EMMA_OPENAI_API_KEY=...
export EMMA_OPENAI_BASE_URL=https://api.openai.com/v1

python environment/hle/preflight_models.py \
  --models gpt-4o-2024-08-06 gpt-5-mini gemini-3-pro-preview o3
```

The preflight tests both `chat.completions` and `responses.create` by default. It classifies each candidate into statuses such as:

- `chat_ok`
- `responses_ok`
- `empty_output`
- `model_not_found`
- `model_price_error`
- `invalid_request`

## Judge mode

The default judge is local exact-match:

```bash
export MEMRL_HLE_JUDGE_MODE=local_exact_match
```

To enable an LLM equivalence judge, opt in explicitly:

```bash
export MEMRL_HLE_JUDGE_MODE=llm_judge
export MEMRL_HLE_JUDGE_MODEL=gpt-4o-2024-08-06
```

Solver, judge, and embedding should be reported separately:

- solver: `llm.model_name` and can be overridden by `MEMRL_OPENAI_MODEL`
- judge: `MEMRL_HLE_JUDGE_MODEL` when `MEMRL_HLE_JUDGE_MODE=llm_judge`
- embedding: `encoder.model_name` and can be overridden by `MEMRL_EMBEDDING_MODEL`

## Protocol sanity check

Before drawing conclusions from a tiny HLE slice, run a protocol sanity check on a manually curated set of known-correct answers:

```bash
python environment/hle/protocol_sanity_check.py \
  --cases environment/hle/protocol_sanity_cases.json \
  --output-md environment/hle/results/protocol_sanity_check.md
```

To compare local exact-match with the LLM judge on the same curated answers:

```bash
export MEMRL_HLE_JUDGE_MODE=llm_judge
export MEMRL_HLE_JUDGE_MODEL=gpt-4o-2024-08-06

python environment/hle/protocol_sanity_check.py \
  --cases environment/hle/protocol_sanity_cases.json \
  --include-llm-judge \
  --output-json environment/hle/results/protocol_sanity_check.json \
  --output-md environment/hle/results/protocol_sanity_check.md
```

Use this harness only to detect protocol contamination:

- answer extraction bugs
- local exact-match false negatives on symbolic families
- judge disagreement between `local_exact_match` and `llm_judge`

Do not treat the sanity set as benchmark evidence for EMMA itself.

## Official HLE override

If you already have legitimate access to the official gated dataset in the runtime:

```bash
export EMMA_HLE_DATASET_NAME=cais/hle
export EMMA_HLE_SPLIT=test
```

If you instead have a local export compatible with the original MemRL HLE schema:

```bash
export EMMA_HLE_DATASET_PATH=/abs/path/to/test-00000-of-00001.parquet
```

## Docker run

```bash
docker build -f environment/hle/Dockerfile -t memrl-hle .

docker run --rm \
  -e EMMA_OPENAI_API_KEY \
  -e EMMA_EMBEDDING_API_KEY \
  -e EMMA_OPENAI_BASE_URL=https://api.openai.com/v1 \
  -e EMMA_EMBEDDING_BASE_URL=https://api.openai.com/v1 \
  -e EMMA_HLE_DATASET_NAME=koiwave/hle-short \
  -v "$PWD/environment/hle/results:/workspace/environment/hle/results" \
  memrl-hle \
  --condition full \
  --episodes 5 \
  --start-index 0 \
  --results-dir /workspace/environment/hle/results/docker_smoke
```

`MEMRL_*` env names remain supported as compatibility aliases, but `EMMA_*` is now the preferred interface.

## Docker dependency note

The HLE image intentionally installs only the minimal shared-runner dependencies needed by:

- `environment/run_memrl_benchmark.py`
- `environment/benchmark_core/*`
- `environment/hle/*`
- `exp/modules/{encoder,llm_core,memory,retriever,rl_optimizer}.py`

It does not install the full `exp/requirements.txt`, because unrelated packages there can block the image build without affecting HLE execution.
