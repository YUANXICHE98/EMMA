# Reproducibility Notes

## Expected Reproducibility

This repository is released to make the EMMA mechanism inspectable and runnable:

- shared memory loop
- structured memory substrate
- retrieval into the prompt
- environment verdict to memory update
- value/Q update
- failure and cautionary memory
- benchmark-local adapters
- HLE protocol and leakage audits

The default smoke commands are not intended to reproduce the exact paper table numbers. They check that the mechanism and benchmark protocol are wired correctly.

See `docs/MODEL_PROTOCOL.md` for the benchmark-specific model roles used to separate paper-primary runs from legacy baselines, scaling analysis, and low-cost smoke tests.

## Why Exact Scores Can Differ

LLM benchmark scores are sensitive to runtime details that are not fully controlled by source code:

- hosted model versions can change behind stable model names
- OpenAI-compatible providers can route the same model name differently
- `chat.completions` and `responses` protocols can return different output shapes
- gated datasets may be unavailable or locally cached at different revisions
- local exact-match and LLM judge modes are not equivalent
- retries, timeouts, max-token limits, and rate-limit behavior affect long runs
- high-capability and low-cost solvers have different ceilings

For that reason, third-party runs should be compared as protocol-matched replications, not as byte-identical reruns.

## Minimum Reporting Fields

When reporting a result, include:

- benchmark and split
- condition, such as `full`, `no_memory`, or another ablation
- number of episodes and start index or dataset ids
- solver model, endpoint/provider, and API protocol
- judge mode and judge model, if applicable
- embedding model and endpoint/provider
- temperature, max tokens, retry count, timeout
- git commit hash
- whether memories were empty, warm-started, or carried across episodes

## Paper Number Replication

The paper experiments used a fixed internal run configuration and contemporaneous model/provider routes. Reproducing the qualitative EMMA effect should use the same ablation protocol and comparable solver capacity. Reproducing exact absolute scores requires matching the dataset revision, model route, judge mode, and budget settings.

If a run uses a cheaper or weaker solver, lower absolute scores are expected. The mechanism should be evaluated by the relative difference between matched conditions and by inspecting whether retrieved memory is actually rendered into the model context.

The recommended paper wording is:

> Main EMMA results use benchmark-appropriate frontier solvers. GPT-4-0125-preview is retained as a legacy comparable baseline. GPT-4o-mini-class and DeepSeek-V3-class models are used only for low-cost sanity checks or cost-efficiency analysis unless explicitly reported.

## HLE-Specific Notes

HLE is a closed-ended reasoning benchmark. The adapter can emit verifier feedback such as arithmetic checks or proof obligations. These fields are adapter-local audit evidence that records why a previous answer was rejected; they are not shared EMMA logic.

Before publishing HLE artifacts:

```bash
python environment/hle/open_source_audit.py path/to/results \
  --output path/to/results/open_source_audit.json
```

Also run:

```bash
python environment/hle/protocol_sanity_check.py \
  --cases environment/hle/protocol_sanity_cases.json

python environment/hle/numeric_boundary_sanity_check.py
```

These checks look for answer extraction bugs, judge disagreements, and accidental leakage of evaluator-only answers into model-visible fields.
