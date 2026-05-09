# Model Protocol

This release separates code defaults from paper-model claims. The repository defaults are low-cost smoke settings. Paper-quality runs should use benchmark-appropriate frontier solvers and explicitly report solver, judge, provider route, API protocol, and budget.

## Model Role Table

| Role | Recommended model(s) | Paper wording |
| --- | --- | --- |
| Main experimental primary | `gpt-5.2` | EMMA main results are reported with GPT-5.2 unless otherwise specified. |
| Code benchmark primary | `gpt-5.2-codex` | Used for code-generation and software-engineering benchmarks. |
| Strong external validation | Claude Opus 4.5 / Claude Sonnet 4.5 | Used as external frontier-model validation, not the default primary. |
| Historical comparable baseline | `gpt-4-0125-preview` | Legacy GPT-4 baseline retained for comparison with prior MemRL / GPT-4-class settings. |
| Scaling analysis | Llama-3.1 8B/70B, Qwen-2.5 14B/32B | Open / smaller-model scaling analysis. |
| Cost-efficiency appendix | GPT-4.1 mini-class, DeepSeek-V3.2-Exp-class, Qwen-32B-class | Low-cost replication only; not main claims. |
| Not recommended as primary | GPT-4o-mini-class, DeepSeek-R1 | Cheap/cost baseline only; do not describe as primary. |

## Benchmark Model Plan

| Benchmark | EMMA primary model | Baseline / comparison | Notes |
| --- | --- | --- | --- |
| HLE | `gpt-5.2` or stronger frontier reasoning model; use high-reasoning settings when available | `gpt-4-0125-preview`; Claude Opus 4.5 as cross-validation | HLE is solver-ceiling sensitive. Weak/cheap solvers should be framed as ceiling or cost analysis. |
| BigCodeBench | `gpt-5.2-codex` | GPT-4.1-class, `gpt-4-0125-preview`, Claude Sonnet 4.5 | Do not use DeepSeek-R1 as the main BigCodeBench solver. |
| SWE-bench / code-agent benchmarks | `gpt-5.2-codex` | Claude Sonnet 4.5, `gpt-5.2` | Codex-class models are the most appropriate primary for software-engineering benchmarks. |
| ALFWorld | `gpt-5.2` with medium reasoning settings | `gpt-4-0125-preview`; Llama/Qwen scaling | Main signal is memory-guided action trajectories; Codex is not required. |
| ScienceWorld | `gpt-5.2` with medium/high reasoning settings | `gpt-4-0125-preview`; Qwen-32B / Llama-70B scaling | Scientific reasoning plus environment interaction. |
| LLB-OS / LifelongAgentBench | `gpt-5.2` or Claude Sonnet 4.5 | `gpt-4-0125-preview`; Qwen/Llama scaling | Long-horizon agent benchmark; emphasize memory loop under the official task boundary. |
| LLB-DB / InterCode-SQL | `gpt-5.2` | GPT-4.1-class, `gpt-4-0125-preview`, Qwen-32B-class | SQL/DB tasks emphasize instruction following and symbolic consistency. |
| WebArena / OSWorld, if retained | Claude Sonnet 4.5 or `gpt-5.2` | `gpt-4-0125-preview` | Treat as external generalization unless included in the main table. |

## Paper Wording Fixes

| Original inconsistency | Recommended wording |
| --- | --- |
| "GPT-4 class models" | "frontier API models" |
| "GPT-4 for main experiments" | "GPT-4-0125-preview is retained as a legacy baseline; main EMMA runs use GPT-5.2-family models." |
| "GPT-4 and GPT-4o-mini served as the primary models" | "GPT-4-0125-preview served as the legacy comparable baseline; GPT-4o-mini was used only for low-cost sanity/cost analysis." |
| DeepSeek-V3 appears only once as cost-efficiency validation | Either remove it, or write "cost-efficiency appendix only; not used for main claims." |
| GPT-4o-mini described as comparable quality | Avoid this. Use "lower-cost replication with expected lower absolute scores." |

## Recommended One-Sentence Policy

Use `gpt-5.2` for general/reasoning benchmarks, `gpt-5.2-codex` for code benchmarks, `gpt-4-0125-preview` as the legacy baseline, Llama/Qwen for scaling, and DeepSeek/GPT-4o-mini-class models only for the cost-efficiency appendix or smoke tests.

## Code Defaults

The release default `EMMA_OPENAI_MODEL=gpt-4o-mini` examples are smoke-test defaults. They are chosen to keep public setup inexpensive and to validate:

- adapter loading
- prompt construction
- memory retrieval/rendering
- value update
- result serialization

They are not expected to reproduce paper-level absolute benchmark scores.
