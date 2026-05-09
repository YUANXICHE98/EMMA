# Model Protocol

This release separates three model roles:

- **paper primary**: models used for main paper-quality runs
- **legacy baseline**: older GPT-4-class runs retained for comparison
- **release smoke**: low-cost defaults used only to check that the code path works

The code defaults to low-cost smoke settings where possible. Those defaults are not paper-primary settings.

## Recommended Paper-Quality Matrix

| Benchmark | Paper-primary solver | Legacy / baseline solver | Scaling / cost-only solvers | Notes |
| --- | --- | --- | --- | --- |
| HLE | `gpt-5.2` or stronger frontier reasoning model | `gpt-4-0125-preview` | `qwen2.5-32b-instruct`, DeepSeek-V3-class, GPT-4o-mini-class | HLE is solver-ceiling sensitive. Use weak/cheap solvers only for ceiling and cost analysis. |
| BigCodeBench | `gpt-5.2-codex` or strongest available coding model | `gpt-4-0125-preview`, GPT-4.1-class | Qwen-32B-coder-class, DeepSeek-V3-class | Code benchmarks should use coding-capable solvers for main claims. |
| SWE-bench | `gpt-5.2-codex` or strongest available coding agent model | GPT-4.1-class, Claude Sonnet-class | Open coding models | Run only in an isolated container or sandbox. |
| ALFWorld | `gpt-5.2` or frontier general agent model | `gpt-4-0125-preview` | Llama-3.1 8B/70B, Qwen-2.5 14B/32B | Use matched conditions to measure memory effects. |
| ScienceWorld | `gpt-5.2` or frontier science/reasoning model | `gpt-4-0125-preview` | Llama-3.1 8B/70B, Qwen-2.5 14B/32B | General reasoning and environment interaction benchmark. |
| LLB-OS / LifelongAgentBench | `gpt-5.2` or frontier agent model | `gpt-4-0125-preview` | Llama/Qwen scaling models | Keep official task boundary unchanged. |
| InterCode-SQL / LLB-DB | `gpt-5.2` or strongest SQL-capable frontier model | `gpt-4-0125-preview`, GPT-4.1-class | Qwen-32B-class | Report DB engine, dataset split, and judge/evaluator mode. |

## Naming Rules For The Paper

Use one consistent convention:

- Main results: **frontier primary model**, benchmark-specific as listed above.
- GPT-4 legacy: `gpt-4-0125-preview` is a historical baseline, not the current primary solver.
- GPT-4o-mini-class models: release smoke and low-cost sanity only.
- DeepSeek-V3-class models: cost-efficiency or secondary validation only unless an explicit benchmark table reports them.
- Llama/Qwen: scaling analysis only.

Avoid saying "GPT-4 and GPT-4o-mini served as the primary models" unless both are actually reported as primary in the same main table. Prefer:

> Main EMMA results use benchmark-appropriate frontier solvers. GPT-4-0125-preview is retained as a legacy comparable baseline. GPT-4o-mini-class and DeepSeek-V3-class models are used only for low-cost sanity checks or cost-efficiency analysis unless explicitly reported.

## Code Defaults

The release default `EMMA_OPENAI_MODEL=gpt-4o-mini` examples are smoke-test defaults. They are chosen to keep public setup inexpensive and to validate:

- adapter loading
- prompt construction
- memory retrieval/rendering
- value update
- result serialization

They are not expected to reproduce paper-level absolute benchmark scores.
