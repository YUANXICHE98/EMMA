# Paper Alignment

This file maps the paper narrative to code surfaces in this release.

## Shared EMMA Mechanism

- Retrieval and prompt integration: `exp/modules/retriever.py`
- LLM call wrapper and output extraction: `exp/modules/llm_core.py`
- Memory persistence: `exp/modules/memory.py`
- Value update: `exp/modules/rl_optimizer.py`
- Shared benchmark loop: `environment/benchmark_core/brain.py`
- Hypergraph-style memory substrate: `environment/benchmark_core/hypergraph.py`

## Benchmark Adapters

Adapters translate benchmark episodes into the shared EMMA interface. They should not own the central memory algorithm.

- HLE: `environment/hle/adapter.py`
- BigCodeBench: `environment/bigcodebench/adapter.py`
- ScienceWorld: `environment/scienceworld/adapter.py`
- InterCode-SQL: `environment/intercode_sql/adapter.py`
- ALFWorld: `environment/alfworld/adapter.py`

## Interpretation Boundary

Benchmark-local verifier outputs are audit signals. Shared EMMA should be evaluated by whether memory is retrieved, rendered, updated, consolidated, and used across matched ablation conditions.

Exact paper scores are configuration-dependent. Release users should expect matched-protocol trends to be more meaningful than matching a single absolute score.
