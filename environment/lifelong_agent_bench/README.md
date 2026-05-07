`lifelong_agent_bench` is wired as a thin adapter over the official task boundary.

Current scope:
- supported task: `knowledge_graph`
- main MemRL path: shared brain + official task environment
- excluded from main path: official `previous_sample_utilization` and `group_self_consistency`

Recommended experiment naming:
- `official_standard`: official benchmark prompt/task boundary only
- `official_previous_sample_utilization`: official flat replay-style memory baseline
- `official_group_self_consistency`: official lifelong-learning baseline
- `memrl_shared_brain`: our main method on the same task boundary

Local setup:
- clone the official repo and set `runner.lab_repo_path` or `MEMRL_LAB_REPO_PATH`
- ensure the official Python dependencies are installed
- for `knowledge_graph`, ensure the official data files and SPARQL endpoint are available

If the official `knowledge_graph` processed files are missing, generate the minimum local layout from the released parquet:

```bash
python environment/lifelong_agent_bench/prepare_knowledge_graph_data.py \
  --parquet /tmp/memrl_lab/lab_kg.parquet \
  --entry-dict-out /path/to/LifelongAgentBench/data/v0303/knowledge_graph/processed/grailqa/v0417_tl2sc50_tl3sc50_tl4sc50_tl5sc50_tl6sc50_tl7sc50_tl8sc50_tl9sc46/entry_dict.json \
  --ontology-dir-out /path/to/LifelongAgentBench/data/v0121/knowledge_graph/ontology
```

This adapter intentionally does not add benchmark-specific controller logic. It only:
- resets an official sample
- serializes the current chat/task state into an observation
- passes one LLM-produced action string back to the official task
- converts the final official metrics into MemRL reward/success fields
