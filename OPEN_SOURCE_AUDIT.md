# Open Source Audit

## Scope

- Project: MemRL / EMMA
- Main open-source risk area: HLE closed-ended reasoning path
- Goal: remove claims that look like benchmark hacks, leakage, or unreproducible provider coupling

## Must Verify

- No `gold_answer` or evaluator-only answer is rendered into prompts, memories, logs, or trajectories.
- No API keys, tokens, or provider secrets appear in source, result files, memory files, or notes.
- HLE adapter logic is clearly separated from core EMMA memory/retrieval logic.
- Benchmark-local verifier rules are documented as adapter-side audits, not core algorithm claims.
- Local exact-match, LLM judge, and symbolic normalization behavior are described separately.
- All result directories are labeled with model, judge mode, dataset path, and offline/online mode.
- Any proxy or base URL setting is opt-in and documented.

## Likely Reviewer Questions

1. Is EMMA just prompt engineering?
2. Are HLE verifier rules benchmark-specific hacks?
3. Does the method depend on strong models to work at all?
4. Can third parties reproduce the same run without hidden env state?
5. Are symbolic false negatives judge artifacts or solver artifacts?

## Required Evidence Package

- One diagram of the EMMA loop: retrieval -> prompt -> action -> environment verdict -> structured memory.
- One table of benchmark adapters and what they are allowed to inspect.
- One table of model / judge / protocol settings for each reported result.
- One small set of raw trajectories showing failure boundary + repair prompt.
- One audit note explaining why HLE verifier feedback is environment-side, not gold leakage.

## Cleanup Before Release

- Remove or redact all secrets from `logs/`, `results/`, `notes/`, and `session_memory.md`.
- Delete or isolate scratch result folders.
- Keep only curated release results.
- Pin dataset access mode and required cache paths.
- Freeze dependency versions and provider config examples.

## HLE-Specific Positioning

- HLE is a closed-ended reasoning benchmark.
- The open-source contribution is the memory/verifier loop, not a claim that HLE-specific math rules are general EMMA logic.
- `verifier_check`, `verifier_failure_summary`, `verifier_repair_constraint`, and `proof_obligation` are the evidence bridge between environment verdicts and reusable memory.
- Historical audit note: `reasoning_failure_pattern`, `recompute_operator`, `next_reasoning_move`, and `proof_obligation` were previously flagged as HLE-specific tutoring-risk fields. If kept, describe them as HLE adapter-local verifier policy, not as shared EMMA core behavior.

## Automated Checks

Run the lightweight HLE artifact audit before packaging any HLE result directory:

```bash
python environment/hle/open_source_audit.py environment/hle/results/RELEASE_CANDIDATE \
  --output environment/hle/results/RELEASE_CANDIDATE/open_source_audit.json
```

The audit scans JSON/JSONL artifacts for:

- secret-like strings such as provider keys
- exact `gold_answer` / `answer` values appearing inside model-visible fields such as `prompt`, `memory_context`, `action`, `obs`, `e`, or `embedding_text`

This is a conservative exact-string check. Passing it does not replace manual review of representative trajectories.

## Release Gate

- [ ] No secrets in repo
- [ ] No gold leakage in prompt/memory
- [ ] HLE adapter docs separate from EMMA core docs
- [ ] Protocol sanity passes
- [ ] Numeric boundary sanity passes
- [ ] `environment/hle/open_source_audit.py` passes on release-candidate HLE artifacts
- [ ] Representative HLE trajectories archived
- [ ] Result tables include model/judge/protocol
- [ ] Open questions and limitations documented
