# Phase 117: Whisper Compare Failure Contract Repair - Context

**Gathered:** 2026-04-27
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase repairs the maintained Whisper compare contract so transcript drift is a hard failure
for the compare wrapper and quality-gate path. It does not change Whisper runtime arithmetic,
model binding, tokenizer ownership, or benchmark performance policy.

</domain>

<decisions>
## Implementation Decisions

### Compare Failure Contract
- Treat `exact_match` as the only successful compare summary status.
- Preserve the existing machine-readable `bounded_drift` status and `transcript_mismatch` reason
  for drift diagnostics.
- Return a nonzero process exit for `bounded_drift`, lane errors, and malformed compare outputs.
- Add shell-backed doctest coverage using fake EMEL/reference runners before changing behavior.

### Rule Constraints
- No SML machine changes are expected in this phase.
- Do not introduce model-family roots or generic Whisper runtime contracts.
- Keep benchmark and parity lanes isolated.

### the agent's Discretion
Implementation details for test helper reuse and exact exit-code routing are at the agent's
discretion as long as the maintained no-drift contract is enforced.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/bench/whisper_benchmark_tests.cpp` already builds fake EMEL/reference runners.
- `tools/bench/whisper_compare.py` already emits `bounded_drift` and `transcript_mismatch`.
- `scripts/quality_gates.sh` trusts the `scripts/bench_whisper_compare.sh` exit code.

### Established Patterns
- Whisper tool tests are doctest C++ tests that shell out to Python wrappers with temp fixtures.
- Compare/benchmark publication writes machine-readable JSON summaries under `build/`.

### Integration Points
- `tools/bench/whisper_compare.py`
- `tools/bench/whisper_benchmark_tests.cpp`
- `tools/bench/CMakeLists.txt`

</code_context>

<specifics>
## Specific Ideas

Use fake runners to prove transcript mismatch fails without depending on real model/audio assets.

</specifics>

<deferred>
## Deferred Ideas

Public actor-interface harness repair remains Phase 118.

</deferred>
