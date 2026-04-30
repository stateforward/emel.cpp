# Phase 128: Whisper Benchmark And Closeout Evidence Cleanup - Context

**Gathered:** 2026-04-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Close non-blocking v1.16 audit debt around Whisper benchmark closeout stability and stale
historical closeout prose. This phase does not reopen active requirements; it improves the
evidence path before archive/tag confirmation.

</domain>

<decisions>
## Implementation Decisions

### Evidence Stability
- Stabilize the default Whisper single-thread closeout path by making the maintained wrapper and
  benchmark driver default to the current 20-iteration sample.
- Preserve explicit `--iterations` and `EMEL_WHISPER_BENCH_ITERATIONS` overrides for focused or
  diagnostic runs.
- Keep material performance-regression failures hard; the benchmark may use a small explicit
  process-wall noise tolerance for closeout evidence.
- Record the 3-iteration noise as historical audit debt, not as an active requirement blocker.

### Historical Artifact Truth
- Mark Phase 122 and Phase 125 closeout statements as superseded by Phases 126-127 and the latest
  tech-debt audit.
- Do not erase historical commands or evidence from those artifacts.
- Keep Phase 127 as the active source-backed closeout truth.
- Update planning state only as evidence cleanup; active requirement traceability remains complete.

### the agent's Discretion
All implementation details may follow existing benchmark and planning-artifact conventions so long
as the maintained compare and benchmark contracts remain source-backed.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/bench_whisper_single_thread.sh` controls closeout wrapper defaults and forwards
  warmup/iteration values into the Python benchmark driver.
- `tools/bench/whisper_benchmark.py` owns benchmark summary generation and hard-fail reasons.
- `tools/bench/whisper_benchmark_tests.cpp` already covers model mismatch, transcript mismatch,
  warmup errors, missing reference transcript, slower EMEL mean, deterministic reference policy,
  and public runtime metadata.

### Established Patterns
- Whisper benchmark wrappers publish JSON summaries and fail nonzero for lane errors, transcript
  drift, model mismatch, and performance regression.
- Planning artifacts preserve superseded evidence with explicit prose instead of deleting history.
- Quality gates select the Whisper benchmark through
  `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare:whisper_single_thread`.

### Integration Points
- Default benchmark stability affects `scripts/quality_gates.sh` when it invokes
  `scripts/bench_whisper_single_thread.sh`.
- Phase 122 and Phase 125 artifacts are historical closeout attempts; Phase 127 and the current
  audit own active closeout truth.

</code_context>

<specifics>
## Specific Ideas

- Prefer 20 measured iterations plus an explicit 2% process-wall tolerance as the default closeout
  policy because Phase 128 reproduced sub-percent benchmark noise after a noisy 10-iteration
  failure.

</specifics>

<deferred>
## Deferred Ideas

None.

</deferred>
