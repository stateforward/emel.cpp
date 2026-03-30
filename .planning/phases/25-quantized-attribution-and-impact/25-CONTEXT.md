# Phase 25: Quantized Attribution And Impact - Context

**Gathered:** 2026-03-25
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 25 must publish maintained benchmark attribution that reflects the now-proven canonical ARM
runtime contract honestly. The benchmark/docs surfaces should explain that the supported workload
stays on the approved `8/4/0/0` contract, show the remaining cost honestly, and preserve the
existing warning-only regression policy unless the user explicitly changes it.

This phase stays inside `tools/bench`, snapshot/docs publication, and related compare artifacts. It
must not change Boost.SML actor structure, public C API boundaries, or update stored snapshots
without explicit user approval.

</domain>

<decisions>
## Implementation Decisions

### Locked From Phase 24
- The maintained canonical ARM workload now proves the approved `8/4/0/0` runtime contract across
  `1/10/100/1000`.
- Benchmark gating remains warning-only for compare regressions under `scripts/quality_gates.sh`.
- The supported canonical workload still has approved dense-f32-by-contract seams; benchmark
  attribution must not overstate the path as “fully quantized everywhere”.

### Phase 25 Question
- Promote the Phase 24 runtime contract into maintained compare output and docs so benchmark
  publication explains both the approved contract and the next remaining bottleneck honestly.
- Stop before any snapshot/docs regeneration until explicit user approval is given, because the
  repo forbids snapshot updates without consent.

### Guardrails
- Do not change Boost.SML transition tables or actor ownership without explicit user approval.
- Keep Phase 25 anchored to `tools/bench`, `snapshots/bench`, and generated docs publication.
- Preserve warning-only benchmark drift policy unless the user explicitly asks to change it.

</decisions>

<canonical_refs>
## Canonical References

- `.planning/ROADMAP.md` — Phase 25 goal and success criteria.
- `.planning/REQUIREMENTS.md` — `BENCH-10`.
- `.planning/PROJECT.md` — current milestone focus after Phase 24.
- `.planning/phases/24-quantized-path-proof-and-regression/24-VERIFICATION.md` — the published
  `8/4/0/0` proof surface that benchmark attribution should now reference.
- `AGENTS.md` — explicit consent is required before snapshot updates.

</canonical_refs>

<code_context>
## Existing Code Insights

- `tools/bench/bench_main.cpp` already validates and publishes flash and quantized dispatch
  evidence through compare output, but it does not yet publish the Phase 24 runtime contract
  counts.
- `tools/bench/generation_bench.cpp` already captures generation-side dispatch evidence from the
  shipped generator wrapper and is the likely place to thread benchmark-time runtime contract
  counts if needed.
- `snapshots/bench/benchmarks_compare.txt` and `docs/benchmarks.md` currently publish dispatch
  evidence plus compare rows, but they do not yet explain the approved dense-f32-by-contract seams
  or the next bottleneck through the Phase 24 proof language.

</code_context>

<specifics>
## Specific Ideas

- Add maintained compare output fields that publish the shipped runtime contract alongside the
  existing quantized dispatch evidence.
- Refresh the generated docs language so the benchmark story references the approved `8/4/0/0`
  contract and the current warning-only regressions honestly.
- Split implementation from publication: code and local proof first, then snapshot/docs refresh
  only after explicit approval.

</specifics>

<deferred>
## Deferred Ideas

- Any benchmark-gate policy change beyond the current warning-only tolerance remains out of scope
  unless the user explicitly asks for it.
- Broader model-matrix or non-canonical benchmark publication remains outside this milestone.

</deferred>

---
*Phase: 25-quantized-attribution-and-impact*
*Context gathered: 2026-03-25*
