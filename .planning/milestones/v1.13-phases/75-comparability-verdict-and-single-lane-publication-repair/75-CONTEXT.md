# Phase 75: Comparability Verdict And Single-Lane Publication Repair - Context

**Gathered:** 2026-04-21
**Status:** Ready for planning
**Mode:** Autonomous smart discuss

<domain>
## Phase Boundary

Close the `v1.13` audit blockers where non-comparable publication was only proven by synthetic
JSONL input and compare verdicts did not consume all material workload contract fields before
publishing `exact_match`, `bounded_drift`, or `non_comparable`.

This phase is bounded to generation benchmark tooling, workload manifests, wrapper behavior,
documentation, and tests. It must not widen `src/` runtime behavior or substitute a simpler
fallback runtime path for the maintained generation lane.

</domain>

<decisions>
## Implementation Decisions

### Single-Lane Publication
- Add a maintained locally-runnable single-lane workload manifest on the existing LFM2 fixture so
  the operator-facing wrapper can prove real non-comparable publication without relying on a
  missing Gemma4 local fixture.
- Keep the single-lane workload explicitly non-parity by setting `comparison_mode` to
  `single_lane` and `comparable` to `false`.
- When a selected workload manifest is non-parity or non-comparable, run the EMEL lane, leave
  `raw/reference.jsonl` empty, and publish `non_comparable` with reason
  `single_lane_emel_workload`.

### Verdict Inputs
- Treat workload identity, fixture identity, prompt fixture, prompt id, formatter mode,
  formatter contract, sampling id, stop id, seed, and max token budget as material
  comparability metadata.
- Reject metadata mismatches as `non_comparable` before comparing output text or checksums.
- Preserve existing reason names for already-covered mismatch types where possible.

### Proof Strategy
- Add synthetic mismatch tests for previously ignored material fields.
- Add wrapper-level end-to-end coverage for a real single-lane workflow.
- Re-run `generation_compare_tests`, CTest, and the full quality gate.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/bench/generation_workload_manifest.hpp` already parses the material workload fields.
- `tools/bench/generation_bench.cpp` owns the fixed manifest list and emits
  `generation_compare/v1` records.
- `tools/bench/generation_compare.py` owns reference backend invocation and summary verdicts.
- `tools/bench/generation_compare_tests.cpp` already has synthetic JSONL tests and one real
  wrapper-level multi-engine test.

### Established Patterns
- Non-comparable workloads are truthful publication output rather than workflow failures.
- Reference backend setup belongs to the reference lane or wrapper, not the EMEL lane.
- `generation_compare_summary/v1` is the machine-readable closeout artifact for operator review.

### Integration Points
- `scripts/bench_generation_compare.sh` passes `--workload-id` through to
  `generation_compare.py`.
- The Python wrapper can inspect checked-in workload manifests before deciding whether the
  selected workload requires a reference backend run.

</code_context>

<specifics>
## Specific Ideas

- Prefer a checked-in single-lane LFM2 manifest over a Gemma4 proof because the LFM2 fixture is
  available in the maintained local test environment.
- Keep skipped reference output as an empty JSONL file rather than synthesizing a reference error
  or fake reference result.
- Document the LFM2 single-lane workload as an operator workflow proof, not as a parity claim.

</specifics>

<deferred>
## Deferred Ideas

- Full requirement evidence tables and milestone-level Nyquist closeout remain Phase 76 scope.
- The broader Gemma4 single-lane workload family remains manifest truth, but is not required for
  this local wrapper proof because the Gemma4 fixture is not present on the maintained host.

</deferred>
