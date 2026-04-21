# Phase 74: Generation Compare Lane Isolation Repair - Context

**Gathered:** 2026-04-21
**Status:** Ready for planning
**Mode:** Autonomous smart discuss

<domain>
## Phase Boundary

Repair the `v1.13` audit blocker where `--mode=emel` JSONL output is routed through
`generation_lane_mode::compare`, causing EMEL result publication to prepare reference fixture
state and run reference stage probing. This phase is limited to the maintained bench tooling
surface and must not widen `src/` runtime behavior.

</domain>

<decisions>
## Implementation Decisions

### Lane Ownership
- Keep `--mode=emel` owned by the EMEL fixture path even when
  `EMEL_GENERATION_BENCH_FORMAT=jsonl` is enabled.
- Keep `--mode=reference` owned by the reference fixture path even when JSONL output is enabled.
- Preserve `--mode=compare` text behavior as the only local combined EMEL/reference fixture mode
  that may capture stage probes.
- Do not add new reference-backend setup into the EMEL lane to satisfy compare publication.

### Proof Strategy
- Add focused bench runner coverage that proves EMEL JSONL output exposes EMEL-owned records,
  including any maintained single-lane records available on the host.
- Keep the existing reference JSONL coverage proving the reference lane still emits comparable
  reference records.
- Re-run the focused bench runner tests and generation compare tests after the repair.

### the agent's Discretion
Implementation details are at the agent's discretion as long as lane ownership is explicit,
bounded to `tools/bench`, and the operator compare workflow remains compatible.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/bench/bench_main.cpp` owns CLI mode selection and JSONL printing.
- `tools/bench/generation_bench.cpp` owns EMEL, reference, and compare fixture preparation.
- `tools/bench/bench_runner_tests.cpp` already exercises `--mode=emel` and `--mode=reference`
  JSONL output through the built `bench_runner`.

### Established Patterns
- Bench tooling keeps lane-local records in `bench::result` and prints
  `generation_compare/v1` JSONL through `print_generation_jsonl`.
- `generation_lane_mode::emel`, `reference`, and `compare` already exist; the bug is incorrect
  mode selection for JSONL, not missing lane vocabulary.

### Integration Points
- `scripts/bench_generation_compare.sh` calls `tools/bench/generation_compare.py`, which invokes
  the runner separately for EMEL and reference raw JSONL.
- Phase 75 will handle real single-lane operator publication and deeper verdict checks; this
  phase only restores lane isolation.

</code_context>

<specifics>
## Specific Ideas

- Prefer changing CLI lane-mode selection over splitting or duplicating fixture preparation.
- Keep the repair inside `tools/bench` and tests.

</specifics>

<deferred>
## Deferred Ideas

- Real maintained single-lane non-comparable publication is deferred to Phase 75.
- Requirement evidence and Nyquist artifact backfill is deferred to Phase 76.

</deferred>
