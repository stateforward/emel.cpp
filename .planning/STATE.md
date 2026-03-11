---
gsd_state_version: 1.0
milestone: v1.1
milestone_name: tools/bench the llama-68M generation against llama.cpp
status: ready_for_new_milestone
stopped_at: Milestone v1.1 archived; next recommended action is $gsd-new-milestone
last_updated: "2026-03-11T01:16:11Z"
last_activity: 2026-03-11 — Archived milestone v1.1 and prepared the planning surface for the next milestone
progress:
  total_phases: 0
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-08)

**Core value:** Prove real end-to-end behavior with explicit SML orchestration and parity-oriented
verification before widening API surface or model scope.
**Current focus:** Start the next milestone cleanly with `$gsd-new-milestone`

## Current Position

Phase: No active milestone
Plan: Waiting for next milestone definition
Status: Ready for `$gsd-new-milestone`
Last activity: 2026-03-11 — Archived v1.1 after a passing milestone audit and local release tag

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Previous milestone velocity:**
- Total plans completed: 15
- Average duration: 17 min
- Total execution time: 4.3 hours

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Initialization: Use `tests/models/Llama-68M-Chat-v1-Q2_K.gguf` as the first end-to-end target.
- Initialization: Define done in `tools/paritychecker/`, not through a new public C API example.
- Initialization: Keep the roadmap as a narrow vertical slice with only minimum supporting changes.
- Phase 1.1 insertion: Implement the real GGUF loader before continuing generation-path work.
- Phase 6 closure: Enforce canonical fixture selection to close the v1.0 audit gap in `HARN-02`.
- v1.0 archive: Reset the live planning surface and keep milestone history under
  `.planning/milestones/`.
- v1.1 kickoff: Keep the new milestone narrow to one truthful `tools/bench` generation benchmark
  for the canonical Llama-68M fixture rather than broadening into multi-model benchmarking.
- v1.1 research: Reuse the proven generation slice and existing `bench_runner` / `scripts/bench.sh`
  surfaces instead of creating a benchmark-only workflow.
- [Phase 07]: Bench harness now caches the canonical Llama-68M fixture outside the timed loop and
  rebuilds request-local generation state per iteration.
- [Phase 07]: Generation bench cases stay bounded via dedicated generation env overrides on top of
  the shared bench config.
- [Phase 07]: The measurement contract is published directly in the case name as
  `generation/preloaded_request/...` so users can tell what segment is being timed from the normal
  bench output.
- [Phase 07.1 planning]: The correction must replace both the EMEL-side reference decode callbacks
  and the dummy topology/plan placeholders before any compare numbers are considered truthful.
- [Phase 07.1 replan]: The first missing substrate is now explicit milestone work: typed Llama
  tensor views, canonical topology and step-plan descriptors, then native callback wiring on top.
- [Phase 07.1-02]: The canonical Llama slice now has typed execution views plus bounded topology
  and step-plan descriptors in `src/emel/model/llama/detail.hpp`.
- [Phase 07.1-03]: The shared native generation backend lives under `tools/` and paritychecker now
  uses it for the EMEL path while retaining a separate explicit reference path.
- [Phase 07.1-04]: `tools/bench` now reuses that same native backend, and seam auditing is env-gated
  so compare output stays compatible with the existing benchmark scripts.
- [Phase 08 planning]: Keep Phase 8 to a narrow compare-surface phase: first harden canonical row
  pairing in `bench_runner --mode=compare`, then publish stable compare evidence through the normal
  `scripts/bench.sh --compare` flow.
- [Phase 08]: Share the canonical generation case name between the benchmark registry and compare
  runner, fail compare mode on duplicate or missing canonical generation rows, and route compare
  build chatter to stderr so stdout stays usable as published compare evidence.
- [Phase 09 planning]: Keep the final milestone phase narrow to two plans: existing compare-flow
  runbook docs first, then snapshot/docs closure through the existing update tooling with explicit
  user approval before any baseline refresh.
- [Phase 09-01]: Publish the canonical Llama-68M generation benchmark as a docs-first runbook via
  the normal `scripts/bench.sh --compare` surface and keep README changes to a discovery pointer.
- [Phase 09-02]: Refresh the benchmark snapshots and generated docs only through the existing
  approved `scripts/bench.sh` + `scripts/generate_docs.sh` path.

### Roadmap Evolution

- Phase 1.1 inserted after Phase 1 during v1.0: implement the gguf::loader that's just stubbed
- Phase 6 added during v1.0 after milestone audit: Fixture Contract Hardening
- New milestone v1.1 starts at Phase 7
- Phase 07.1 inserted after Phase 7 during v1.1: Replace the reference-backed decode path with
  native EMEL decode before any further benchmark work (URGENT)

### Pending Todos

None yet.

### Blockers/Concerns

- The exact low-iteration `scripts/bench.sh --snapshot --compare` check remains noisy under the
  repo's default 10% tolerance and still reports unrelated regressions on this machine.
- `gsd-tools init milestone-op` resolved the current milestone incorrectly during audit; use an
  explicit version when running milestone closeout commands.
- `scripts/quality_gates.sh` still treats benchmark snapshot regression drift as non-blocking repo
  policy.
- Tool-local fixture setup still derives tokenizer/vocab metadata from the explicit `llama.cpp`
  reference model before initialize; decode and logits extraction are now native on the EMEL path.
- `.gitignore`, generated architecture docs, snapshot timing output, and
  `tests/generator/action_guard_tests.cpp` remain as unrelated workspace changes outside this
  milestone work.

## Session Continuity

Last session: 2026-03-11T01:16:11Z
Stopped at: Milestone v1.1 archived; next recommended action is $gsd-new-milestone
Resume file: None
