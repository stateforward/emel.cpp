---
gsd_state_version: 1.0
milestone: v1.25
milestone_name: I/O Read Loading Strategy
status: shipped
stopped_at: Completed Phase 226 Read Batch Cap And Closeout Evidence Refresh
last_updated: "2026-05-06T18:44:38.175Z"
last_activity: 2026-05-06
progress:
  total_phases: 16
  completed_phases: 16
  total_plans: 21
  completed_plans: 21
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-05)

**Core value:** Prove real end-to-end behavior with explicit SML orchestration and
parity-oriented verification before widening API surface or model scope.
**Current focus:** v1.25 shipped and archived — ready to define the next milestone.
Phase 226 closed the refreshed audit tech debt by independently capping the public
`io/read` batch API and refreshing closeout evidence so historical dyld fallback notes
are distinguished from current direct focused CTest evidence.

## Current Position

Milestone: v1.25 I/O Read Loading Strategy — SHIPPED
Plan: complete
Status: Milestone complete — PR #84 open for review
Phase 226 closed `.planning/v1.25-MILESTONE-AUDIT.md` tech debt: public
`read_tensor_batch` now has an independent span cap, and closeout evidence records both
historical dyld fallback wording and current focused CTest pass evidence.
Last activity: 2026-05-06

Progress: [##########] 100%

## Performance Metrics

**Latest audited milestone:** `v1.25 I/O Read Loading Strategy`

- v1.25 cleanup-only Phase 226 is complete. 13/13 v1.25 requirements remain
  mapped and satisfied; the current milestone audit is `passed`.

- v1.24 shipped on 2026-05-04 after Phase 210 closing full-scope quality gate passed with
  no override. 13/13 v1.24 requirements satisfied (MMAP-01..03, TIO-01..03, PLAT-01,
  LIFE-01, ERR-01, VAL-01..04).

- v1.23 shipped on 2026-05-04 after final source-backed audit passed with 15/15 active
  requirements satisfied.

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting this work (v1.25 scope):

- `v1.25` implements only the read/copy strategy under `src/emel/io/read`; mmap
  (v1.24) remains untouched at the runtime level.

- `model/tensor` continues to own tensor load, bind, evict, and residency semantics;
  the read strategy never takes residency ownership.

- The read strategy operates on a caller-provided owned target buffer and never owns
  tensor residency. Phase 214.1 repaired the Phase 214 evidence to use a caller-provided
  source span plus explicit source-result errors, so no filesystem call or kernel handle
  lifetime exists inside SML dispatch.

- Staged/chunked constrained-memory read policy, cooperative async loading, device-
  specific strategies, model-family widening, loader-owned byte access, and tool-only
  read scaffolds are out of scope for v1.25.

- v1.25 phase numbering continues after v1.24's last phase (211), with gap
  closure and cleanup phases extending the active milestone through Phase 226.

- User explicitly authorized maintained snapshot, benchmark, and model artifact updates
  when required to follow existing rules and conventions during v1.25 closeout.

- User explicitly authorized maintained snapshot, benchmark, and model artifact updates
  as necessary for Phase 226 cleanup.

- Phase 212 established `src/emel/io/read` as a boundary-only fail-closed actor with no
  request-value validation, platform read primitive, transient resource lifetime, tensor
  integration, or loader byte access.

- Phase 213 introduces validation/platform gates but still does not perform file open,
  seek, read, or close. Supported-platform requests reach `state_read_attempt_decision`
  only after all preconditions pass, then fail closed with `unsupported_resource` until
  Phase 214 replaces the placeholder with real execution.

- Phase 214.1 superseded the stale Phase 214 platform-call plan with source-backed
  validation: `src/emel/io/read` has no `actions.cpp`, stores no dispatch-local request
  data in context, and publishes copied-byte `_done` plus deterministic `_error`
  outcomes through explicit states/events.

- Phase 215 added tensor-owned `request_read_load` orchestration. `model/tensor` injects
  `emel::io::read::sm`, dispatches through public `io/read` events, commits only the
  caller-owned target buffer as resident on read success, and maps read validation,
  unsupported, file-open, and file-read outcomes through explicit tensor error states
  while preserving the upstream `io/read` error for diagnostics.

- Phase 216 added public model-loader load-strategy evidence and maintained tool
  selection/reporting through `EMEL_MODEL_LOAD_IO_STRATEGY`, `io::loader::sm` injection,
  and model-loader done/error callbacks. Generation, Sortformer diarization, embedded
  probe, and paritychecker lanes no longer infer strategy usage through load-callback
  actor probes.

- Phase 217 renamed the read strategy to `read_copy`, regenerated `io_loader`
  architecture docs, updated lint snapshot through the maintained script, and added
  guardrails proving public behavior coverage, tensor-owned read/copy residency, and
  no direct `io/read` event reach-through from model-loader or maintained tools.

- Phase 218 finalized publication truth. README, docs roadmap, README template,
  generated architecture docs, ROADMAP, REQUIREMENTS, STATE, PROJECT, MILESTONES, and the
  v1.25 audit now describe read/copy as shipped without claiming staged/chunked, async,
  or device strategy support. The Jinja formatter benchmark snapshot was refreshed through
  `scripts/bench.sh --snapshot --suite=jinja_formatter --update` after repeated isolated
  closeout-gate timing failures with no Jinja source changes. The serialized full quality
  gate then passed.

- Phase 219 replaced tool-local read source helpers in generation, Sortformer
  diarization, embedded probe, and paritychecker EMEL lanes. Phase 222 later
  moved the maintained helper out of `io/read/detail.hpp` to public
  `emel::io::source::load_file_bytes`, so Phase 219's original helper placement
  is superseded while its no-tool-local-substitute intent remains valid.

- Phase 220 replaced callback-mutated tensor read outcome selection with an
  explicit same-RTC `io/read::events::read_tensor_result` carrier and tensor
  guards/transitions over that typed result. Public tensor callbacks remain
  immediate reply publication, not outcome-routing inputs.

- Phase 221 is a superseded closeout planning stub. It owns no runtime or
  requirement validation; Phase 223 owns final closeout truth.

- Phase 222 repaired the maintained source-byte contract by adding
  `emel::io::source::load_file_bytes` in `src/emel/io/source/any.hpp`, removing
  the setup-time file loader from `io/read/detail.hpp`, and rewiring generation,
  Sortformer diarization, embedded probe, and paritychecker lanes away from
  actor-detail reach-through. Model-loader, paritychecker, generation compare,
  domain-boundary, and changed-file quality gate evidence passed.

- Phase 223 reconciled final closeout truth after Phase 222. Generated docs
  checks, lint snapshot checks, public-dispatch doctests, paritychecker tests,
  maintained generation compare, domain-boundary checks, consistency checks, and
  the source-backed milestone audit all passed. A stable `batch/planner_simple`
  snapshot regression surfaced during the full closeout gate and was repaired by
  removing redundant dispatch-local scratch-array clears from the batch planner;
  the final full-scope quality gate passed with no benchmark-regression
  override.

Carried-forward decisions from v1.24 still in effect:

- `model/loader` remains orchestration-only and must not absorb low-level read or mmap
  byte access.

- Public C ABI boundaries continue to use fixed-width integers and error codes (no
  exceptions across the boundary).

- Tensor-to-I/O integration uses public events (`request_*` / `release_*`) on
  `model::tensor::event`, not direct cross-actor function calls.

- [Phase 225-read-closeout-runtime-validation-and-sml-repair]: Batch failed-index scans use monotonic branchless helper logic so actions do not call helper-local if/break control flow.
- [Phase 225-read-closeout-runtime-validation-and-sml-repair]: Coverage gate changed-file input must be colon-separated; space-separated input is parsed as one filename by the existing script.
- [Phase 225-read-closeout-runtime-validation-and-sml-repair]: Batch read/copy result status lives in wrapper-local stack storage, not io/loader context.
- [Phase 225-read-closeout-runtime-validation-and-sml-repair]: Batch route actions use separate batch-specific action symbols so SML completion propagation resolves the originating runtime event.
- [Phase 225-read-closeout-runtime-validation-and-sml-repair]: Model-loader batch readiness is guarded in sm.hpp/guards.hpp; actions only bind already-selected tensor metadata and dispatch one child event.
- [Phase 225-read-closeout-runtime-validation-and-sml-repair]: used_io_strategy is marked only after io_load_done_all observes the public io/loader batch success.
- [Phase 225-read-closeout-runtime-validation-and-sml-repair]: io/loader batch errors now publish failed_index so model-loader can preserve concrete same-RTC error evidence.
- [Phase 225-read-closeout-runtime-validation-and-sml-repair]: Maintained caller scratch is owned by request fixture/state and resized before dispatch, matching the existing effect request/result setup pattern.
- [Phase 225-read-closeout-runtime-validation-and-sml-repair]: Guardrails stay source-scoped to maintained EMEL lanes and reject actor-internal io/read reach-through.
- [Phase 225-read-closeout-runtime-validation-and-sml-repair]: Kept Phase 225 requirements pending until Plan 06 publishes current command evidence.
- [Phase 225-read-closeout-runtime-validation-and-sml-repair]: Kept STATE at Plan 5 of 6 because Plans 01-04 are already complete in the current execution state.
- [Phase 225-read-closeout-runtime-validation-and-sml-repair]: Direct build/zig dyld launch failures are treated as environment blockers only because coverage-built focused doctests and source-backed scans passed.
- [Phase 225-read-closeout-runtime-validation-and-sml-repair]: The relevant changed-file quality gate uses generation and diarization_sortformer benchmark suite scoping to avoid unrelated jinja timing noise without weakening Phase 225 evidence.

### Pending Todos

- 2026-04-02 - Move eager quant prepack into generator initializer
- 2026-04-02 - Reuse q8 RHS across LFM2.5 prefill matmuls
- 2026-04-02 - Optimize LFM2.5 q4 prefill kernel
- 2026-04-02 - Optimize LFM2.5 q6 prefill kernel

### Blockers/Concerns

- No v1.25 blocker remains before milestone closeout.

- Two encoder benchmark suites (`text/encoders/spm_short`, `text/encoders/wpm_long`) showed
  intermittent under-load timing spikes (~31% above prior baseline) during the v1.24
  Phase 210 closing gate runs and were refreshed via the maintained scoped update path.
  Worth monitoring on v1.25 gates; not a v1.25 blocker.

- The previously deferred non-v1.23 quick task and four optimization todos remain
  carried forward and are not blockers for v1.25.

## Deferred Items

Items acknowledged and deferred at v1.25 milestone close on 2026-05-06:

| Category | Item | Status |
|----------|------|--------|
| quick_task | 260401-ejm-add-non-blocking-benchmark-binary-size-c | missing |
| todo | 2026-04-02-move-eager-quant-prepack-into-generator-initializer.md | pending |
| todo | 2026-04-02-optimize-lfm2-5-q4-prefill-kernel.md | pending |
| todo | 2026-04-02-optimize-lfm2-5-q6-prefill-kernel.md | pending |
| todo | 2026-04-02-reuse-q8-rhs-across-lfm2-5-prefill-matmuls.md | pending |

## Session Continuity

Last session: 2026-05-06T15:50:03.923Z
Stopped at: Completed 225-read-closeout-runtime-validation-and-sml-repair-06-PLAN.md
Resume file: None
