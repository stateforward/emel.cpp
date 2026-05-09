---
gsd_state_version: 1.0
milestone: v1.27
milestone_name: co_sm Cooperative Async I/O Strategy
status: completed
stopped_at: Phase 252 complete; milestone ready for final audit and closeout.
last_updated: "2026-05-09T21:43:10.524Z"
last_activity: 2026-05-09
progress:
  total_phases: 14
  completed_phases: 14
  total_plans: 14
  completed_plans: 14
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-09)

**Core value:** Prove real end-to-end behavior with explicit SML orchestration and
parity-oriented verification before widening API surface or model scope.

**Current focus:** Define milestone `v1.27 co_sm Cooperative Async I/O Strategy`
(GitHub issue #64): first approved `co_sm` / cooperative async I/O strategy for
resumable tensor loading through the tensor-to-I/O boundary, with explicit
coroutine actor rules and tensor-owned residency unchanged.

## Current Position

Phase: 252
Plan: 01 complete
Status: Phase 252 constrained-RAM cooperative async profiling complete; milestone ready for final
audit and closeout.
Last activity: 2026-05-09

Progress: [||||||||||] 100%

**Milestone setup context:**

- v1.26 staged-read closeout passed its final source-backed milestone audit.
- Issue #64 is the first approved real use of `stateforward::sml` `co_sm` /
  async dispatch in the repo.

- Phases 239-252 complete. Phase 250 closed the maintained async performance evidence gap after
  preserving loader-level resumability; Phase 251 reconciled roadmap, requirements, state, and
  evidence claims; Phase 252 closed `PERF-02` with maintained constrained-RAM profiling evidence.

Phase 237 source repair evidence remains in `237-VERIFICATION.md`.

**Residual / preserved gate truth:**

- **Phase 235:** quality gate **not attempted** in final milestone-only pass; Phase 235 makes **no** separate `scripts/quality_gates.sh` pass claim.
- **Phase 233/234:** earlier scoped gate truth remains as recorded in their phase artifacts.
- **Phase 232:** scoped gate **exit 2** residual — **unchanged** in **`232-VERIFICATION.md`**.
- **Phase 236:** serial full-repo quality gate **passed** with exit **0**; this is the maintained
  milestone-level gate evidence for closeout.

## Performance Metrics

**Prior audited milestone:** `v1.26 I/O Staged Read Loading Strategy` is the latest completed
I/O milestone (source-backed audit passed after Phase 238). v1.27 planning continues phase
numbering with Phase 239.

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. v1.27 source: GitHub issue #64.

Carry-forward architectural constraints from shipped I/O milestones:

- `model/tensor` remains the canonical residency owner; I/O strategies do not claim tensor buffer ownership.
- `model/loader` stays orchestration-only with no low-level byte strategy in loader code paths.
- Runtime behavior selection remains explicit guards and transitions (AGENTS.md / `docs/rules/sml.rules.md`).
- `co_sm` use must be codified before implementation, with suspension-safe request/progress
  ownership and no hidden mailbox or stored-callback behavior.

### Pending Todos

- 2026-04-02 - Move eager quant prepack into generator initializer
- 2026-04-02 - Reuse q8 RHS across LFM2.5 prefill matmuls
- 2026-04-02 - Optimize LFM2.5 q4 prefill kernel
- 2026-04-02 - Optimize LFM2.5 q6 prefill kernel

### Blockers/Concerns

- `PERF-01` is closed: maintained generation accepts `cooperative_async` through public
  model-loader and `io/loader` dispatches. The Phase 250 source-backed benchmark/compare run
  recorded `416407417 ns/op` EMEL vs `319636708 ns/op` reference (`1.303x`) for the selected
  generation case.

- `PERF-02` is closed. Phase 252 measured the maintained public model-loader -> `io/loader` ->
  tensor path on the publication generation fixture with constrained-RAM emulation. A 64 KiB
  window required 11,165 load dispatches and 26,901,458 ns load time; the scoped 1 MiB window run
  required 780 dispatches and 23,317,750 ns load time, with truthful source residency reporting.

- Broader decode/tokenizer/platform/accelerator async work remains explicitly deferred in
  `REQUIREMENTS.md`.

### Roadmap Evolution

- Phase 248 added initial maintained cooperative async E2E execution evidence.
- Phase 249 repaired reachable scheduler/resource terminal errors.
- Phase 250 preserved public loader resumability and renewed maintained cooperative async
  benchmark/compare evidence.

- Phase 251 repaired milestone evidence consistency without claiming Phase 252 large-model proof.
- Phase 252 added maintained generation load-profile reporting and constrained async window
  evidence for `PERF-02`.

### Prior milestone notes

The following summarized v1.25 execution context and remains historical reference:

<details>
<summary>v1.25 phase trail (collapsed)</summary>

- Phase 225/226 refined read batch APIs, audit evidence, and SML hygiene for shipped `io/read`.
- Public `read_tensor_batch` has an independent span cap; benchmarks and parity lanes use `emel::io::source::load_file_bytes` for setup-time bytes.

</details>

## Deferred Items

Items acknowledged and deferred at v1.25 milestone close on 2026-05-06 (unchanged):

| Category | Item | Status |
|----------|------|--------|
| quick_task | 260401-ejm-add-non-blocking-benchmark-binary-size-c | missing |
| todo | 2026-04-02-move-eager-quant-prepack-into-generator-initializer.md | pending |
| todo | 2026-04-02-optimize-lfm2-5-q4-prefill-kernel.md | pending |
| todo | 2026-04-02-optimize-lfm2-5-q6-prefill-kernel.md | pending |
| todo | 2026-04-02-reuse-q8-rhs-across-lfm2-5-prefill-matmuls.md | pending |

## Session Continuity

Last session: 2026-05-09 (v1.27 autonomous execution)
Stopped at: Phase 252 complete; milestone ready for final audit and closeout.
Resume file: None
