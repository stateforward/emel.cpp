---
gsd_state_version: 1.0
milestone: v1.25
milestone_name: v1.25 I/O Read Loading Strategy
status: in_progress
stopped_at: Phase 212 read strategy component boundary validated; ready for Phase 213.
last_updated: "2026-05-05T14:52:27Z"
last_activity: 2026-05-05
progress:
  total_phases: 7
  completed_phases: 1
  total_plans: 7
  completed_plans: 1
  percent: 14
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-05)

**Core value:** Prove real end-to-end behavior with explicit SML orchestration and
parity-oriented verification before widening API surface or model scope.
**Current focus:** v1.25 I/O Read Loading Strategy — add a dedicated `src/emel/io/read`
Stateforward.SML actor for explicit read/copy tensor loading beneath tensor-owned
residency, per GitHub issue #62.

## Current Position

Phase: 213 (2 of 7) — pending plan
Plan: 01 — pending
Status: Phase 212 validated 2026-05-05. `src/emel/io/read` now exposes the canonical
read/copy boundary actor and lifecycle tests; scoped quality gates passed with
source-backed evidence. Phase 213 is ready to plan request validation and platform gating.
Last activity: 2026-05-05

Progress: [#         ] 14%

## Performance Metrics

**Latest audited milestone:** `v1.24 I/O Mmap Loading Strategy`

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
- The read strategy operates on a caller-provided owned target buffer and releases all
  transient OS resources (file descriptor / handle) before publishing `_done`. There is
  no persistent kernel handle pool analogous to the mmap slot pool.
- Staged/chunked constrained-memory read policy, cooperative async loading, device-
  specific strategies, model-family widening, loader-owned byte access, and tool-only
  read scaffolds are out of scope for v1.25.
- v1.25 phase numbering continues after v1.24's last phase (211), so phases run 212-218.
- User explicitly authorized maintained snapshot, benchmark, and model artifact updates
  when required to follow existing rules and conventions during v1.25 closeout.
- Phase 212 keeps `src/emel/io/read` boundary-only and fail-closed: accepted requests
  publish `unsupported_platform`; no request-value validation, platform read primitive,
  transient resource lifetime, tensor integration, or loader byte access is implemented
  before Phase 213/214.

Carried-forward decisions from v1.24 still in effect:

- `model/loader` remains orchestration-only and must not absorb low-level read or mmap
  byte access.
- Public C ABI boundaries continue to use fixed-width integers and error codes (no
  exceptions across the boundary).
- Tensor-to-I/O integration uses public events (`request_*` / `release_*`) on
  `model::tensor::event`, not direct cross-actor function calls.

### Pending Todos

- 2026-04-02 - Move eager quant prepack into generator initializer
- 2026-04-02 - Reuse q8 RHS across LFM2.5 prefill matmuls
- 2026-04-02 - Optimize LFM2.5 q4 prefill kernel
- 2026-04-02 - Optimize LFM2.5 q6 prefill kernel

### Blockers/Concerns

- v1.25 has no open blockers at planning time.

- Two encoder benchmark suites (`text/encoders/spm_short`, `text/encoders/wpm_long`) showed
  intermittent under-load timing spikes (~31% above prior baseline) during the v1.24
  Phase 210 closing gate runs and were refreshed via the maintained scoped update path.
  Worth monitoring on v1.25 gates; not a v1.25 blocker.

- The previously deferred non-v1.23 quick task and four optimization todos remain
  carried forward and are not blockers for v1.25.

## Deferred Items

Items acknowledged and deferred at v1.22 milestone close on 2026-05-03 (still carried):

| Category | Item | Status |
|----------|------|--------|
| quick_task | 260401-ejm-add-non-blocking-benchmark-binary-size-c | missing |
| todo | 2026-04-02-move-eager-quant-prepack-into-generator-initializer.md | pending |
| todo | 2026-04-02-optimize-lfm2-5-q4-prefill-kernel.md | pending |
| todo | 2026-04-02-optimize-lfm2-5-q6-prefill-kernel.md | pending |
| todo | 2026-04-02-reuse-q8-rhs-across-lfm2-5-prefill-matmuls.md | pending |

## Session Continuity

Last session: 2026-05-05T14:52:27Z
Stopped at: Phase 212 read strategy component boundary validated; ready to start Phase 213.
Resume file: `.planning/REQUIREMENTS.md` (v1.25 active).
