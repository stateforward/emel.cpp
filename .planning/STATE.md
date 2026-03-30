---
gsd_state_version: 1.0
milestone: v1.7
milestone_name: Generator Prefill Submachine Decomposition
status: milestone_complete
stopped_at: "v1.7 archived; ready to define the next milestone"
last_updated: "2026-03-30T04:41:15Z"
last_activity: 2026-03-29
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 6
  completed_plans: 6
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-30)

**Core value:** Prove real end-to-end behavior with explicit SML orchestration and parity-oriented
verification before widening API surface or model scope.
**Current focus:** Define the next milestone after shipping v1.7.

## Current Position

Status: Between milestones
Last activity: 2026-03-30 — Archived v1.7 after the explicit `generator/prefill` extraction,
top-level generator shrink, maintained generator/parity/compare proof refresh, and user-approved
waiver for unrelated broad benchmark regressions outside `generator/prefill`

Progress: [██████████] 100%

## Performance Metrics

**Last shipped milestone:**

- Milestone: v1.7 Generator Prefill Submachine Decomposition
- Phases complete: 3/3
- Plans complete: 6/6
- Audit status: not run

**Previous shipped milestone:**

- Milestone: v1.6 Qwen3-0.6B Parity And Benchmark
- Phases complete: 5/5
- Plans complete: 12/12
- Audit status: tech_debt

**Next action:**

- Define the next milestone with `$gsd-new-milestone`.

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v1.6 stayed narrow to one canonical `Qwen3-0.6B-Q8_0.gguf` maintained slice.
- The maintained Qwen formatter contract comes from the primary GGUF
  `tokenizer.chat_template`, with structured chat-message input and no implicit raw fallback.
- Phase `26.1` inserted before Phase `27` so native `q8_0` runtime support landed in `src/emel`
  before truthful Qwen generator bring-up resumed.
- The prior Llama anchor remains protected while the canonical Qwen parity and benchmark surfaces
  are maintained alongside it.
- v1.7 started with `generator/prefill` as the first decomposition cut rather than forcing a full
  generator split or separate session actor redesign.
- v1.7 shipped without a separate milestone audit; the user chose direct completion after the
  maintained generator-specific proof and explicit benchmark-noise waiver.

### Pending Todos

- None.

### Blockers/Concerns

- Non-blocking benchmark warning debt remains in `batch/planner_simple`, `memory/hybrid_full`,
  `kernel/aarch64/op_log`, `logits/sampler_raw/vocab_32000`, and `kernel/aarch64/op_soft_max`.
- The next milestone scope is not defined yet.
- Decode extraction and any attention-family `sm_any` work remain deferred until the next
  milestone explicitly picks them up.
- Warning-only benchmark debt still exists outside `generator/prefill`; v1.7 closeout accepted
  that carried debt rather than hardening the broader repo gate.

## Session Continuity

Last session: 2026-03-30T04:41:15Z
Stopped at: v1.7 archived; ready for `$gsd-new-milestone`
Resume file: None
