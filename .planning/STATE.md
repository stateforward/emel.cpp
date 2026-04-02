---
gsd_state_version: 1.0
milestone: v2.0
milestone_name: Bonsai 1.7B 1-Bit Bring-Up
status: roadmap_defined
stopped_at: "Roadmap created; Phase 38 is ready to plan"
last_updated: "2026-04-02T00:00:00Z"
last_activity: 2026-04-02
progress:
  total_phases: 5
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-02)

**Core value:** Prove real end-to-end behavior with explicit SML orchestration and parity-oriented
verification before widening API surface or model scope.
**Current focus:** Planning Phase 38 fixture provenance and metadata truth for the maintained
`Bonsai-1.7B.gguf` slice.

## Current Position

Phase: 38 of 42 (Fixture Provenance And Metadata Truth)
Plan: —
Status: Ready to plan
Last activity: 2026-04-02 — Replaced the active roadmap with the v2.0 Bonsai phase structure for
fixture truth, conditioning contract, native `Q1_0_g128` runtime, parity/regression, and benchmark
publication

Progress: [----------] 0%

## Performance Metrics

**Current active milestone:**

- Milestone: v2.0 Bonsai 1.7B 1-Bit Bring-Up
- Phases complete: 0/5
- Plans complete: 0/0
- Audit status: not run

**Last shipped milestone:**

- Milestone: v1.7 Generator Prefill Submachine Decomposition
- Phases complete: 3/3
- Plans complete: 6/6
- Audit status: not run

**Next action:**

- Plan Phase 38 from the frozen Bonsai fixture and GGUF metadata truth requirements.

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v2.0 keeps one maintained fixture only: `tests/models/Bonsai-1.7B.gguf`.
- Bonsai stays on the existing `qwen3` architecture path; the widening is the native
  `Q1_0_g128` operand path, not a new model family.
- The maintained Bonsai request surface is one explicit formatter contract with `tools=none`,
  `enable_thinking=false`, and no raw fallback.
- Prism's `llama.cpp` fork is the truthful parity/benchmark reference lane, and it stays confined
  to tooling only.

### Pending Todos

- 2026-04-02 — Move eager quant prepack into generator initializer
- 2026-04-02 — Reuse q8 RHS across LFM2.5 prefill matmuls
- 2026-04-02 — Optimize LFM2.5 q4 prefill kernel
- 2026-04-02 — Optimize LFM2.5 q6 prefill kernel

### Blockers/Concerns

- Native `Q1_0_g128` implementation shape is still the main technical risk for v2.0.
- Prism-specific parity and benchmark truth need an explicit pinned reference commit during later
  phase planning.
- The embedded Bonsai template exposes broader tool/thinking branches that must remain explicitly
  rejected on the maintained first slice.

## Session Continuity

Last session: 2026-04-02T00:00:00Z
Stopped at: Roadmap created; Phase 38 is ready to plan
Resume file: None
