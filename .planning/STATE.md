---
gsd_state_version: 1.0
milestone: v1.9
milestone_name: Liquid LFM2.5-1.2B Thinking ARM Slice
status: ready_for_new_milestone
stopped_at: "v1.9 archived after merging the shipped v1.8 milestone artifacts; ready for $gsd-new-milestone"
last_updated: "2026-04-02T23:45:00Z"
last_activity: 2026-04-02
progress:
  total_phases: 8
  completed_phases: 8
  total_plans: 9
  completed_plans: 9
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-02)

**Core value:** Prove real end-to-end behavior with explicit SML orchestration and parity-oriented
verification before widening API surface or model scope.
**Current focus:** Await next milestone definition after v1.8/v1.9 archive.

## Current Position

Phase: —
Plan: —
Status: Milestones archived
Last activity: 2026-04-02 — Merged the shipped v1.8 milestone artifacts from main into the branch
planning history after archiving v1.9

Progress: [##########] 100%

## Performance Metrics

**Latest shipped milestone:**

- Milestone: v1.9 Liquid LFM2.5-1.2B Thinking ARM Slice
- Phases complete: 8/8
- Plans complete: 9/9
- Audit status: passed

**Next action:**

- Start the next milestone with `$gsd-new-milestone`.

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v1.9 is fixed to one official `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` maintained fixture.
- The maintained Liquid slice will derive truth from GGUF/config metadata (`lfm2`, `128000`
  context) rather than stale prose on the model card.
- The just-merged v1.8 milestone fixed the maintained executable-size claim to final linked
  executables on the canonical Qwen3 slice.
- The maintained Liquid request surface is one structured chat-message contract with `tools=none`
  and no thinking-history replay.

- Roadmap coverage is fixed at five phases: 33 fixture/metadata/contract, 34 `lfm2` contract, 35
  runtime, 36 parity/regression, 37 benchmark/docs.

- The user explicitly reprioritized the milestone to the docs-recommended `Q4_K_M` quant instead
  of the earlier `Q8_0` anchor.

- Decode extraction and broader generator decomposition remain deferred on this branch.

### Pending Todos

- 2026-04-02 — Move eager quant prepack into generator initializer
- 2026-04-02 — Reuse q8 RHS across LFM2.5 prefill matmuls
- 2026-04-02 — Optimize LFM2.5 q4 prefill kernel
- 2026-04-02 — Optimize LFM2.5 q6 prefill kernel
- Decide whether the next milestone prioritizes broader Liquid work or a broader executable-size
  publication surface.

### Blockers/Concerns

- Current runtime and tooling explicitly accept `llama` and `qwen3`; `lfm2` is new architecture
  scope.

- Switching the maintained anchor from `Q8_0` to `Q4_K_M` widens Phase 35 runtime risk relative to
  the earlier research baseline.

- The current supported formatter contract is Qwen-shaped and will need a Liquid-specific truth
  surface.

- Official Liquid prose and executable metadata disagree on context length, so docs and tests must
  use metadata truth consistently.

- Benchmark warning debt still exists outside the maintained Liquid scope and is not part of v1.9.

## Session Continuity

Last session: 2026-04-02T17:46:31Z
Stopped at: v1.9 archived; v1.8 archive merged from main; ready for `$gsd-new-milestone`
Resume file: None
