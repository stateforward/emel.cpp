---
gsd_state_version: 1.0
milestone: v1.10
milestone_name: Bonsai 1.7B 1-Bit Bring-Up
status: ready_to_plan
stopped_at: "Phase 41.1 closed with a green quality gate; Phase 42 benchmark and publication work is next"
last_updated: "2026-04-03T04:24:56Z"
last_activity: 2026-04-02
progress:
  total_phases: 7
  completed_phases: 6
  total_plans: 5
  completed_plans: 5
  percent: 86
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-02)

**Core value:** Prove real end-to-end behavior with explicit SML orchestration and parity-oriented
verification before widening API surface or model scope.
**Current focus:** Planning the final benchmark-and-publication phase for the maintained
`Bonsai-1.7B.gguf` slice after Phase 41.1 reduced the isolated implementation ratio from `3.173x`
to `1.343x`.

## Current Position

Phase: 42 of 42 (Benchmark And Publication)
Plan: —
Status: Ready to plan
Last activity: 2026-04-02 — Closed Phase 41.1 after the full quality gate stayed green and the
 maintained Bonsai implementation bench improved materially.

Progress: [#########-] 86%

## Performance Metrics

**Current active milestone:**

- Milestone: v1.10 Bonsai 1.7B 1-Bit Bring-Up
- Phases complete: 6/7
- Plans complete: 5/5
- Audit status: not run

**Last shipped milestone:**

- Milestone: v1.7 Generator Prefill Submachine Decomposition
- Phases complete: 3/3
- Plans complete: 6/6
- Audit status: not run

**Next action:**

- Discuss and plan Phase 42 so the maintained Bonsai benchmark/doc publication path can be landed
  without violating the explicit snapshot-consent rule.

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v1.10 keeps one maintained fixture only: `tests/models/Bonsai-1.7B.gguf`.
- Bonsai stays on the existing `qwen3` architecture path; the widening is the native
  `Q1_0_g128` operand path, not a new model family.
- The maintained Bonsai request surface is one explicit formatter contract with `tools=none`,
  `enable_thinking=false`, and no raw fallback.
- Embedded chat-template truth now flows into `model_data.meta.tokenizer_data`, and generator
  construction resolves the formatter contract in `src` rather than in tool-only code.
- Prism's `llama.cpp` fork is the truthful parity/benchmark reference lane, and it stays confined
  to tooling only.
- Maintained fixture identity and generation-supported tooling identity are now separate so Bonsai
  is visible in the repo without being overclaimed on pre-runtime parity and bench paths.
- Real upstream GGUF tensor type `41` is now reserved for `Q1_0_g128`; EMEL-owned packed/prepared
  pseudo-dtypes moved out of the upstream id range.
- The shipped generator path now accepts native `Q1_0_g128` tensors on the existing `qwen3` lane
  without a whole-tensor dequantize-to-f32 fallback.
- Qwen-family RoPE is now a real EMEL runtime domain: `model::data` publishes rope type and rope
  scaling metadata, and the shipped generator applies explicit `neox` plus YaRN semantics instead
  of plain adjacent-pair RoPE.
- The maintained Qwen generation fixture now uses live reference generation, because the old
  append-only stored baseline no longer matched the truthful current runtime contract.
- Maintained parity is now an isolated per-model lane contract rather than one global
  monolithic reference build; the scripted parity workflow is the truthful proof surface.
- The maintained Bonsai token-embedding stage now audits as `native_quantized`, and the shipped
  AArch64 `Q1_0_g128 x q8_0` path materially improved the isolated implementation benchmark from
  `3.173x` slower to `1.343x` slower against the pinned Prism lane.

### Roadmap Evolution

- Phase 41.1 inserted after Phase 41: full quantized as discussed to close the 3.17x gap (URGENT)
- Phase 41.1 closed on 2026-04-03 after the full quality gate stayed green.

### Pending Todos

- 2026-04-02 — Move eager quant prepack into generator initializer
- 2026-04-02 — Reuse q8 RHS across LFM2.5 prefill matmuls
- 2026-04-02 — Optimize LFM2.5 q4 prefill kernel
- 2026-04-02 — Optimize LFM2.5 q6 prefill kernel

### Blockers/Concerns

- Phase 42 likely requires snapshot and generated benchmark-evidence updates under `snapshots/bench`
  and `docs/`, and repo policy requires explicit user consent before landing those updates.
- The embedded Bonsai template still exposes broader tool/thinking branches that must remain
  explicitly rejected on the maintained first slice during publication.
- `tools/bench` still reports a non-fatal snapshot regression on `text/encoders/ugm_short`, but
  `scripts/quality_gates.sh` explicitly tolerates benchmark snapshot drift in the current gate.
- Benchmark and publication work now becomes the only remaining milestone phase.

## Session Continuity

Last session: 2026-04-02T00:00:00Z
Stopped at: Phase 41.1 complete; benchmark and publication planning is next
Resume file: None
