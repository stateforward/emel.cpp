---
gsd_state_version: 1.0
milestone: v1.6
milestone_name: Qwen3-0.6B Parity And Benchmark
status: ready_for_milestone_audit
stopped_at: "Completed 29-02 canonical Qwen benchmark publication and full gate verification"
last_updated: "2026-03-28T13:42:57Z"
progress:
  total_phases: 5
  completed_phases: 5
  total_plans: 12
  completed_plans: 12
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-27)

**Core value:** Prove real end-to-end behavior with explicit SML orchestration and parity-oriented
verification before widening API surface or model scope.
**Current focus:** Milestone closeout for v1.6

## Current Position

Phase: 29 (qwen3-benchmark-publication) — COMPLETE
All v1.6 phases are complete. The canonical Qwen3 slice now proves maintained stored-baseline
generation parity on `1/10/100/1000`, maintained benchmark compare and snapshot publication use
the same GGUF-derived formatter contract and native `q8_0` runtime evidence, and the preserved ARM
Llama flash baseline remains documented as a historical artifact instead of a live comparison row.

## Performance Metrics

**Current milestone:**

- Milestone: v1.6 Qwen3-0.6B Parity And Benchmark
- Phases complete: 5/5
- Plans complete: 12/12
- Audit status: not started

**Last shipped milestone:**

- Milestone: v1.5 Full ARM Quantized Path
- Phases complete: 5/5
- Plans complete: 10/10
- Audit status: tech_debt

**Next action:**

- Run `$gsd-audit-milestone v1.6` or `$gsd-complete-milestone v1.6`.

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- v1.5 closed the canonical ARM quantized-path contract at `8/4/0/0`.
- Phase 25.1 restored canonical flash dispatch and aligned stored compare/docs evidence with the
  live proof surface.

- Benchmark drift remains warning-only repo policy until a future milestone explicitly changes it.
- v1.6 stays narrow to one canonical Qwen3-0.6B GGUF slice on the maintained paritychecker and
  benchmark surfaces.

- v1.6 uses one explicit request-conditioning contract, does not treat `qwen3` as a Llama alias,
  and now has approval to widen the formatter/request shape to structured chat messages for the
  canonical maintained slice.

- Phase 26 was re-scoped during execution so the maintained paritychecker and bench paths must fail
  truthfully on the canonical Qwen fixture before runtime bring-up exists; successful Qwen runtime
  support remains Phase 27 work.

- Phase 26-02 widened formatter and conditioner to explicit structured chat messages.
- Phase 26-03 widened `generator::event::generate` to structured chat messages and removed the
  temporary prompt-reconstruction bridge.

- Phase 26-04 bound the maintained formatter contract from the primary GGUF template on parity and
  bench tool surfaces, publishing that contract before truthful pre-runtime failure.

- [Phase 27]: Canonical qwen3 execution-view construction requires explicit attn_q_norm and attn_k_norm tensors. — The canonical qwen3 GGUF slice declares dedicated q/k norm weights, so the maintained execution-view contract must reject missing tensors instead of treating qwen3 as a llama alias.
- [Phase 27]: Qwen3-only quantized audit stages are reported as not_applicable outside qwen3. — This preserves prior llama audit truth while keeping the expanded shared inventory explicit for the canonical qwen3 slice.
- [Phase 26.1]: Native hot-path proof for the canonical Qwen slice must be reported with `native_q8_0_dispatch_calls`, not reused `q2/q3/q6` attribution counters from the earlier ARM slice.
- [Phase 27]: Structured-message planner requests stay explicit about sequence metadata. Single-sequence multi-token prompt planning sends null sequence metadata unless multiple sequences are actually present.
- [Phase 27]: Maintained Qwen runtime bring-up is proven on the temp-baseline write path first; stored snapshot-backed parity comparison remains Phase 28 work.
- [Phase 28]: Maintained Qwen generation parity is only truthful in stored compare mode when the
  reference tokenizer uses the same GGUF-derived formatter contract and attribution replay consumes
  the same conditioned prompt tokens as the shipped generator.
- [Phase 28]: The prior Llama anchor stays protected by keeping the canonical Qwen compare-mode
  work inside the existing paritychecker suite instead of inventing a Qwen-only regression lane.
- [Phase 29]: Stored benchmark publication records explicit `benchmark_config` metadata so the
  published contract names the exact iteration and warmup envelope instead of leaving it implicit
  in scripts.
- [Phase 29]: The canonical maintained benchmark identity is now Qwen `1/10/100/1000` on both
  compare and snapshot surfaces; the prior ARM Llama flash baseline is preserved as historical
  evidence only and is not presented as a live comparator for the Qwen publication slice.

### Roadmap Evolution

- Phase 26.1 inserted after Phase 26: Native q8_0 projection and output runtime support for canonical Qwen3 (URGENT)
- Phase 26.1 completed as two narrow steps: explicit native `q8_0` shared-runtime support, then canonical Qwen backend-unblock proof before Phase 27 resumed.

### Pending Todos

- None.

### Blockers/Concerns

- Non-blocking benchmark warning debt remains in `batch/planner_simple`, `memory/hybrid_full`,
  `kernel/aarch64/op_log`, `logits/sampler_raw/vocab_32000`, and `kernel/aarch64/op_soft_max`.

- Qwen bring-up must stay narrow and truthful so the milestone does not over-claim broader model
  support.

- The formatter/request boundary widening is approved only for the explicit maintained Qwen slice
  captured in Phase 26 and should not be generalized implicitly during execution.

## Session Continuity

Last session: 2026-03-28T13:42:57Z
Stopped at: Completed 29-02 canonical Qwen benchmark publication and full gate verification
Resume file: None
