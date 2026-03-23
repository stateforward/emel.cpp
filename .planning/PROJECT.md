# EMEL

## What This Is

EMEL is a deterministic C++ inference engine built around Boost.SML orchestration, with behavior
modeled as explicit actors instead of ad hoc control flow. Shipped v1.3 now proves four narrow
brownfield outcomes on the canonical Llama-68M slice: a parity-checked generation path in
`tools/paritychecker`, a truthful maintained benchmark workflow in `tools/bench`, an EMEL-owned
flash-attention runtime path, and an optimized AArch64 flash implementation with maintained
runtime/parity/benchmark attribution, all aligned with `docs/rules/sml.rules.md`.

## Core Value

Prove real end-to-end behavior with explicit SML orchestration and parity-oriented verification
before widening API surface or model scope.

## Current Milestone: v1.4 Full Vectorized Quantized Kernels

**Goal:** Close the remaining canonical ARM inference gap by replacing the shipped scalar
`q2_K/q3_K/q6_K x q8_K` hot path with EMEL-owned vectorized AArch64 kernels while preserving the
existing Boost.SML orchestration contract.

**Target features:**
- Vectorized AArch64 `q2_K`, `q3_K`, and `q6_K` quantized dot-product coverage for the maintained
  canonical operand path
- Shipped runtime selection and deterministic fallback behavior for the canonical Llama-68M ARM
  workload
- Maintained parity, benchmark, and profiling evidence that the scalar row helpers are no longer
  the dominant hot path

## Requirements

### Validated

- ✓ Boost.SML actor machines remain the orchestration source of truth under `src/emel/**/sm.hpp`
  — existing
- ✓ The repo already includes parity and benchmark tooling plus local GGUF fixtures under
  `tools/paritychecker/`, `tools/bench/`, and `tests/models/` — existing
- ✓ One canonical `tests/models/Llama-68M-Chat-v1-Q2_K.gguf` generation slice is wired end to end
  through paritychecker — v1.0
- ✓ The slice uses real EMEL GGUF/model loading, generator initialization, bounded generation, and
  reference comparison — v1.0
- ✓ Generation mode now has subprocess success and failure coverage through the standard parity gate
  chain — v1.0
- ✓ The milestone audit passed with 14/14 requirements satisfied for the Llama-68M slice — v1.0
- ✓ One truthful `tools/bench` generation benchmark exists for the canonical Llama-68M workload
  — v1.1
- ✓ EMEL and `llama.cpp` timings for that workload are published through the existing compare flow
  — v1.1
- ✓ The canonical generation benchmark is maintained through the existing snapshot and docsgen
  surfaces — v1.1
- ✓ The v1.1 milestone audit passed with 10/10 requirements satisfied — v1.1
- ✓ The canonical Llama-68M generation path now executes through an EMEL-owned flash-attention
  path in `src/emel/generator` and `src/emel/kernel` — v1.2 Phases 10-11
- ✓ `tools/paritychecker --generation` now proves the flash-attention path on the canonical
  `tests/models/Llama-68M-Chat-v1-Q2_K.gguf` workload through the normal repo surface — v1.2
  Phase 12
- ✓ The shipped graph and generator runtime now use `emel::tensor::sm` as the only tensor
  lifecycle authority, with formal reserve/fill/release cutover and alloc-free dispatch proof
  — v1.2 Phase 12.1
- ✓ The maintained benchmark workflow now proves canonical flash execution, preserves a pre-flash
  baseline artifact, and publishes generated benchmark evidence showing measurable improvement
  on the maintained short compare case — v1.2 Phase 13
- ✓ The canonical CPU-hosted Llama-68M ARM slice now executes `op_flash_attn_ext` through an
  EMEL-owned optimized AArch64 implementation with backend-owned reusable scratch — v1.3 Phases
  14-15
- ✓ `tools/paritychecker --generation` and `tools/bench` now publish optimized-vs-shared ARM
  flash attribution on the maintained canonical workload — v1.3 Phases 15-16
- ✓ Maintained benchmark publication now preserves the prior ARM baseline separately and documents
  a `1.140x` canonical short-case speedup over that baseline — v1.3 Phase 16

### Active

- [ ] `PORT-04`: The canonical Llama-68M ARM generation slice executes `q2_K/q3_K/q6_K x q8_K`
  hot-path dot products through EMEL-owned vectorized AArch64 kernels instead of scalar row
  helpers.
- [ ] `PORT-07`: The vectorized quantized kernels preserve zero-allocation hot-path behavior and
  keep the maintained effective operand class without dequantize-to-f32 fallbacks.
- [ ] `ARCH-02`: The optimization remains a data-plane replacement inside the existing generator
  -> graph -> processor -> kernel chain and does not widen the public API surface or rewrite
  actor structure.
- [ ] `PAR-04`: `tools/paritychecker --generation` keeps the maintained `1/10/100/1000` token
  checks and proves the canonical ARM workload exercised the vectorized quantized path.
- [ ] `VER-03`: Tests cover vectorized `q2_K/q3_K/q6_K` correctness, scalar equivalence, and
  deterministic fallback behavior on AArch64.
- [ ] `BENCH-08`: `tools/bench` publishes maintained canonical ARM compare output with attribution
  that distinguishes the vectorized quantized path from the current scalar row-helper path.
- [ ] `BENCH-09`: Maintained `1/10/100/1000` token compare output shows truthful end-to-end
  impact over the current v1.3 baseline, with measurable improvement on at least one maintained
  length.

### Out of Scope

- Broad repository cleanup unrelated to a milestone goal
- Non-paritychecker product surfaces until a milestone explicitly broadens the acceptance boundary
- Non-ARM backend flash specialization until the canonical ARM follow-on work is complete
- Whole-program state-machine or orchestration rewrites unrelated to a milestone acceptance surface

## Current State

Shipped version: `v1.3`

- 7 phases and 15 plans delivered the first canonical Llama-68M parity slice in v1.0.
- 4 additional phases and 10 plans delivered the truthful benchmark slice in v1.1.
- 5 additional phases and 13 plans delivered v1.2: flash attention is live in the shipped
  canonical generator path, paritychecker proves it against fetched upstream `llama.cpp`,
  graph/generator runtime ownership is hard-cut through `emel::tensor::sm`, and the maintained
  benchmark workflow now publishes flash evidence plus a preserved pre-flash baseline.
- 3 additional phases and 7 plans delivered v1.3: the canonical ARM slice now ships an optimized
  AArch64 flash kernel, the shipped runtime/parity surfaces publish optimized-vs-shared flash
  attribution, and maintained benchmark docs preserve the prior ARM baseline while publishing a
  measured short-case improvement.
- The validated E2E flows now include
  `paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1`
  plus the maintained `scripts/bench.sh --compare` publication path that feeds
  `snapshots/bench/benchmarks_compare.txt` and `docs/benchmarks.md` with
  `optimized_flash_dispatch_calls=2`, `shared_flash_dispatch_calls=0`, and a preserved ARM
  baseline comparison.
- The maintained generation and parity matrix now publishes `1`, `10`, `100`, and `1000` token
  lengths, with parity currently passing `1` and `10` while drifting later in longer decode
  sequences.
- Focused ARM profiling on the maintained `1000`-token workload shows the remaining hot leaf time
  is dominated by quantized matmul: about `89.8%` of sampled leaf cost is in
  `dot_q2_k_q8_k_block_scalar`, `dot_q6_k_q8_k_row_scalar`, and `dot_q3_k_q8_k_row_scalar`,
  with flash attention down near `7.3%`.
- The shipped AArch64 backend already routes quantized matmul through `execute_neon_mul_mat`, but
  the maintained `q2_K/q3_K/q6_K x q8_K` row path still calls scalar helpers rather than the
  existing NEON block kernels.
- Current non-blocking debt remains narrow: benchmark drift is still warning-only in
  `scripts/quality_gates.sh`, compare snapshot publication still refreshes the whole maintained
  suite, proof is re-derived from common generator counters across parity/bench/docs, and the
  bench/parity/docs toolchain still emits non-blocking environment warnings on this machine.

## Current Milestone Goals

- Replace the profiled scalar `q2_K/q3_K/q6_K x q8_K` inner loops with full vectorized AArch64
  kernels on the maintained canonical workload.
- Preserve the existing generator -> graph -> processor -> kernel architecture contract and make
  fallback behavior explicit when requests fall outside the maintained optimized path.
- Publish maintained parity, benchmark, and profiling evidence over `1/10/100/1000` token
  lengths before revisiting broader generator math or model-coverage work.

## Context

This is a brownfield repository with an existing codebase map under `.planning/codebase/`. v1.0
showed that the existing subsystem families under `src/emel/` are sufficient to prove one real
generation path without widening the public API surface. The repo remains governed by `AGENTS.md`
and `docs/rules/sml.rules.md`, so future work still needs to preserve same-RTC actor semantics,
explicit error publication, bounded actions, and deliberate machine-structure changes.

## Constraints

- **Architecture**: Follow `docs/rules/sml.rules.md` and the local machine conventions in
  `AGENTS.md` — the generation slice must preserve the RTC actor model and no-queue invariant
- **Acceptance boundary**: Milestones should expand intentionally; v1.0 proved paritychecker-first
  generation, not a general public API surface
- **Model scope**: v1.0 locked one canonical Llama-68M fixture first; broader model matrices should
  only expand under an explicit next-milestone goal
- **Performance philosophy**: Favor explicit orchestration correctness first, then optimize once
  parity-oriented behavior is locked
- **Repository state**: This is an active brownfield codebase, so milestone work should still
  minimize unrelated churn

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Target the Llama-68M GGUF model first | It is a small local test model already present in `tests/models/` and is suitable for proving a narrow vertical slice | ✓ Good |
| Define done in paritychecker, not the public C API | The immediate goal is parity-oriented end-to-end generation proof, not API surface expansion | ✓ Good |
| Optimize for a narrow vertical slice | This reduces planning noise and avoids turning the first proof into a general cleanup campaign | ✓ Good |
| Enforce SML rules as a first-class planning constraint | The repo architecture and user requirements both depend on `docs/rules/sml.rules.md` semantics | ✓ Good |
| Insert Phase 1.1 to implement the real GGUF loader | The stubbed loader became the hard blocker for the milestone | ✓ Good |
| Close `HARN-02` with canonical-path fixture enforcement | The audit exposed that basename-only validation overstated the shipped contract | ✓ Good |
| Keep the normal paritychecker and quality-gate scripts as the verification surface | Reusing existing gates kept the slice honest and minimized repo churn | ✓ Good |
| Leave benchmark snapshot drift policy unchanged during v1.0 | The milestone prioritized generation correctness over repo-wide gate-policy changes | ⚠ Revisit |
| Reuse the v1.0 generation slice in `tools/bench` first | This kept the benchmark milestone narrow and anchored to proven behavior | ✓ Good |
| Insert Phase 07.1 before publishing benchmark numbers | Truthful EMEL benchmark evidence required replacing the circular reference-backed decode seam | ✓ Good |
| Keep benchmark maintenance on the existing compare/snapshot/docs surfaces | This preserved one operator workflow instead of creating benchmark-only tooling | ✓ Good |
| Start v1.2 with flash attention in paritychecker plus bench | This kept the milestone narrow, measurable, and aligned with the shipped generation surfaces | ✓ Good |
| Insert Phase 12.1 to hard-cut tensor lifecycle through `emel::tensor::sm` | Flash runtime work had left an ad hoc lifecycle bypass in graph/generator, which violated the architecture contract and needed removal before benchmarking | ✓ Good |
| Preserve pre-flash benchmark evidence as a standalone artifact | Benchmark claims needed a durable non-flash baseline separate from the current flash compare snapshot | ✓ Good |
| Keep the shared flash validator and replace only the AArch64 data plane | This preserved the Boost.SML orchestration contract while targeting the hottest ARM cost center first | ✓ Good |
| Expose optimized/shared flash counts through wrappers instead of new events or transition rows | Runtime truth had to be observable without changing machine structure | ✓ Good |
| Extend the existing parity and benchmark proof channels instead of inventing new ones | Reusing maintained operator surfaces kept proof honest and minimized publication churn | ✓ Good |
| Preserve the pre-refresh ARM compare row as a dedicated baseline artifact | Maintained benchmark claims needed a durable shared-scalar ARM comparison point before snapshot refresh | ✓ Good |
| Target the quantized `q2_K/q3_K/q6_K x q8_K` row-dot path next | The latest flame graph shows the remaining ARM gap is dominated by scalar quantized matmul leaf cost, not orchestration frames | — Pending |
| Skip milestone research and keep the acceptance boundary on the canonical ARM Llama-68M slice | This milestone is a direct continuation of already-profiled kernel work, so new-domain research would add delay without reducing uncertainty | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `$gsd-transition`):
1. Requirements invalidated? -> Move to Out of Scope with reason
2. Requirements validated? -> Move to Validated with phase reference
3. New requirements emerged? -> Add to Active
4. Decisions to log? -> Add to Key Decisions
5. "What This Is" still accurate? -> Update if drifted

**After each milestone** (via `$gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check -> still the right priority?
3. Audit Out of Scope -> reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-22 after starting milestone v1.4*
