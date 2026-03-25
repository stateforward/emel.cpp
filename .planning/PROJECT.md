# EMEL

## What This Is

EMEL is a deterministic C++ inference engine built around Boost.SML orchestration, with behavior
modeled as explicit actors instead of ad hoc control flow. Shipped v1.4 now proves five narrow
brownfield outcomes on the canonical Llama-68M slice: a parity-checked generation path in
`tools/paritychecker`, a truthful maintained benchmark workflow in `tools/bench`, an EMEL-owned
flash-attention runtime path, an optimized AArch64 flash implementation, and EMEL-owned vectorized
AArch64 `q2_K/q3_K/q6_K` quantized hot-path kernels with maintained runtime/parity/benchmark
publication, all aligned with `docs/rules/sml.rules.md`.

## Core Value

Prove real end-to-end behavior with explicit SML orchestration and parity-oriented verification
before widening API surface or model scope.

## Current State

Shipped version: `v1.4`

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
- ✓ The canonical ARM slice now executes maintained `q2_K/q3_K/q6_K x q8_K` hot-path dot
  products through EMEL-owned vectorized AArch64 kernels instead of scalar row helpers — v1.4
  Phases 17-19
- ✓ The maintained quantized hot path preserves zero-allocation operand fidelity without
  dequantize-to-f32 fallback and no longer depends on the scalar row helpers for supported
  requests — v1.4 Phase 19
- ✓ The shipped generator -> graph -> processor -> kernel chain now publishes maintained q2/q3/q6
  runtime attribution and proves canonical parity across `1/10/100/1000` without actor rewrites
  or API widening — v1.4 Phase 20
- ✓ Kernel, runtime, and parity coverage now prove vectorized q2/q3/q6 correctness and
  deterministic no-claim behavior on unsupported paths — v1.4 Phase 20
- ✓ Maintained benchmark compare output and generated docs now publish quantized attribution plus
  refreshed `1/10/100/1000` evidence against the preserved v1.3 scalar baseline — v1.4 Phase 21

### Active

- [ ] `GEN-03`: Optimize ARM generator-side RMSNorm, RoPE, residual-add, and SwiGLU math after
  the vectorized quantized kernel gain is measured.
- [ ] `FLASH-03`: Broaden flash attention beyond the canonical Llama-68M shape and workload
  contract.
- [ ] `MODEL-01`: Roll optimized ARM flash attention out to additional model fixtures after the
  canonical path remains correct and benchmarked.
- [ ] `BENCH-07`: Revisit whether noisy benchmark drift should become a blocking repo gate once
  ARM compare evidence stabilizes.

### Out of Scope

- Broad repository cleanup unrelated to a milestone goal
- Non-paritychecker product surfaces until a milestone explicitly broadens the acceptance boundary
- Non-ARM backend kernel specialization until a milestone explicitly broadens beyond the canonical
  ARM truth anchor
- Whole-program state-machine or orchestration rewrites unrelated to a milestone acceptance surface
- Dequantize-to-f32 or tool-only compute fallbacks in the shipped canonical hot path without
  explicit milestone approval

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
- 5 additional phases and 11 plans delivered v1.4: the canonical ARM slice now ships EMEL-owned
  vectorized `q2_K/q3_K/q6_K` kernels, the runtime surfaces publish q2/q3/q6 attribution, full
  maintained parity is green at `1/10/100/1000`, and the compare/docs workflow republishes
  quantized evidence against the preserved v1.3 scalar baseline.
- The validated E2E flows now include
  `paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1`
  plus the maintained `scripts/bench.sh --compare` publication path that feeds
  `snapshots/bench/benchmarks_compare.txt` and `docs/benchmarks.md` with
  `optimized_flash_dispatch_calls=2`, `shared_flash_dispatch_calls=0`, and a preserved ARM
  baseline comparison.
- The maintained generation and parity matrix now publishes `1`, `10`, `100`, and `1000` token
  lengths with parity green across the full maintained set on the canonical ARM workload.
- Focused ARM profiling on the maintained `1000`-token workload identified scalar quantized matmul
  as the remaining hot leaf; v1.4 closes that hotspot with native vectorized kernels on the same
  maintained operand path.
- Current non-blocking debt remains narrow: benchmark drift is still warning-only in
  `scripts/quality_gates.sh`, compare snapshot publication still refreshes the whole maintained
  suite, proof is re-derived from common generator counters across parity/bench/docs, and the
  bench/parity/docs toolchain still emits non-blocking environment warnings on this machine.

## Next Milestone Goals

- Measure and optimize ARM generator-side math around the now-vectorized quantized path when the
  benchmark evidence shows the next highest-return hotspot.
- Decide whether broader flash/model coverage or benchmark-gate hardening is the next milestone's
  narrowest honest scope.
- Keep new milestone work anchored to the canonical Llama-68M slice until a broader contract is
  explicitly planned and accepted.

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
| Restore long-decode parity with the exact masked nonflash attention path | The maintained `100/1000` decode mismatch was a data-plane semantic drift, so the narrowest honest repair was to match the reference masked nonflash path instead of claiming flash dispatch | ✓ Good |

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
*Last updated: 2026-03-25 after shipping milestone v1.4*
