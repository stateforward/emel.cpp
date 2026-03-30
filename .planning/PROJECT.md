# EMEL

## What This Is

EMEL is a deterministic C++ inference engine built around Boost.SML orchestration, with behavior
modeled as explicit actors instead of ad hoc control flow. Shipped v1.5 now proves six narrow
brownfield outcomes on the canonical Llama-68M ARM slice: a parity-checked generation path in
`tools/paritychecker`, a truthful maintained benchmark workflow in `tools/bench`, an EMEL-owned
flash-attention runtime path, an optimized AArch64 flash implementation, EMEL-owned vectorized
AArch64 `q2_K/q3_K/q6_K` quantized hot-path kernels, and a maintained end-to-end quantized-path
contract with restored checked-in flash attribution and publication, all aligned with
`docs/rules/sml.rules.md`.

## Core Value

Prove real end-to-end behavior with explicit SML orchestration and parity-oriented verification
before widening API surface or model scope.

## Current State

Current milestone: `v1.6`

Status: Defining requirements for a new narrow brownfield milestone. The shipped canonical ARM
Llama-68M slice still anchors truth for the current repo surfaces, while the next milestone is
intended to prove one maintained Qwen3-0.6B GGUF slice through the same paritychecker and
benchmark channels without overstating model breadth.

Transition summary: `v1.5 Full ARM Quantized Path` shipped on `2026-03-27` with `5` phases and
`10` plans. `v1.6` now starts from that shipped baseline and keeps the existing benchmark-warning
policy as carried technical debt.

## Current Milestone: v1.6 Qwen3-0.6B Parity And Benchmark

**Goal:** Prove one truthful canonical Qwen3-0.6B GGUF slice through the maintained EMEL
generation, paritychecker, and benchmark workflow without widening the acceptance boundary beyond
that slice.

**Target features:**
- Add one maintained canonical Qwen3-0.6B GGUF fixture with documented provenance and stable
  operator-facing identity.
- Extend the shipped brownfield generator/parity path only as far as needed to load, run, and
  compare that Qwen slice honestly.
- Publish one maintained benchmark compare/docs flow for the same Qwen3-0.6B slice so parity and
  benchmark claims stay aligned.

## Latest Milestone: v1.5 Full ARM Quantized Path

<details>
<summary>Shipped on 2026-03-27</summary>

**Goal:** Prove the maintained canonical ARM generation slice stays on the intended quantized
operand path end to end, eliminate any disallowed f32/dequant detours that still exist, and make
the remaining contract explicit where full quantized coverage still does not apply.

**Delivered:**
- Audited the maintained canonical ARM generation chain and classified each quantized stage as
  native quantized, approved dense-f32-by-contract, or explicit no-claim when unsupported.
- Codified and proved the supported shipped runtime contract as
  `native_quantized=8 approved_dense_f32_by_contract=4 disallowed_fallback=0 explicit_no_claim=0`.
- Hardened maintained paritychecker, runtime regression, and benchmark publication so supported
  canonical requests fail on disallowed fallback.
- Restored canonical flash-attention dispatch on the maintained generator path and refreshed the
  stored compare/docs publication so checked-in evidence matches the live flash proof again.

</details>

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
- ✓ The canonical ARM generation slice now has a shared execution-view audit that classifies each
  maintained stage family as native quantized, approved dense-f32-by-contract, or explicit
  no-claim when unsupported — v1.5 Phase 22
- ✓ Unsupported quantized stage families now publish explicit no-claim behavior through the shared
  audit helper and maintained paritychecker output instead of silently inheriting approved claims
  — v1.5 Phase 22
- ✓ The shipped generator runtime now exposes additive audited stage-count accessors and proves the
  supported canonical initialized fixture reports `native_quantized=8`,
  `approved_dense_f32_by_contract=4`, `disallowed_fallback=0`, and
  `explicit_no_claim=0` — v1.5 Phase 23
- ✓ Maintained paritychecker generation now publishes the shipped runtime contract and proves the
  canonical ARM workload stays on the approved `8/4/0/0` contract across `1/10/100/1000`
  tokens — v1.5 Phase 24
- ✓ Generator and parity regression coverage now prove the supported contract survives a real
  `generate` call without collapsing unsupported-stage proof away from explicit `no-claim`
  behavior — v1.5 Phase 24
- ✓ Maintained benchmark compare output, stored snapshot evidence, and generated docs now publish
  the shipped `8/4/0/0` runtime contract and honest dense-f32-by-contract attribution for the
  canonical ARM workload — v1.5 Phase 25
- ✓ The maintained generator path, compare snapshot, and generated benchmark docs now publish
  restored canonical flash attribution alongside the same shipped `8/4/0/0` runtime contract
  again — v1.5 Phase 25.1

### Active

- [ ] `QWEN-01`: A single maintained canonical Qwen3-0.6B GGUF fixture is documented and usable as
  the v1.6 truth anchor.
- [ ] `QWEN-02`: The shipped generator and paritychecker surfaces can load and run that canonical
  Qwen3-0.6B slice without widening support claims to other Qwen variants.
- [ ] `QWEN-03`: Parity output proves EMEL against the reference implementation on that one
  canonical Qwen3-0.6B slice through the maintained operator workflow.
- [ ] `QWEN-04`: `tools/bench` publishes one truthful maintained compare/docs path for the same
  canonical Qwen3-0.6B slice.

### Out of Scope

- Broad repository cleanup unrelated to a milestone goal
- Non-paritychecker product surfaces until a milestone explicitly broadens the acceptance boundary
- Non-ARM backend kernel specialization until a milestone explicitly broadens beyond the canonical
  ARM truth anchor
- Whole-program state-machine or orchestration rewrites unrelated to a milestone acceptance surface
- Dequantize-to-f32 or tool-only compute fallbacks in the shipped canonical hot path without
  explicit milestone approval

## Milestone Focus

- Keep the milestone narrow to one canonical Qwen3-0.6B GGUF fixture instead of broad Qwen-family
  enablement.
- Preserve the existing acceptance boundary on the maintained generator, paritychecker, compare,
  snapshot, and docs surfaces.
- Prefer truthful brownfield bring-up over premature optimization so benchmark claims do not outrun
  actual parity coverage.

## Context

This is a brownfield repository with an existing codebase map under `.planning/codebase/`. v1.5
showed that the existing subsystem families under `src/emel/` are also sufficient to audit, prove,
and publish the canonical ARM quantized-path contract end to end without widening the public C API
surface. The repo remains governed by `AGENTS.md` and `docs/rules/sml.rules.md`, so future work
still needs to preserve same-RTC actor semantics, explicit error publication, bounded actions, and
deliberate machine-structure changes. The current repo already recognizes Qwen-family model
metadata in some loader/tokenizer paths, but the maintained generator, paritychecker, and benchmark
truth anchors are still Llama-specific today. Non-blocking benchmark warnings remain the main
carried technical debt entering the next milestone.

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
| Target the quantized `q2_K/q3_K/q6_K x q8_K` row-dot path next | The latest flame graph shows the remaining ARM gap is dominated by scalar quantized matmul leaf cost, not orchestration frames | ✓ Good |
| Restore long-decode parity with the exact masked nonflash attention path | The maintained `100/1000` decode mismatch was a data-plane semantic drift, so the narrowest honest repair was to match the reference masked nonflash path instead of claiming flash dispatch | ✓ Good |
| Treat the remaining ARM quantized-path question as a contract audit first, optimization second | The repo now needs proof about where operand formats widen before it can honestly claim a full quantized runtime path | ✓ Good |
| Build the quantized-path audit from `llama::detail::execution_view` plus maintained stage-family mapping | Aggregate dispatch counters alone cannot classify token embedding, norm-vector seams, or unsupported branches truthfully | ✓ Good |
| Treat canonical q2/q3/q6 matmul stages as `native_quantized` and token embedding or norm-vector seams as `approved_dense_f32_by_contract` | This matches the shipped operand contract and avoids mislabeling the dense-rhs `q8_K` repack path as fallback | ✓ Good |
| Publish unsupported quantized branches as explicit `no-claim` rows in paritychecker output | Unsupported cases must stay visible without silently inheriting an approved contract label | ✓ Good |
| Close Phase 23 by codifying and proving the zero-gap runtime contract instead of inventing a fake fallback bug | Phase 22 research plus direct runtime inspection showed the supported canonical path already had zero disallowed-fallback stages, so honest closure meant additive publication and proof only | ✓ Good |
| Promote parity proof from model-local audit printing to shipped runtime-contract enforcement | The maintained proof surface should fail on supported-path regression based on what the generator actually ran, while still cross-checking the stage audit for consistency | ✓ Good |
| Reuse the existing explicit-no-claim negative surface instead of fabricating a supported fallback fixture for Phase 24 | Unsupported-stage proof was already truthful and deterministic, so additional fake fallback scaffolding would have reduced clarity rather than improved coverage | ✓ Good |
| Publish benchmark-time runtime contract through the existing compare/docs workflow after explicit approval | This completed `BENCH-10` without inventing parallel publication tooling and kept the approved dense-f32 seams explicit in stored evidence | ✓ Good |
| Insert Phase 25.1 before closing v1.5 | The live canonical generator path and the stored benchmark publication had drifted apart, so truthful closeout required restoring flash dispatch and refreshing checked-in compare/docs evidence before archival | ✓ Good |
| Keep v1.6 scoped to one canonical Qwen3-0.6B GGUF slice | The maintained generator, paritychecker, and benchmark surfaces are still Llama-shaped today, so truthful progress requires one narrow vertical slice before any broader Qwen rollout | — Pending |
| Reuse the shipped paritychecker and `tools/bench` surfaces as the only v1.6 acceptance boundary | The milestone is about honest brownfield expansion, not a new product surface or benchmark-only harness | — Pending |

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
*Last updated: 2026-03-27 after starting milestone v1.6*
