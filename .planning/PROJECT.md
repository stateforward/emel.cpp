# EMEL

## What This Is

EMEL is a deterministic C++ inference engine built around Boost.SML orchestration, with behavior
modeled as explicit actors instead of ad hoc control flow. Shipped v1.2 now proves three narrow
brownfield outcomes on the canonical Llama-68M slice: a parity-checked generation path in
`tools/paritychecker`, a truthful maintained benchmark workflow in `tools/bench`, and an
EMEL-owned flash-attention runtime path carried through those existing repo surfaces, all aligned
with `docs/rules/sml.rules.md`.

## Core Value

Prove real end-to-end behavior with explicit SML orchestration and parity-oriented verification
before widening API surface or model scope.

## Current Milestone

No active milestone.

**Next action:** Start the next milestone with `$gsd-new-milestone` after deciding whether the next
scope expansion is broader flash coverage, backend specialization, or benchmark-policy hardening.

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

### Active

- [ ] Broaden flash attention beyond the canonical Llama-68M shape and workload contract.
- [ ] Add backend-specific optimized flash-attention implementations where benchmark evidence
  justifies them.
- [ ] Roll flash attention out to additional model fixtures after the canonical path remains
  stable.
- [ ] Decide whether benchmark-policy hardening should become a real milestone requirement instead
  of warning-only repo policy.

### Out of Scope

- Broad repository cleanup unrelated to a milestone goal
- Snapshot churn or architecture-doc churn that is not explicitly approved for update
- Non-paritychecker product surfaces until a milestone explicitly broadens the acceptance boundary
- GPU-specific flash-attention work outside the current CPU-hosted canonical generation slice
- Multi-model flash-attention rollout before the canonical Llama-68M path is correct and benchmarked

## Current State

Shipped version: `v1.2`

- 7 phases and 15 plans delivered the first canonical Llama-68M parity slice in v1.0.
- 4 additional phases and 10 plans delivered the truthful benchmark slice in v1.1.
- 5 additional phases and 13 plans delivered v1.2: flash attention is live in the shipped
  canonical generator path, paritychecker proves it against fetched upstream `llama.cpp`,
  graph/generator runtime ownership is hard-cut through `emel::tensor::sm`, and the maintained
  benchmark workflow now publishes flash evidence plus a preserved pre-flash baseline.
- The validated E2E flows now include
  `paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1`
  and the maintained `scripts/bench.sh --compare` publication path that feeds
  `snapshots/bench/benchmarks_compare.txt` and `docs/benchmarks.md`.
- No active milestone is currently defined; the next milestone should state explicitly whether it
  is broadening model scope, backend scope, or benchmark policy.
- Current non-blocking debt remains narrow: the low-iteration benchmark smoke is noisy on this
  machine, benchmark drift is still warning-only in `scripts/quality_gates.sh`, and broader flash
  claims are intentionally out of scope until a future milestone defines them.

## Next Milestone Goals

- Choose the next deliberate widening step: broader flash operand/model coverage, backend-specific
  optimization, or benchmark-policy hardening.
- Keep acceptance boundaries explicit before broadening product surfaces beyond paritychecker and
  bench.
- Preserve the canonical Llama-68M slice as the truth anchor while any broader rollout is proven.

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

---
*Last updated: 2026-03-22 after shipping v1.2*
