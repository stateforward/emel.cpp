# EMEL

## What This Is

EMEL is a deterministic C++ inference engine built around Boost.SML orchestration, with behavior
modeled as explicit actors instead of ad hoc control flow. Shipped v1.1 now proves two narrow
brownfield outcomes: a parity-checked Llama-68M generation slice in `tools/paritychecker` and a
truthful canonical Llama-68M generation benchmark in `tools/bench`, both aligned with
`docs/rules/sml.rules.md`.

## Core Value

Prove real end-to-end behavior with explicit SML orchestration and parity-oriented verification
before widening API surface or model scope.

## Current Milestone: v1.2 flash attention

**Goal:** Add an EMEL-owned flash-attention path for the canonical Llama-68M generation slice, then
verify it through paritychecker and bench under the normal repo surfaces.

**Target features:**
- A real flash-attention execution path in the shipped `src/emel/generator` generation flow for the
  canonical Llama-68M case
- Parity-oriented verification for the flash-attention path through `tools/paritychecker`
- Benchmark visibility for the flash-attention path through the existing `tools/bench` compare flow

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

### Active

- [ ] The canonical Llama-68M generation path can execute through an EMEL-owned flash-attention
  path in `src/emel/generator` and associated `src/emel/kernel` code.
- [ ] `tools/paritychecker` can verify the flash-attention path on the canonical
  `tests/models/Llama-68M-Chat-v1-Q2_K.gguf` workload without broadening into a new API surface.
- [ ] `tools/bench` can compare the canonical EMEL flash-attention path against `llama.cpp`
  through the existing compare workflow.

### Out of Scope

- Broad repository cleanup unrelated to a milestone goal
- Snapshot churn or architecture-doc churn that is not explicitly approved for update
- Non-paritychecker product surfaces until a milestone explicitly broadens the acceptance boundary
- GPU-specific flash-attention work outside the current CPU-hosted canonical generation slice
- Multi-model flash-attention rollout before the canonical Llama-68M path is correct and benchmarked

## Current State

Shipped version: `v1.1`

- 7 phases and 15 plans delivered the first canonical Llama-68M parity slice in v1.0.
- 4 additional phases and 10 plans delivered the truthful benchmark slice in v1.1.
- The audited E2E flow is `paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1`.
- The audited benchmark flow is `scripts/bench.sh --compare` with the canonical row
  `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`.
- Current non-blocking debt is narrow: the low-iteration snapshot smoke check is noisy, benchmark
  drift is still non-blocking in `scripts/quality_gates.sh`, and the shipped tool-local generation
  slice still derives some tokenizer/vocab metadata from the explicit `llama.cpp` reference model
  before initialize.

## Next Milestone Goals

- Land flash attention first in the canonical paritychecked Llama-68M generation path.
- Reuse the existing paritychecker and bench surfaces instead of inventing a broader milestone
  boundary.
- Keep correctness first, then evaluate the benchmark delta from the new path.

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
| Start v1.2 with flash attention in paritychecker plus bench | This keeps the next milestone narrow, measurable, and aligned with the shipped generation surfaces | — Pending |

---
*Last updated: 2026-03-11 after starting milestone v1.2*
