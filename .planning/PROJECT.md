# EMEL Llama-68M Generation Slice

## What This Is

EMEL is a deterministic C++ inference engine built around Boost.SML orchestration, with behavior
modeled as explicit actors instead of ad hoc control flow. This planning cycle focuses on one
narrow brownfield outcome: proving an end-to-end Llama-68M generation path inside
`tools/paritychecker` while following `docs/rules/sml.rules.md`.

## Core Value

Prove a real end-to-end generation path with explicit SML orchestration and parity-oriented
verification before widening API surface or model scope.

## Requirements

### Validated

- ✓ The codebase is organized around Boost.SML actor machines with deterministic same-RTC
  orchestration in `src/emel/**/sm.hpp` — existing
- ✓ EMEL already has machine families for GGUF loading, model loading, text conditioning,
  graph execution, memory, logits sampling, and generation orchestration under `src/emel/`
  — existing
- ✓ The repository already includes parity and benchmark tooling in `tools/paritychecker/`
  and `tools/bench/`, with local test models under `tests/models/` — existing
- ✓ The project already treats `docs/rules/sml.rules.md` as the governing orchestration contract
  and generates architecture documentation from the machine headers — existing

### Active

- [ ] Load `tests/models/Llama-68M-Chat-v1-Q2_K.gguf` through the existing model path and make it
  available to a paritychecker-driven generation run
- [ ] Execute one end-to-end generation flow in `tools/paritychecker/` that covers prompt
  conditioning, graph execution, sampling, and rendered output
- [ ] Keep the generation path compliant with `docs/rules/sml.rules.md`, especially the RTC
  actor model, no-queue invariant, explicit unexpected-event handling, and phase-bounded internal
  completion chains
- [ ] Add paritychecker-focused verification that demonstrates the Llama-68M path works without
  requiring a new public C API example

### Out of Scope

- Public C API expansion or a standalone end-user example program — paritychecker is the target
  acceptance boundary for this slice
- Broad multi-model support beyond the minimum required to make the Llama-68M path work
  — this cycle optimizes for one proven vertical slice first
- Large-scale repository cleanup unrelated to the generation path — only supporting changes needed
  to satisfy the slice and SML rules belong in scope

## Context

This is a brownfield repository with an existing codebase map under `.planning/codebase/`.
`README.md` positions EMEL as a production-grade but still WIP inference engine that prioritizes
explicit state-machine semantics, parity against `llama.cpp`, and deterministic behavior over early
API breadth. The architecture already includes the major subsystems needed for generation:
`src/emel/gguf/loader/`, `src/emel/model/loader/`, `src/emel/text/`, `src/emel/graph/`,
`src/emel/memory/`, `src/emel/logits/`, and `src/emel/generator/`.

The chosen target model is documented in `tests/models/README.md` as
`Llama-68M-Chat-v1-Q2_K.gguf`, a 34 MB Apache-2.0 GGUF artifact. Acceptance is intentionally
narrow: success means a real end-to-end generation path works inside `tools/paritychecker/`, not
that the public C ABI or general application examples are complete.

The repo has strong local engineering constraints from `AGENTS.md` and `docs/rules/sml.rules.md`.
Those rules materially affect planning: runtime branching must be expressed through guards and
states, actions must stay bounded and allocation-free during dispatch, event data should flow
through typed internal events instead of context mirroring, and any structural change to a machine
should be treated deliberately.

## Constraints

- **Architecture**: Follow `docs/rules/sml.rules.md` and the local machine conventions in
  `AGENTS.md` — the generation slice must preserve the RTC actor model and no-queue invariant
- **Acceptance boundary**: Verification must land in `tools/paritychecker/` rather than a new C
  API example — that is the agreed definition of done for this slice
- **Model scope**: Use `tests/models/Llama-68M-Chat-v1-Q2_K.gguf` as the primary target
  — the first slice should prove one concrete model path before generalization
- **Performance philosophy**: Favor explicit orchestration correctness first, then optimize
  once parity-oriented behavior is locked, consistent with `README.md`
- **Repository state**: This is an active brownfield codebase with ongoing local changes, so plans
  should minimize unrelated churn and keep work scoped to the generation path

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Target the Llama-68M GGUF model first | It is a small local test model already present in `tests/models/` and is suitable for proving a narrow vertical slice | — Pending |
| Define done in paritychecker, not the public C API | The immediate goal is parity-oriented end-to-end generation proof, not API surface expansion | — Pending |
| Optimize for a narrow vertical slice | This reduces planning noise and avoids turning the first proof into a general cleanup campaign | — Pending |
| Enforce SML rules as a first-class planning constraint | The user explicitly wants the slice to follow `docs/rules/sml.rules.md`, and the repo architecture is built around that contract | — Pending |

---
*Last updated: 2026-03-07 after initialization*
