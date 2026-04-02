# EMEL

## What This Is

EMEL is a deterministic C++ inference engine built around Boost.SML orchestration, with behavior
modeled as explicit actors instead of ad hoc control flow. The active planning scope is `v1.11`,
which adds one truthful maintained Gemma 4 E2B text-generation slice on top of adjacent `v1.8`
size-benchmark work, merged `v1.9` Liquid/LFM2.5 support, and inflight `v2.0` Bonsai `Q1`
kernel work.

## Core Value

Prove real end-to-end behavior with explicit SML orchestration and parity-oriented verification
before widening API surface or model scope.

## Current State

Latest archived milestone: `v1.7`

Status: Milestone `v1.11` is defined and ready for phase planning on this branch. The milestone
archive still only closes through `v1.7`, but current user context for new work also includes
adjacent `v1.8` size benchmarks, merged `v1.9` Liquid/LFM2.5 support, and inflight `v2.0`
Bonsai `Q1` kernel work that are not the scope of this milestone.

Release summary: `v1.7 Generator Prefill Submachine Decomposition` shipped on `2026-03-30` with
`3` phases and `6` plans. Remaining adjacent work includes benchmark-surface widening, Liquid
follow-on hardening, and Bonsai kernel work outside this new Gemma 4 acceptance surface.

## Current Milestone: v1.11 Gemma 4 E2B Text Generation Slice

**Goal:** Prove one truthful maintained text-generation slice for the brand new official
`ggml-org/gemma-4-E2B-it-GGUF` release through EMEL's existing generator, paritychecker, and
benchmark workflow, anchored on `gemma-4-e2b-it-Q8_0.gguf` and with explicit `gemma4`
model/runtime handling.

**Target features:**
- One official `gemma-4-e2b-it-Q8_0.gguf` fixture with reproducible provenance under
  `tests/models/`
- One explicit Gemma 4 text-only request-conditioning contract derived from the official
  `chat_template`
- Explicit `gemma4` model/runtime bring-up on the maintained generator path
- Maintained parity, regression protection, and benchmark publication for the same text slice
- Explicit non-support for `mmproj`, image/audio/video inputs, and tool-call surfaces in this
  milestone

## Latest Milestone: v1.7 Generator Prefill Submachine Decomposition

<details>
<summary>Shipped on 2026-03-30</summary>

**Goal:** Decompose the generator's prefill orchestration into a smaller explicit machine inside
the `generator` domain while keeping request-scoped behavior modeled via typed events, guards, and
transitions instead of hidden action/detail control flow.

**Delivered:**
- Collapsed the repeated top-level prefill routing matrix into explicit request-scoped prefill
  compute contracts.
- Extracted `src/emel/generator/prefill` as the first generator-owned orchestration submachine for
  prefill slots, snapshot, compute dispatch, and handoff.
- Kept prefill request-scoped orchestration data on typed runtime/internal events instead of
  generator context flags or phase fields.
- Materially shrank the top-level generator surface and published split `generator_prefill`
  architecture docs.
- Re-proved maintained generator, paritychecker, and compare behavior on the extracted prefill
  boundary while carrying forward the existing warning-only benchmark policy.

</details>

## Requirements

### Validated

- ✓ v1.0 proved one canonical `Llama-68M-Chat-v1-Q2_K.gguf` slice end to end through
  `tools/paritychecker` with real GGUF/model loading, bounded generation, and subprocess coverage.
- ✓ v1.1 added one truthful canonical generation benchmark in `tools/bench`, published through the
  existing compare, snapshot, and docsgen workflow.
- ✓ v1.2 shipped an EMEL-owned flash-attention path plus hard cutover to `emel::tensor::sm` on the
  canonical Llama generation slice.
- ✓ v1.3 shipped optimized AArch64 flash execution and maintained optimized-vs-shared attribution
  on the canonical ARM workload.
- ✓ v1.4 shipped EMEL-owned vectorized AArch64 `q2_K/q3_K/q6_K x q8_K` hot-path kernels and
  maintained `1/10/100/1000` parity on the canonical ARM slice.
- ✓ v1.5 closed the canonical ARM quantized-path contract at
  `native_quantized=8 approved_dense_f32_by_contract=4 disallowed_fallback=0 explicit_no_claim=0`
  and restored checked-in flash publication so stored evidence matches live proof.
- ✓ v1.6 documented one official canonical `Qwen3-0.6B-Q8_0.gguf` fixture and bound the maintained
  Qwen path to one explicit GGUF-derived conditioning contract with structured chat-message input
  and no implicit raw fallback.
- ✓ v1.6 added native `src/emel` `q8_0` runtime support plus explicit canonical `qwen3` topology
  handling on the shipped generator path.
- ✓ v1.6 proved maintained canonical Qwen parity against `llama.cpp` and protected the prior Llama
  anchor with stored compare coverage on `1/10/100/1000`, `--dump`, and `--attribution`.
- ✓ v1.6 published one truthful canonical Qwen benchmark compare/docs path aligned with the same
  parity-backed formatter/runtime contract.
- ✓ The v1.6 milestone audit passed with `8/8` requirements satisfied and only tech-debt findings.
- ✓ v1.7 extracted an explicit `src/emel/generator/prefill` orchestration machine for prefill
  slots, snapshot, compute dispatch, and handoff.
- ✓ v1.7 kept prefill request-scoped orchestration data on typed runtime/internal events instead of
  generator context phase fields.
- ✓ v1.7 collapsed the old top-level prefill routing matrix into explicit request-scoped prefill
  compute contracts.
- ✓ v1.7 preserved explicit route selection through guards, states, and transitions with no hidden
  action/detail routing branch added by the decomposition.
- ✓ v1.7 materially shrank the top-level generator surface and published the split
  `generator_prefill` architecture.
- ✓ v1.7 preserved maintained generator, paritychecker, and compare proof; unrelated broad
  benchmark snapshot regressions outside `generator/prefill` were explicitly waived for closeout.

### Active

- [ ] `FIX-03`: Document one official `gemma-4-e2b-it-Q8_0.gguf` maintained fixture with checksum,
  source, stable path, and download URL.
- [ ] `COND-04`: Add one explicit Gemma 4 text-only request-conditioning contract derived from the
  official `chat_template`, with structured text chat messages, `add_generation_prompt=true`, and
  no implicit raw fallback.
- [ ] `RUN-07`: Add truthful `gemma4` architecture/runtime support for one maintained Gemma 4 E2B
  text slice on the shipped generator path.
- [ ] `REF-01`: Upgrade or confirm the pinned `llama.cpp` reference lane so the canonical Gemma 4
  fixture can be loaded by `tools/paritychecker` and `tools/bench`.
- [ ] `PAR-03`: Prove the maintained Gemma 4 text slice against `llama.cpp` on the same fixture and
  conditioning contract.
- [ ] `BENCH-09`: Publish one truthful Gemma 4 benchmark compare/docs path for the same
  parity-backed maintained text slice.

### Out of Scope

- `mmproj-gemma-4-e2b-it-f16.gguf` and any image/audio/video request path in this milestone
- Tool use, function calling, or assistant tool-call replay on the maintained Gemma 4 path
- Broad Gemma 4 family or multi-fixture support beyond one maintained `gemma-4-e2b-it-Q8_0.gguf`
  slice
- The `gemma-4-e2b-it-f16.gguf` sibling or any future Gemma 4 quant matrix without explicit later
  milestone approval
- Raw prompt fallback on the maintained Gemma 4 path
- Re-planning adjacent `v1.8` size-benchmark work or inflight `v2.0` Bonsai `Q1` kernel work as
  part of Gemma 4 milestone acceptance
- Broad new public C API or CLI surface without explicit milestone approval
- Whole-program actor or orchestration rewrites unrelated to the maintained Gemma 4 acceptance
  surface
- Dequantize-to-f32 or tool-only compute fallbacks in the shipped canonical hot path without
  explicit milestone approval

## Next Milestone Goals

- Land one truthful maintained Gemma 4 E2B text-generation slice on the existing generator,
  paritychecker, and benchmark surfaces.
- Preserve maintained Llama, Qwen, and merged Liquid proof while Gemma 4 support is added.
- Keep the milestone narrow to one explicit `gemma4` fixture, one explicit text-only conditioning
  contract, and one explicit reference-lane readiness check.
- Revisit `mmproj`, multimodal input support, broader Gemma 4 coverage, and deeper benchmark-size
  integration only after the Gemma 4 text slice is parity-backed and benchmarked.

## Context

This is a brownfield repository with an existing codebase map under `.planning/codebase/`.
Generator decomposition work is already in place, and merged Liquid/LFM2.5 work widened the
repo's maintained model surface beyond `llama` and `qwen3`. The new `v1.11` scope is different:
the official `ggml-org/gemma-4-E2B-it-GGUF` release is brand new as of `2026-04-02`,
`general.architecture=gemma4`, `context_length=131072`, and currently ships only
`gemma-4-e2b-it-Q8_0.gguf`, `gemma-4-e2b-it-f16.gguf`, and a separate
`mmproj-gemma-4-e2b-it-f16.gguf` companion file. The base model is explicitly multimodal
(`Gemma4ForConditionalGeneration` with image/audio/video token ids), so this milestone must keep
the maintained acceptance surface honest by treating Gemma 4 as a text-generation slice only until
EMEL has a real multimodal pipeline.

The current repo still follows `AGENTS.md` and `docs/rules/sml.rules.md`, so future work must
preserve same-RTC actor semantics, explicit error publication, bounded actions, and deliberate
machine-structure changes. The existing pinned `llama.cpp` reference commit used by
`tools/paritychecker` and `tools/bench` does not appear to contain `gemma4`, while current upstream
`llama.cpp` master does, so reference-lane readiness is part of the milestone instead of an
assumed precondition.

## Constraints

- **Architecture**: Follow `docs/rules/sml.rules.md` and the local machine conventions in
  `AGENTS.md` so runtime/model work preserves the RTC actor model and no-queue invariant.
- **Explicit behavior modeling**: Any Gemma 4-specific orchestration changes must keep control flow
  on guards, states, typed events, and transitions rather than action/detail routing shortcuts.
- **Truth anchor**: `v1.11` is one official `gemma-4-e2b-it-Q8_0.gguf` fixture, not broad Gemma 4
  family or multimodal support.
- **Scope honesty**: The upstream release is multimodal and ships a separate `mmproj` file, so the
  maintained path must reject unsupported media request shapes explicitly instead of implying they
  work.
- **Conditioning**: The maintained Gemma 4 path must use one explicit text-only structured
  contract derived from the official `chat_template` and must not fall back to raw prompting
  silently.
- **Reference boundary**: Maintained proof still lives on the shipped generator, paritychecker,
  benchmark, snapshot, and docs surfaces, and the pinned `llama.cpp` reference lane may require a
  Gemma 4-capable update.
- **Repository state**: This is an active brownfield codebase with adjacent `v1.8`, `v1.9`, and
  `v2.0` work, so milestone work should minimize unrelated churn and leave non-milestone artifacts
  alone.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Start decomposition with `generator/prefill` instead of splitting the whole generator at once | Prefill was the clearest request-phase boundary and the largest current duplication source | ✓ Good |
| Defer decode extraction until the prefill pattern is proven | Decode also owns sampling, rendering, and loop control, so it was the riskier first cut | ✓ Good |
| Keep `v1.11` fixed to one official `ggml-org/gemma-4-E2B-it-GGUF` slice | The repo needs one truthful maintained Gemma 4 anchor before any broader family or multimodal claims | — Pending |
| Use `gemma-4-e2b-it-Q8_0.gguf` as the maintained fixture | The official GGUF repo currently ships `Q8_0`, `F16`, and a separate `mmproj` file; `Q8_0` is the one quantized text fixture closest to the repo's current maintained runtime surfaces | — Pending |
| Treat `v1.11` as a text-generation slice and defer `mmproj` plus image/audio/video inputs | The upstream model is any-to-any, but EMEL's current maintained acceptance boundary is still text generator/parity/bench only | — Pending |
| Derive the maintained Gemma 4 contract from the official `chat_template` but keep tools and media disabled | This keeps prompt conditioning truthful without widening into unsupported tool or multimodal surfaces | — Pending |
| Make reference-lane readiness explicit in the milestone | The pinned `llama.cpp` reference commit used by parity/bench appears not to contain `gemma4`, so parity would otherwise be blocked by an unstated assumption | — Pending |
| Keep milestone label `v1.11` for this work | The user explicitly wants this Gemma 4 milestone tracked as `v1.11` even though adjacent work also references `v2.0` | — Pending |
| Continue roadmap numbering at Phase 38 | The current visible roadmap ceiling in `.planning/phases/` is Phase 37 | — Pending |

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
*Last updated: 2026-04-02 after starting v1.11*
