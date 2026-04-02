# EMEL

## What This Is

EMEL is a deterministic C++ inference engine built around Boost.SML orchestration, with behavior
modeled as explicit actors instead of ad hoc control flow. The active planning scope is v1.10,
which adds one truthful maintained Prism ML `Bonsai-1.7B.gguf` slice on top of the repo's existing
maintained Llama ARM and canonical Qwen anchors, while preserving the engineering constraints in
`docs/rules/sml.rules.md` and `AGENTS.md`.

## Core Value

Prove real end-to-end behavior with explicit SML orchestration and parity-oriented verification
before widening API surface or model scope.

## Current State

Shipped version: `v1.7`

Status: Milestone v1.10 is being defined on this branch while v1.8 and v1.9 continue separately in
flight. The shipped repo still reflects v1.7 plus the existing maintained Llama ARM and canonical
Qwen acceptance surfaces.

Release summary: `v1.7 Generator Prefill Submachine Decomposition` shipped on `2026-03-30` with
`3` phases and `6` plans. Remaining open debt is decode-side generator decomposition plus the
existing warning-only benchmark drift outside `generator/prefill`.

## Current Milestone: v1.10 Bonsai 1.7B 1-Bit Bring-Up

**Goal:** Prove one truthful maintained Prism ML `Bonsai-1.7B.gguf` slice through the existing
EMEL generator, paritychecker, and benchmark workflow, with the live Hugging Face artifact as the
truth anchor and without broadening into generic 1-bit or custom-GGUF-family support.

**Target features:**
- One official `Bonsai-1.7B.gguf` fixture with reproducible provenance under `tests/models/`
- One explicit Bonsai request-conditioning contract derived from the shipped tokenizer/chat
  metadata
- Truthful EMEL model-loading and runtime support for the Bonsai 1-bit `Q1_0_g128` operand path
- Maintained parity, regression protection, and benchmark publication for the same slice

## Latest Milestone: v1.7 Generator Prefill Submachine Decomposition

<details>
<summary>Shipped on 2026-03-30</summary>

**Goal:** Decompose the generator’s prefill orchestration into a smaller explicit machine inside
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
- ✓ v1.6 proved maintained canonical Qwen parity against `llama.cpp` and protected the prior
  Llama anchor with stored compare coverage on `1/10/100/1000`, `--dump`, and `--attribution`.
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

- [ ] `FIX-03`: Document one official `Bonsai-1.7B.gguf` maintained fixture with checksum, source,
  stable path, and direct Hugging Face download URL.
- [ ] `META-02`: Record executable model truth for the Bonsai slice from live GGUF metadata and the
  Hugging Face repository state, including the actual file name, architecture family, context
  length, and 1-bit quant format.
- [ ] `COND-04`: Add one explicit Bonsai request-conditioning contract derived from the maintained
  tokenizer/chat metadata, with no silent fallback to an unrelated existing formatter surface.
- [ ] `RUN-07`: Add truthful EMEL-owned runtime support for the maintained Bonsai slice on the
  shipped generator path.
- [ ] `PAR-03`: Prove the maintained Bonsai slice against the designated reference path on the same
  fixture and conditioning contract.
- [ ] `BENCH-09`: Publish one truthful Bonsai benchmark compare/docs path for the same parity-
  backed maintained slice.

### Out of Scope

- Decode extraction or broader generator-family decomposition without an explicit milestone goal
- Attention-family `sm_any` extraction before the remaining generator routing shape is clear
- Broad repository cleanup unrelated to a milestone goal
- Broad Prism/Bonsai-family support beyond one maintained `Bonsai-1.7B.gguf` slice
- Alternate Bonsai export formats, sibling checkpoints, or future quant variants without explicit
  later-milestone approval
- Generic custom-GGUF or arbitrary third-party model support without an explicit milestone goal
- Tool use, function calling, or multi-turn agent workflows on the maintained Bonsai path
- Broad Qwen-family or multi-fixture support without an explicit maintained identity and milestone
  goal
- Broad new public C API or CLI surface without explicit milestone approval
- Non-ARM backend kernel specialization until a milestone explicitly broadens beyond the current
  maintained truth anchors
- Whole-program actor or orchestration rewrites unrelated to a milestone acceptance surface
- Dequantize-to-f32 or tool-only compute fallbacks in the shipped canonical hot path without
  explicit milestone approval

## Next Milestone Goals

- Land one truthful maintained Prism ML `Bonsai-1.7B.gguf` slice.
- Preserve maintained Llama/Qwen proof while Bonsai support is added.
- Keep the milestone narrow to one explicit Bonsai fixture and one explicit conditioning contract.
- Decide explicitly whether EMEL will support the Bonsai-specific `Q1_0_g128` path natively before
  making parity or benchmark claims.

## Context

This is a brownfield repository with an existing codebase map under `.planning/codebase/`. v1.7
proved that generator decomposition can stay inside the `generator` domain and preserve explicit
same-RTC event handoff instead of falling back to context phase flags or hidden helper routing. The
new v1.10 scope widens the maintained model surface in a different direction: the live Hugging Face
repo `prism-ml/Bonsai-1.7B-gguf` was published on `2026-03-31`, exposes one file named
`Bonsai-1.7B.gguf`, describes a Qwen3-1.7B-based dense architecture, and advertises a custom 1-bit
`Q1_0_g128` GGUF weight format with support currently demonstrated through Prism's `llama.cpp`
fork. The repo remains governed by `AGENTS.md` and `docs/rules/sml.rules.md`, so future work still
needs to preserve same-RTC actor semantics, explicit error publication, bounded actions, and
deliberate machine-structure changes.

## Constraints

- **Architecture**: Follow `docs/rules/sml.rules.md` and the local machine conventions in
  `AGENTS.md` so runtime/model work preserves the RTC actor model and no-queue invariant.
- **Explicit behavior modeling**: Any Bonsai-specific orchestration changes must keep control flow
  on guards, states, typed events, and transitions rather than action/detail routing shortcuts.
- **Truth anchor**: v1.10 is one official `Bonsai-1.7B.gguf` fixture, not generic 1-bit or
  third-party GGUF support.
- **Conditioning**: The maintained Bonsai path must use one explicit structured chat-message
  contract and must not fall back to raw prompting or an unrelated existing formatter silently.
- **Runtime honesty**: The milestone must not claim Bonsai support unless EMEL either implements
  the required `Q1_0_g128` path or explicitly proves an approved alternative with user sign-off.
- **Acceptance boundary**: Maintained proof still lives on the shipped generator, paritychecker,
  benchmark, snapshot, and docs surfaces.
- **Repository state**: This is an active brownfield codebase, so milestone work should minimize
  unrelated churn and leave non-milestone artifacts alone.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Start decomposition with `generator/prefill` instead of splitting the whole generator at once | Prefill was the clearest request-phase boundary and the largest current duplication source | ✓ Good |
| Collapse the prefill compute-routing matrix before broader extraction | File movement alone would have just relocated the cartesian product instead of reducing it | ✓ Good |
| Keep prefill request-scoped data on typed runtime/internal events | This preserved explicit behavior modeling and avoided context phase flags | ✓ Good |
| Defer decode extraction until the prefill pattern is proven | Decode also owns sampling, rendering, and loop control, so it was the riskier first cut | ✓ Good |
| Defer `attention::any` / `sm_any` extraction until after prefill collapse | Attention mode is only one axis of the duplication and should not hide unresolved top-level routing | ⚠ Revisit |
| Start Bonsai as v1.10 while v1.8 and v1.9 continue in flight | The user wants Bonsai tracked as the next minor milestone rather than a major-version reset | — Pending |
| Keep v1.10 fixed to one official `Bonsai-1.7B.gguf` slice | The repo needs one truthful maintained Bonsai anchor before any broader 1-bit or vendor claims | — Pending |
| Use live Hugging Face repo state plus executable GGUF metadata as the maintained truth source for Bonsai | The model card README, quickstart snippets, and file tree already disagree on the published filename, so docs must follow executable truth | — Pending |
| Treat `Q1_0_g128` as milestone-defining scope, not an incidental detail | Prism describes Bonsai support through a custom 1-bit operand path; the milestone is not honest if it ignores that kernel/runtime requirement | — Pending |
| Keep the maintained Bonsai contract on one explicit formatter/request surface first | Multi-turn or tool-rich Bonsai workflows would widen the request boundary before the first maintained slice is proven | — Pending |

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
*Last updated: 2026-04-02 after starting v1.10*
