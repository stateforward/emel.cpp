# EMEL

## What This Is

EMEL is a deterministic C++ inference engine built around Boost.SML orchestration, with behavior
modeled as explicit actors instead of ad hoc control flow. The active planning scope is v1.9,
which adds one truthful maintained LiquidAI `LFM2.5-1.2B-Thinking-GGUF` ARM slice on top of the
repo's existing maintained Llama ARM and canonical Qwen anchors, all aligned with
`docs/rules/sml.rules.md`.

## Core Value

Prove real end-to-end behavior with explicit SML orchestration and parity-oriented verification
before widening API surface or model scope.

## Current State

Shipped version: `v1.7`

Status: Milestone v1.9 is complete on this branch. The shipped repo now includes one maintained
Liquid `lfm2` ARM slice alongside the existing maintained Llama ARM and canonical Qwen surfaces.

Release summary: `v1.9 Liquid LFM2.5-1.2B Thinking ARM Slice` shipped on `2026-04-02` with `8`
phases after gap-closure reconstruction. Remaining open debt is broader Liquid coverage, richer
Liquid prompting, further Liquid prefill optimization, and the existing warning-only benchmark
drift outside the maintained Liquid slice.

## Latest Milestone: v1.9 Liquid LFM2.5-1.2B Thinking ARM Slice

**Goal:** Prove one truthful maintained LiquidAI `LFM2.5-1.2B-Thinking-GGUF` ARM slice through the
existing EMEL generator, paritychecker, and benchmark workflow, with `Q4_K_M` as the maintained
truth anchor and without broadening into generic Liquid-family support.

**Target features:**
- One official `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture with reproducible provenance under
  `tests/models/`
- One explicit Liquid request-conditioning contract derived from the official primary chat template
- Explicit `lfm2` runtime bring-up on the maintained generator path
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
- ✓ v1.9 documented one official maintained `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture and bound
  the maintained Liquid path to explicit executable metadata truth plus one explicit structured
  chat contract.
- ✓ v1.9 added explicit `lfm2` model/runtime support for one maintained Liquid ARM slice through
  the shipped generator path.
- ✓ v1.9 proved maintained Liquid parity against `llama.cpp`, preserved additive maintained Qwen
  coverage, and published one truthful Liquid benchmark/docs path aligned with the parity-backed
  slice.
- ✓ v1.9 repaired its missing closeout artifacts, validation layer, and milestone bookkeeping so
  the milestone could pass audit and archive cleanly.

### Active

- [ ] `MODEL-03`: Broaden beyond the canonical `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture to
  additional Liquid checkpoints or sibling quantizations after the first slice is proven.
- [ ] `COND-04`: Add richer Liquid system-message, multi-turn thinking-history, or tool-calling
  request surfaces after the canonical maintained slice has a stable contract.
- [ ] `GEN-04`: Optimize Liquid-specific runtime hot spots now that the canonical slice is correct,
  parity-backed, and benchmarked.
- [ ] `BENCH-07`: Revisit whether noisy benchmark drift should remain warning-only once the
  maintained compare surfaces are stable enough to justify a harder gate.

### Out of Scope

- Decode extraction or broader generator-family decomposition without an explicit milestone goal
- Attention-family `sm_any` extraction before the remaining generator routing shape is clear
- Broad repository cleanup unrelated to a milestone goal
- Broad Liquid-family support beyond one maintained `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` slice
- Sibling Liquid quantizations, including `Q4_0`, `Q5_K_M`, `Q6_K`, `Q8_0`, `BF16`, or `F16`, without explicit
  later-milestone approval
- Tool use, function calling, or multi-turn thinking-history replay on the maintained Liquid path
- Broad Qwen-family or multi-fixture support without an explicit maintained identity and milestone
  goal
- Broad new public C API or CLI surface without explicit milestone approval
- Non-ARM backend kernel specialization until a milestone explicitly broadens beyond the current
  maintained truth anchors
- Whole-program actor or orchestration rewrites unrelated to a milestone acceptance surface
- Dequantize-to-f32 or tool-only compute fallbacks in the shipped canonical hot path without
  explicit milestone approval

## Next Milestone Goals

- Broaden Liquid support beyond the first maintained `Q4_K_M` truth anchor only when the next
  milestone explicitly approves it.
- Improve Liquid prompting breadth and runtime performance without weakening the actor-model and
  maintained-proof constraints.
- Revisit benchmark-gate hardening only after the latest-reference compare lane is stable enough to
  stop producing warning-only drift.

## Context

This is a brownfield repository with an existing codebase map under `.planning/codebase/`. v1.7
proved that generator decomposition can stay inside the `generator` domain and preserve explicit
same-RTC event handoff instead of falling back to context phase flags or hidden helper routing. The
new v1.9 scope is a different kind of widening: the repo currently treats `llama` and `qwen3` as
its explicit maintained model architectures, while official Liquid sources identify the target
fixture as `lfm2` with a hybrid block contract and a primary chat template that includes
`keep_past_thinking`. The repo remains governed by `AGENTS.md` and `docs/rules/sml.rules.md`, so
future work still needs to preserve same-RTC actor semantics, explicit error publication, bounded
actions, and deliberate machine-structure changes.

## Constraints

- **Architecture**: Follow `docs/rules/sml.rules.md` and the local machine conventions in
  `AGENTS.md` so runtime/model work preserves the RTC actor model and no-queue invariant.
- **Explicit behavior modeling**: Any Liquid-specific orchestration changes must keep control flow
  on guards, states, typed events, and transitions rather than action/detail routing shortcuts.
- **Truth anchor**: v1.9 is one official `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture, not generic
  Liquid-family or multi-quant support.
- **Conditioning**: The maintained Liquid path must use one explicit structured chat-message
  contract and must not fall back to raw prompting silently.
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
| Keep v1.9 fixed to one official Liquid Thinking GGUF slice | The repo needs one truthful maintained Liquid anchor before any broader family claims | ✓ Good |
| Use GGUF/config metadata as the maintained truth source for Liquid | Official prose and executable metadata disagree on context length, so docs must follow executable truth | ✓ Good |
| Start v1.9 with `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` | The user explicitly reprioritized the milestone around the docs-recommended quant, accepting that this broadens runtime scope beyond the prior `Q8_0` anchor | ✓ Good |
| Keep the maintained Liquid contract on `tools=none` and no thinking-history replay | Tool use and `keep_past_thinking` widen the request surface beyond the first-slice milestone | ✓ Good |

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
*Last updated: 2026-04-02 after completing v1.9*
