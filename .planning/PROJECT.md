# EMEL

## What This Is

EMEL is a deterministic C++ inference engine built around Boost.SML orchestration, with behavior
modeled as explicit actors instead of ad hoc control flow. The latest shipped surfaces now include
the truthful maintained Qwen3 executable-size publication from v1.8 and the maintained Liquid
LFM2 ARM slice from v1.9, all aligned with `docs/rules/sml.rules.md`.

## Core Value

Prove real end-to-end behavior with explicit SML orchestration and parity-oriented verification
before widening API surface or model scope.

## Current State

Active milestone: `v1.10`

Status: `v1.10` is now active for a planner-family hard cutover that brings the batch planner and
its child mode submachines into explicit alignment with `AGENTS.md` naming conventions and machine
rules.

Release summary: `v1.9 Liquid LFM2.5-1.2B Thinking ARM Slice` shipped on `2026-04-02` with `8`
phases after gap-closure reconstruction. `v1.10` shifts focus from maintained model-slice
publication back to planner-family architecture conformance.

## Current Milestone: v1.10 Planner Family AGENTS Hard Cutover

**Goal:** Hard-cut `src/emel/batch/planner` and its child planner-mode submachines over to the
`AGENTS.md` naming, layout, and SML orchestration contract without broadening the change into
unrelated machine families.

**Target features:**
- Rename and reorganize the planner-family files, namespaces, aliases, and public type names so
  they follow the component/file-base conventions in `AGENTS.md`.
- Bring planner and planner-mode state machines into explicit rule compliance for destination-first
  transition layout, event naming, context ownership, and same-RTC internal handoff semantics.
- Preserve planner behavior with focused proof so the hard cutover does not silently change the
  maintained batching contract.
- Keep the milestone bounded to `src/emel/batch/planner` and its child mode submachines rather
  than widening into generator child machines or broader repository cleanup.

## Previous Milestone: v1.9 Liquid LFM2.5-1.2B Thinking ARM Slice

**Goal:** Prove one truthful maintained LiquidAI `LFM2.5-1.2B-Thinking-GGUF` ARM slice through the
existing EMEL generator, paritychecker, and benchmark workflow, with `Q4_K_M` as the maintained
truth anchor and without broadening into generic Liquid-family support.

**Target features:**
- One official `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture with reproducible provenance under
  `tests/models/`
- One explicit Liquid request-conditioning contract derived from the official primary chat template
- Explicit `lfm2` runtime bring-up on the maintained generator path
- Maintained parity, regression protection, and benchmark publication for the same slice

## Previous Milestone: v1.8 Truthful Qwen3 E2E Embedded Size

<details>
<summary>Shipped on 2026-04-02</summary>

**Goal:** Publish one truthful maintained executable-size comparison for the canonical
`Qwen3-0.6B-Q8_0.gguf` slice using final end-to-end runner executables, explicit smoke proof, and
generated README docs without weakening the workload boundary or claiming whole-product parity.

**Delivered:**
- Locked one maintained workload boundary on `tests/models/Qwen3-0.6B-Q8_0.gguf`,
  structured `hello`, and `max_tokens=1`.
- Corrected the EMEL probe into a truthful final-executable measurement and removed redundant
  fallback-vocab bloat that had inflated the binary into the 56 MB range.
- Kept the published comparator set narrow to EMEL and one matched `llama.cpp` reference
  executable with shared `runtime_smoke=passed` proof.
- Refreshed `snapshots/embedded_size/summary.txt` and the generated `README.md` so the published
  result reports `4,073,016` raw bytes for EMEL versus `3,334,264` for the reference row.
- Backfilled the missing proof chain and closed the milestone with a passing `8/8` requirements
  audit.

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
- ✓ v1.8 published one truthful maintained Qwen3 executable-size comparison on final linked
  executables for the canonical `hello` -> first-token path.
- ✓ v1.8 kept the published EMEL row EMEL-owned end to end, with no `llama.cpp` bootstrap for the
  maintained probe path.
- ✓ v1.8 refreshed the stored embedded-size snapshot and generated README evidence to the corrected
  EMEL/reference executable numbers and closed with a passing `8/8` milestone audit.
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

- [ ] Cut over `src/emel/batch/planner` to the `AGENTS.md` naming and file-layout contract.
- [ ] Cut over planner-mode child machines in `src/emel/batch/planner/modes/` to the same
  contract without introducing hidden control flow or context phase flags.
- [ ] Preserve the maintained batching behavior with explicit proof after the structural cutover.
- [ ] Leave generator child machines, broad repository cleanup, and non-planner family refactors
  out of this milestone unless a later milestone explicitly approves them.

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

- After the planner-family hard cutover lands, decide whether the next milestone should continue
  architecture-contract cleanup into other machine families or return to maintained runtime/model
  scope.
- Revisit the queued Liquid runtime and benchmark hardening todos once the planner-family structure
  is aligned with the stricter `AGENTS.md` contract.

## Context

This is a brownfield repository with an existing codebase map under `.planning/codebase/`. The repo
remains governed by `AGENTS.md` and `docs/rules/sml.rules.md`, and the user has now explicitly
requested a planner-family hard cutover to those conventions. The planner domain already exists as
`src/emel/batch/planner` with child mode submachines under `modes/`; the current milestone is to
make that family the contract-aligned reference without broadening into unrelated machine families.

## Constraints

- **Architecture**: Follow `docs/rules/sml.rules.md` and the local machine conventions in
  `AGENTS.md` so planner work preserves the RTC actor model and no-queue invariant.
- **Scope boundary**: This milestone is limited to `src/emel/batch/planner` and its child mode
  submachines, not generator child machines or whole-repo cleanup.
- **Explicit behavior modeling**: The cutover must keep runtime control flow in guards, states,
  typed events, and transitions rather than action/detail routing shortcuts.
- **Proof boundary**: Structural renames or file moves are not sufficient by themselves; behavior
  preservation needs focused proof before the milestone can close.
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
| Scope v1.10 to the planner family only | The user asked for planner and planner-submachine compliance, not a broader machine-family rewrite | — Pending |
| Treat structural cutover as incomplete without proof preservation | AGENTS compliance changes must not silently break the maintained batching contract | — Pending |

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
*Last updated: 2026-04-04 after starting v1.10*
