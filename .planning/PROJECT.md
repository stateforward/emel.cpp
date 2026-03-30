# EMEL

## What This Is

EMEL is a deterministic C++ inference engine built around Boost.SML orchestration, with behavior
modeled as explicit actors instead of ad hoc control flow. Shipped v1.6 now proves two truthful
maintained brownfield anchors: the canonical Llama-68M ARM quantized/flash slice and one canonical
`Qwen3-0.6B-Q8_0.gguf` slice with explicit GGUF-derived conditioning, native `src/emel` `q8_0`
runtime support, stored parity, and benchmark publication, all aligned with
`docs/rules/sml.rules.md`.

## Core Value

Prove real end-to-end behavior with explicit SML orchestration and parity-oriented verification
before widening API surface or model scope.

## Current State

Shipped version: `v1.6`

Status: Between milestones. The repo now ships truthful maintained Llama ARM and canonical Qwen
surfaces through the generator, paritychecker, benchmark, snapshot, and docs workflows.

Release summary: `v1.6 Qwen3-0.6B Parity And Benchmark` shipped on `2026-03-30` with `5` phases
and `12` plans. Remaining open debt is milestone-bookkeeping freshness plus benchmark-warning
review under the existing warning-only repo policy.

## Latest Milestone: v1.6 Qwen3-0.6B Parity And Benchmark

<details>
<summary>Shipped on 2026-03-30</summary>

**Goal:** Prove one truthful canonical Qwen3-0.6B GGUF slice through the maintained EMEL
generator, paritychecker, and benchmark workflow without widening the acceptance boundary beyond
that slice.

**Delivered:**
- Locked one official `Qwen3-0.6B-Q8_0.gguf` maintained fixture with explicit provenance and
  stable tool identity.
- Bound the maintained Qwen path to the primary GGUF `tokenizer.chat_template` through an
  explicit structured-message conditioning contract with no implicit raw fallback.
- Added native `src/emel` `q8_0` runtime support and explicit `qwen3` topology handling for the
  canonical slice.
- Proved maintained stored-baseline parity on `1/10/100/1000` while preserving the prior Llama
  anchor.
- Published one truthful canonical Qwen benchmark compare/docs path aligned to the same
  formatter/runtime contract.

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

### Active

- [ ] `MODEL-02`: Broaden beyond the canonical Qwen3-0.6B fixture to additional Qwen
  architectures or quantizations once the first slice is proven and benchmarked.
- [ ] `COND-02`: Add richer Qwen chat or tool-calling request surfaces only after the canonical
  maintained slice has an explicit and stable conditioning contract.
- [ ] `GEN-03`: Optimize remaining generator-side hot spots after the canonical Qwen slice is
  correct, parity-backed, and benchmarked.
- [ ] `BENCH-07`: Revisit whether noisy benchmark drift should become a blocking repo gate once the
  maintained compare surfaces are stable enough to justify it.

### Out of Scope

- Broad repository cleanup unrelated to a milestone goal
- Broad Qwen-family or multi-fixture support without an explicit maintained identity and milestone
  goal
- Broad new public C API or CLI surface without explicit milestone approval
- Non-ARM backend kernel specialization until a milestone explicitly broadens beyond the current
  maintained truth anchors
- Whole-program actor or orchestration rewrites unrelated to a milestone acceptance surface
- Dequantize-to-f32 or tool-only compute fallbacks in the shipped canonical hot path without
  explicit milestone approval

## Next Milestone Goals

- Decide whether the next milestone should broaden beyond the canonical Qwen3-0.6B slice or stay
  focused on performance and debt cleanup on the current maintained truth anchors.
- Add richer Qwen request surfaces only if the next milestone explicitly broadens the conditioning
  contract beyond the current canonical template boundary.
- Optimize remaining generator-side hot spots now that the canonical Qwen slice is parity-backed
  and benchmarked.
- Revisit benchmark drift policy and publication freshness only after the maintained compare
  surfaces are stable enough to justify a stricter gate.

## Context

This is a brownfield repository with an existing codebase map under `.planning/codebase/`. v1.6
showed that the existing subsystem families under `src/emel/` are sufficient to bring up a
non-Llama maintained slice truthfully without widening the public C API surface or hiding
formatter/runtime behavior behind implicit fallbacks. The repo remains governed by `AGENTS.md` and
`docs/rules/sml.rules.md`, so future work still needs to preserve same-RTC actor semantics,
explicit error publication, bounded actions, and deliberate machine-structure changes.
Non-blocking benchmark warnings plus milestone-evidence freshness remain the main carried technical
debt entering the next milestone.

## Constraints

- **Architecture**: Follow `docs/rules/sml.rules.md` and the local machine conventions in
  `AGENTS.md` so generation work preserves the RTC actor model and no-queue invariant.
- **Acceptance boundary**: Milestones should expand intentionally; maintained proof still lives on
  the shipped generator, paritychecker, benchmark, snapshot, and docs surfaces.
- **Model scope**: The repo now has one canonical Llama ARM anchor and one canonical Qwen anchor;
  broader model matrices should still widen only under an explicit milestone goal.
- **Performance philosophy**: Favor explicit orchestration correctness first, then optimize once
  parity-oriented behavior is locked.
- **Repository state**: This is an active brownfield codebase, so milestone work should still
  minimize unrelated churn.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Define done in paritychecker and maintained bench surfaces, not the public C API | The immediate goal is truthful brownfield end-to-end proof, not API surface expansion | ✓ Good |
| Keep milestones narrow to one canonical maintained slice at a time | This avoids overstating support while runtime and conditioning contracts are still being hardened | ✓ Good |
| Keep benchmark maintenance on the existing compare/snapshot/docs surfaces | One operator workflow is more honest than benchmark-only publication paths | ✓ Good |
| Preserve benchmark drift as warning-only until a milestone explicitly hardens the repo gate | This avoids turning noisy publication debt into false-negative release blockers too early | ⚠ Revisit |
| Restore stored evidence whenever live maintained proof changes materially | Stored compare/docs artifacts must match the live proof contract | ✓ Good |
| Keep v1.6 scoped to one canonical `Qwen3-0.6B-Q8_0.gguf` slice | The maintained generator, paritychecker, and benchmark surfaces were still Llama-shaped, so truthful Qwen progress required one narrow vertical slice first | ✓ Good |
| Use the primary GGUF `tokenizer.chat_template` as the maintained formatter source of truth | This kept EMEL, paritychecker, and benchmark publication on one explicit conditioning contract | ✓ Good |
| Insert Phase `26.1` before resuming Qwen bring-up | Native `q8_0` runtime support was a real blocker and had to land in `src/emel` before truthful generator bring-up | ✓ Good |
| Reuse the shipped paritychecker and `tools/bench` surfaces as the only v1.6 acceptance boundary | This kept the milestone brownfield and made parity and benchmark claims share one maintained contract | ✓ Good |
| Preserve the prior Llama anchor while widening to canonical Qwen | The maintained Qwen lane had to expand without silently displacing the prior truthful anchor | ✓ Good |

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
*Last updated: 2026-03-30 after shipping v1.6*
