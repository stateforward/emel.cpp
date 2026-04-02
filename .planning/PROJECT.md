# EMEL

## What This Is

EMEL is a deterministic C++ inference engine built around Boost.SML orchestration, with behavior
modeled as explicit actors instead of ad hoc control flow. The latest shipped surface now includes
one truthful maintained Qwen3 end-to-end executable-size publication alongside the existing Llama,
Qwen, parity, benchmark, and generated-doc proof surfaces.

## Core Value

Prove real end-to-end behavior with explicit SML orchestration and parity-oriented verification
before widening API surface or model scope.

## Current State

Shipped version: `v1.8`

Status: `v1.8 Truthful Qwen3 E2E Embedded Size` shipped on `2026-04-02`. There is no active
milestone yet; the planning surface is ready for the next milestone definition.

Release summary: v1.8 closed the earlier quick binary-size experiment into a truthful maintained
publication: one canonical `Qwen3-0.6B-Q8_0.gguf` `hello` -> first-token executable-size
comparison between an EMEL-owned runner and one matched `llama.cpp` reference executable, with
runtime smoke proof, generated README publication, and a passing `8/8` milestone audit.

## Latest Milestone: v1.8 Truthful Qwen3 E2E Embedded Size

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
  result now reports `4,073,016` raw bytes for EMEL versus `3,334,264` for the reference row.
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

### Active

- [ ] Define the next milestone and decide whether to resume the deferred Liquid work immediately.
- [ ] Decide whether executable-size evidence should stay publication-only or grow into a broader
  measurement harness that also covers bundle size or gate policy.
- [ ] Decide whether executable-size optimization belongs in the next milestone now that the
  truthful baseline is published.

### Out of Scope

- Whole-product feature-parity size claims across EMEL and `llama.cpp` without a matched
  executable boundary
- Static-library or shared-library artifact size comparisons as the primary published metric
- Deployable bundle size including model, tokenizer, or ancillary asset payloads until a dedicated
  post-v1.8 milestone explicitly claims that surface
- Additional comparator runtimes or alternate model formats without a new milestone that widens the
  claim deliberately
- Blocking quality-gate enforcement for size regressions before the executable-size surface proves
  stable and trustworthy

## Next Milestone Goals

- Resume the deferred Liquid milestone if it is still the highest-leverage product/runtime goal.
- Revisit deployable bundle-size publication only after executable-only measurement remains stable.
- Decide later whether size regressions belong in a blocking gate once the maintained size surface
  has enough signal.
- Broaden beyond the canonical Qwen3 slice only through an explicit new milestone.

## Context

This remains a brownfield repository with an existing codebase map under `.planning/codebase/`.
The v1.8 work started as quick task `260401-ejm`, but it had to expand into a milestone once the
user required final executables instead of library artifacts and true end-to-end Qwen3 inference
instead of kernel-only slices. The repo also had to eliminate a reference-assisted vocab bootstrap
before the EMEL size claim was honest. That truth boundary is now closed, but the size publication
surface is still intentionally non-blocking and narrowly scoped.

## Constraints

- **Architecture**: Follow `docs/rules/sml.rules.md` and `AGENTS.md` so runtime changes keep the
  RTC actor model, explicit events, and no-queue invariant intact.
- **Truth boundary**: Published executable-size claims must stay on final linked executables for
  the maintained `Qwen3-0.6B-Q8_0.gguf` `hello` -> first-token slice unless a new milestone widens
  that scope explicitly.
- **Native EMEL path**: Maintained EMEL proof surfaces must not bootstrap vocab, tokenizer, or
  generation behavior through `llama.cpp` or `ggml`.
- **Comparator policy**: The published executable-size comparison remains limited to EMEL and one
  matched `llama.cpp` reference executable until a future milestone deliberately broadens it.
- **Publication**: The README embedded-size section must stay generated from
  `snapshots/embedded_size/summary.txt`, not hand-edited.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Replace the quick task with milestone v1.8 | The claim depended on multiple repo surfaces, truthful E2E boundaries, and roadmap-level scope control | ✓ Good |
| Measure final linked executables instead of libraries | Bare-metal relevance comes from shipped binaries, not archive artifacts | ✓ Good |
| Lock v1.8 to the canonical Qwen3 `hello` -> first-token path | The repo already had a maintained Qwen fixture and formatter contract, which kept the claim narrow and auditable | ✓ Good |
| Require EMEL-owned vocab/tokenizer/model loading in the published EMEL probe | A truthful EMEL size claim could not depend on `llama.cpp` bootstrap paths | ✓ Good |
| Keep v1.8 comparator scope to EMEL and one matched `llama.cpp` reference | Narrower scope avoided comparator churn and kept the publication claim auditable | ✓ Good |
| Keep executable-size publication non-blocking through v1.8 closeout | The new signal is still too immature to treat as a hard gate | ✓ Good |
| Defer the earlier Liquid plan until after the size-claim boundary closed | The executable-size milestone had to settle before broadening back into new model-family runtime work | ✓ Good |

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
*Last updated: 2026-04-02 after shipping v1.8*
