# Phase 34: `lfm2` Model Contract Bring-Up - Context

**Gathered:** 2026-03-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 34 makes EMEL-owned model-loading surfaces truthfully accept the maintained Liquid Thinking
fixture as `lfm2` and expose an explicit maintained topology contract. The phase stays inside
`src/emel` model-contract surfaces plus the repo-facing proof surfaces that currently report model
identity in paritychecker and bench.

This phase is still pre-runtime. It must reject false positives instead of letting the canonical
Liquid slice drift through existing `llama` or `qwen3` assumptions. Architecture detection alone is
not enough; the maintained `Q4_K_M` fixture must have an explicit metadata, tensor, and topology
contract before Phase 35 can claim runtime execution.

</domain>

<decisions>
## Implementation Decisions

### Maintained Scope
- **D-01:** Add an explicit `lfm2` architecture path in EMEL, but keep maintained acceptance scoped
  to `tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf`.
- **D-02:** Defer any broader `lfm2` compatibility beyond that one maintained fixture.
- **D-03:** If the canonical fixture requires tied output or embedding-reuse behavior, treat it as
  part of the maintained model contract.
- **D-04:** Prefer exact canonical-slice truth over preemptive generalized Liquid-family
  abstraction.

### Partial-Model Tolerance
- **D-05:** Reject `lfm2` fixtures that are missing required maintained metadata or tensor names.
- **D-06:** If EMEL can detect `general.architecture=lfm2` but cannot yet prove the full maintained
  topology contract, fail before runtime bring-up.
- **D-07:** Do not alias “close enough” Liquid structure onto existing `llama` or `qwen3` paths.
- **D-08:** Pin required tensor-name and topology mapping to the maintained `Q4_K_M` fixture rather
  than inferring from sibling quants or future models.

### Operator-Visible Model Evidence
- **D-09:** Repo-facing proof surfaces should visibly report the maintained architecture as `lfm2`,
  not only succeed silently.
- **D-10:** Contract failures should stay explicit at the `model_invalid` boundary rather than
  drifting into vague loader/runtime errors.
- **D-11:** Paritychecker and bench should name both the exact maintained fixture and the dedicated
  `lfm2` architecture slice.
- **D-12:** Do not advertise broader “Liquid-compatible” wording anywhere in this phase.

### Future-Facing Compatibility Boundaries
- **D-13:** Keep the `lfm2` bring-up additive to existing maintained Llama and Qwen paths rather
  than refactoring them into a premature shared abstraction.
- **D-14:** Do not widen public API or operator promises to sibling quants, other Liquid families,
  or generic model-family discovery in this phase.
- **D-15:** Any broader multi-quant `lfm2` support remains a later explicit expansion, with
  `Q4_K_M` as the maintained truth anchor for v1.9.

### the agent's Discretion
- Internal helper placement can stay local as long as the authoritative `lfm2` contract lives in
  EMEL-owned code under `src/emel` and not only in tool-local seams.
- Exact names for additive `lfm2` helpers may follow existing repo conventions as long as they do
  not blur the maintained boundary with existing Llama or Qwen contracts.

</decisions>

<specifics>
## Specific Ideas

- The current repo has hard-coded architecture gates that only accept `llama` or `qwen3`; Phase 34
  exists to replace that false binary with one explicit maintained `lfm2` path.
- The user wants autonomous follow-through and has locked remaining phase discussion to the
  recommended defaults.
- The user also wants broader Liquid coverage eventually, but v1.9 remains one maintained
  `Q4_K_M` slice.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Repo Rules
- `AGENTS.md` — explicit maintained-scope truth, no silent fallback, and no performance-contract
  substitution without approval
- `docs/rules/sml.rules.md` — RTC actor-model and explicit control-flow rules

### Milestone Scope
- `.planning/PROJECT.md` — v1.9 scope guardrails after the `Q4_K_M` rescope
- `.planning/REQUIREMENTS.md` — `RUN-03` and `RUN-05`
- `.planning/ROADMAP.md` — Phase 34 goal and success criteria
- `.planning/phases/33-fixture-metadata-and-contract-lock/33-CONTEXT.md` — maintained fixture,
  formatter, and unsupported-request contract

### Current Code Seams
- `src/emel/model/data.cpp` — current quantized-path audit and architecture-dependent contract
  assumptions
- `tools/paritychecker/parity_runner.cpp` — current `general.architecture` gate and maintained
  fixture reporting
- `tools/bench/generation_bench.cpp` — benchmark-side metadata gate and maintained proof output
- `src/emel/generator/detail.hpp` — downstream runtime path still keyed to Qwen-only behavior that
  Phase 35 will need after Phase 34 lands the explicit `lfm2` contract

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/model/data.cpp` already exposes architecture-name and quantized-path audit seams that
  can be widened additively for `lfm2`.
- `tools/paritychecker/parity_runner.cpp` and `tools/bench/generation_bench.cpp` already parse
  `general.architecture`, publish fixture identity, and classify unsupported models as
  `model_invalid`.

### Established Patterns
- Prior maintained-slice work kept architecture truth narrow and explicit instead of claiming model
  family compatibility from partial metadata.
- The repo already treats fixture identity and operator-visible contract strings as auditable truth
  surfaces.

### Integration Points
- `src/emel/model/data.cpp` is the main EMEL-owned model-contract boundary.
- `tools/paritychecker/parity_runner.cpp` and `tools/bench/generation_bench.cpp` must mirror the
  same maintained `lfm2` contract rather than keeping separate permissive logic.

### Known Current Concern
- Paritychecker currently hard-codes the old Qwen maintained fixture and only accepts `llama` /
  `qwen3` at its metadata gate. Phase 34 planning should assume those seams need synchronized
  conversion.

</code_context>

<deferred>
## Deferred Ideas

- Sibling Liquid quants beyond `Q4_K_M`.
- Broad `lfm2` family compatibility.
- Full Liquid-family support beyond the maintained Thinking slice.
- Tool use and multi-turn thinking-history support.

</deferred>

---
*Phase: 34-lfm2-model-contract-bring-up*
*Context gathered: 2026-03-31*
