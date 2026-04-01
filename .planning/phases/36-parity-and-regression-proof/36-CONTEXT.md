# Phase 36: Parity And Regression Proof - Context

**Gathered:** 2026-03-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 36 proves the exact maintained Liquid slice against `llama.cpp` and adds regression evidence
without breaking the prior maintained anchors. The phase covers paritychecker generation proof,
stored regression artifacts, and operator-visible parity evidence for the exact fixture and
conditioning contract already locked in earlier phases.

This phase should prove the maintained slice, not broaden it. Parity evidence must stay tied to the
exact `Q4_K_M` fixture and maintained Liquid contract.

</domain>

<decisions>
## Implementation Decisions

### Parity Scope
- **D-01:** Parity proof is for the exact maintained fixture
  `tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf` only.
- **D-02:** Use the same maintained conditioning contract from Phase 33 and the explicit `lfm2`
  model/runtime contract from Phases 34 and 35.
- **D-03:** Do not imply sibling-quant or generic Liquid-family parity from this proof.

### Proof Corpus
- **D-04:** Expand maintained proof just enough to stop being artificially narrower than the locked
  contract: keep the corpus small and deterministic, but include both the baseline single-user path
  and at least one maintained `system + user` case.
- **D-05:** Keep prior `assistant` history, tool use, named-template variants, and
  `keep_past_thinking` semantics out of parity success cases; they should remain explicit rejection
  cases until a later scope expansion.
- **D-06:** Preserve deterministic, bounded proof inputs and outputs so artifacts stay reviewable.

### Regression Protection
- **D-07:** Stored regression evidence must cover the maintained Liquid slice and the prior
  maintained Llama and Qwen anchors.
- **D-08:** Artifact headers or equivalent stored proof metadata should make fixture identity,
  architecture, and maintained formatter contract auditable.
- **D-09:** Any artifact-format expansion deferred by Phase 33 can happen here, but only additively
  and only to support auditable maintained parity truth.

### Operator-Visible Evidence
- **D-10:** Paritychecker output should name the exact maintained fixture, explicit `lfm2`
  architecture slice, and maintained contract together.
- **D-11:** Reference attribution should stay explicit and reviewer-auditable.
- **D-12:** Failure output should stay narrow and truthful rather than suggesting generic Liquid
  parity readiness.

### the agent's Discretion
- Exact corpus prompt text can stay local as long as it is deterministic, minimal, and clearly tied
  to the maintained request contract.
- Artifact-field naming can evolve additively as long as older maintained flows are not broken
  without reason.

</decisions>

<specifics>
## Specific Ideas

- The user previously called out that proof coverage should eventually catch up to the broader
  maintained `system + user` contract. Phase 36 is the right place for that limited widening.
- This phase is where Liquid stops being merely “runnable” and becomes regression-protected.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Repo Rules
- `AGENTS.md` — maintained truth, no false parity claims, and regression evidence expectations
- `docs/rules/sml.rules.md` — explicit machine/rule guidance

### Milestone Scope
- `.planning/REQUIREMENTS.md` — `PAR-02` and `VER-02`
- `.planning/ROADMAP.md` — Phase 36 goal and success criteria
- `.planning/phases/33-fixture-metadata-and-contract-lock/33-CONTEXT.md`
- `.planning/phases/34-lfm2-model-contract-bring-up/34-CONTEXT.md`
- `.planning/phases/35-maintained-runtime-execution-on-arm/35-CONTEXT.md`

### Current Code Seams
- `tools/paritychecker/parity_runner.cpp` — maintained generation proof path, fixture enforcement,
  baseline I/O, and operator-visible parity reporting
- `tools/paritychecker/paritychecker_tests.cpp` — existing maintained proof/test patterns
- `tools/generation_formatter_contract.hpp` — maintained request formatting and contract binding
- `tests/models/README.md` — provenance truth for the maintained fixture

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/paritychecker/parity_runner.cpp` already carries stored baseline machinery, fixture
  identity fields, formatter-contract reporting, and explicit reference metadata surfaces.
- `tools/paritychecker/paritychecker_tests.cpp` already exercises maintained formatter and parity
  seams that can be extended for the Liquid slice.

### Established Patterns
- Earlier maintained phases used one exact fixture plus stored artifact truth rather than loose
  “compatible model” parity.
- The repo already expects maintained regression work to protect previous anchors, not only the new
  slice.

### Integration Points
- `tools/paritychecker/parity_runner.cpp` is the main proof publication seam.
- `tools/paritychecker/paritychecker_tests.cpp` should carry the regression intent at test level.
- `tools/generation_formatter_contract.hpp` remains the maintained request-shape truth source for
  parity inputs.

### Known Current Concern
- The existing branch has paritychecker build debt against upstream reference changes. Phase 36
  planning should preserve time to clear that friction before claiming green maintained proof.

</code_context>

<deferred>
## Deferred Ideas

- Multi-turn/history parity.
- Tool-use parity.
- Sibling-quant parity matrices.
- Broad Liquid-family regression suites.

</deferred>

---
*Phase: 36-parity-and-regression-proof*
*Context gathered: 2026-03-31*
