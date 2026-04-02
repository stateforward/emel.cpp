# Phase 35: Maintained Runtime Execution On ARM - Context

**Gathered:** 2026-03-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 35 brings the shipped EMEL generator path up on the maintained
`LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture on ARM. The phase starts only after Phase 34 has made
the `lfm2` model contract explicit. It covers native runtime initialization, bounded generation,
and truthful operator-visible runtime evidence for the maintained slice.

This phase must stay native and truthful. It cannot satisfy the milestone by using tool-only
runtime shims, broad family claims, or whole-path dequantize-to-f32 fallbacks in the hot path.

</domain>

<decisions>
## Implementation Decisions

### Runtime Scope
- **D-01:** Phase 35 should make the shipped EMEL generator initialize and generate on the exact
  `tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture only.
- **D-02:** Keep runtime claims scoped to the maintained `Q4_K_M` slice; do not imply sibling-quant
  readiness.
- **D-03:** Reuse the Phase 33 maintained conditioning contract and Phase 34 maintained `lfm2`
  model contract without broadening request semantics.

### Native Runtime Truth
- **D-04:** The maintained Liquid path must execute through EMEL-owned code in `src/emel`.
- **D-05:** Do not substitute a tool-local compute scaffold or whole-tensor dequantize-to-f32 hot
  path as the milestone runtime.
- **D-06:** If `Q4_K_M` requires additional native quantized-path work, that work is part of this
  phase rather than a reason to silently narrow the claim back to another quant.
- **D-07:** Preserve explicit `lfm2` runtime selection instead of aliasing the maintained path to a
  generic existing backend.

### Failure Behavior And Visibility
- **D-08:** Unsupported or incomplete maintained-runtime cases should fail explicitly and keep
  naming the exact fixture, architecture, and formatter contract on operator-visible surfaces.
- **D-09:** Do not fall back to broader Liquid-family or sibling-quant acceptance if the maintained
  `Q4_K_M` path is not ready.
- **D-10:** Runtime evidence should publish a truthful maintained quantized-path contract for the
  official `Q4_K_M` fixture only.

### ARM Verification Boundary
- **D-11:** “Done” for this phase means bounded generator initialization and token production on ARM,
  not merely metadata acceptance or partial host-only setup.
- **D-12:** Keep verification focused on one maintained ARM slice rather than turning the phase into
  a quant matrix or multi-platform matrix.
- **D-13:** Do not publish benchmark-grade performance claims in this phase; keep it at runtime
  truth and bounded execution.

### the agent's Discretion
- The exact split between model/runtime helper changes can stay local as long as the end state is
  an EMEL-owned `lfm2` runtime path with no misleading fallback.
- Internal additive runtime helpers may follow existing generator/detail conventions as long as they
  keep the maintained Liquid path explicit.

</decisions>

<specifics>
## Specific Ideas

- The maintained fixture changed from `Q8_0` to `Q4_K_M`, so new quant-runtime work is now inside
  the milestone rather than deferred by default.
- Existing generator code has Qwen-specific runtime branches today; Phase 35 is where Liquid
  runtime-specific handling becomes real if the maintained slice needs it.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Repo Rules
- `AGENTS.md` — no silent performance-contract substitutions, no dequantize-to-f32 hot-path
  fallback without explicit approval
- `docs/rules/sml.rules.md` — explicit control flow and runtime structure rules

### Milestone Scope
- `.planning/REQUIREMENTS.md` — `RUN-04` and `RUN-06`
- `.planning/ROADMAP.md` — Phase 35 goal and success criteria
- `.planning/phases/33-fixture-metadata-and-contract-lock/33-CONTEXT.md`
- `.planning/phases/34-lfm2-model-contract-bring-up/34-CONTEXT.md`

### Current Code Seams
- `src/emel/generator/detail.hpp` — current runtime branching and backend setup, including
  Qwen-specific special cases
- `src/emel/model/data.cpp` — quantized-path audit surface the runtime evidence will rely on
- `tools/paritychecker/parity_runner.cpp` — maintained generation entrypoint and runtime error
  publication
- `tools/bench/generation_bench.cpp` — maintained benchmark setup path that must remain aligned with
  the same runtime truth later

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/generator/detail.hpp` already centralizes runtime setup and architecture-dependent
  behavior, so the maintained `lfm2` path can stay additive instead of scattering new special cases.
- `tools/paritychecker/parity_runner.cpp` already emits detailed setup and initialize failures that
  can carry the maintained Liquid truth surfaces.

### Established Patterns
- Prior maintained runtime phases proved one exact fixture first, then widened parity and benchmark
  evidence later.
- The repo treats native quantized-path ownership as part of “done” for maintained inference work.

### Integration Points
- `src/emel/generator/detail.hpp` is the main native runtime seam.
- `tools/paritychecker/parity_runner.cpp` is the first maintained runtime caller.
- `tools/bench/generation_bench.cpp` should stay aligned with the same maintained runtime contract,
  even if Phase 37 is where benchmark publication becomes official.

### Known Current Concern
- The current branch still has pre-existing paritychecker build debt against upstream `llama.cpp`
  layer-field changes. That should not change Phase 35 scope, but planners should expect local
  verification friction until that debt is cleared.

</code_context>

<deferred>
## Deferred Ideas

- Sibling Liquid quants and broader multi-quant runtime claims.
- Non-ARM performance work beyond what is needed to truthfully execute the maintained slice.
- Tool use, history replay, and broader conversation support.

</deferred>

---
*Phase: 35-maintained-runtime-execution-on-arm*
*Context gathered: 2026-03-31*
