# Phase 37: Benchmark And Docs Publication - Context

**Gathered:** 2026-03-31
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 37 publishes one parity-backed benchmark and docs path for the maintained Liquid slice. The
phase covers benchmark case inclusion, stored benchmark evidence, and docs updates that clearly
identify the exact fixture and maintained conditioning contract used.

This phase should publish only what the earlier phases have already proved. It must not widen the
milestone into generic Liquid or sibling-quant benchmark claims.

</domain>

<decisions>
## Implementation Decisions

### Benchmark Scope
- **D-01:** Publish exactly one maintained Liquid benchmark case for
  `LFM2.5-1.2B-Thinking-Q4_K_M.gguf`.
- **D-02:** Keep benchmark publication tied to the same parity-backed fixture and maintained
  conditioning contract from earlier phases.
- **D-03:** Do not add sibling Liquid quant benchmark cases in this milestone.

### Publication Truth
- **D-04:** Benchmark output and stored evidence should name the exact fixture, explicit `lfm2`
  architecture slice, and maintained formatter contract together.
- **D-05:** Documentation should make the maintained boundary obvious: one official Liquid Thinking
  `Q4_K_M` ARM slice, nothing broader.
- **D-06:** Keep benchmark naming readable and auditable alongside the existing maintained Llama and
  Qwen benchmark families.

### Docs Boundary
- **D-07:** Update repo docs only where needed to publish the maintained benchmark truth and fixture
  provenance; do not expand public docs into generic Liquid-family guidance.
- **D-08:** Docs should continue to call out executable metadata truth where it overrides stale
  upstream prose.
- **D-09:** Benchmark publication should not hide any remaining scope limits such as unsupported
  sibling quants, tool use, or history semantics.

### Release Discipline
- **D-10:** Publish benchmark evidence only after the maintained Liquid slice is parity-backed.
- **D-11:** Keep the benchmark case set minimal and representative rather than turning publication
  into a broad matrix exercise.
- **D-12:** Any future benchmark broadening for sibling quants or broader Liquid families should be
  a later explicit milestone.

### the agent's Discretion
- Exact case labels and doc wording can stay local as long as the maintained boundary is explicit
  and consistent across compare output and planning/docs surfaces.
- Benchmark artifact placement may follow existing repo conventions as long as it remains easy for a
  reviewer to trace published claims back to the exact maintained fixture and contract.

</decisions>

<specifics>
## Specific Ideas

- The user wants to go fully autonomous after discussion, so this context deliberately closes off
  benchmark-scope ambiguity before planning starts.
- Phase 37 is publication, not discovery. If a claim is not parity-backed by Phase 36, it does not
  belong in the benchmark/docs surface.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Repo Rules
- `AGENTS.md` — no false benchmark claims and no broader-family implication from one maintained
  slice
- `docs/rules/sml.rules.md` — general engineering and machine-structure rules

### Milestone Scope
- `.planning/REQUIREMENTS.md` — `BENCH-08`
- `.planning/ROADMAP.md` — Phase 37 goal and success criteria
- `.planning/phases/33-fixture-metadata-and-contract-lock/33-CONTEXT.md`
- `.planning/phases/36-parity-and-regression-proof/36-CONTEXT.md`

### Current Code Seams
- `tools/bench/generation_bench.cpp` — maintained generation benchmark setup and per-case output
- `tools/bench/bench_main.cpp` — compare/report publication surface
- `tests/models/README.md` — fixture provenance ledger
- `.planning/research/SUMMARY.md` and `.planning/PROJECT.md` — milestone-facing maintained-scope docs

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/bench/generation_bench.cpp` already publishes formatter-contract strings and can carry the
  maintained Liquid case once runtime and parity are ready.
- `tools/bench/bench_main.cpp` already prints benchmark configuration and reference metadata, which
  are natural places to keep the maintained Liquid slice auditable.

### Established Patterns
- Earlier maintained slices were published as one explicit benchmark family, not broad model-family
  marketing.
- Docs and benchmark output already act as reviewer-facing truth surfaces in this repo.

### Integration Points
- `tools/bench/generation_bench.cpp` defines the maintained generation benchmark case.
- `tools/bench/bench_main.cpp` exposes compare output and shared publication metadata.
- `tests/models/README.md` and milestone docs keep the benchmark claim grounded in exact fixture
  provenance and scope.

### Known Current Concern
- Benchmark publication depends on parity-backed runtime truth first. Any remaining runtime or
  parity instability should block publication rather than being papered over in docs.

</code_context>

<deferred>
## Deferred Ideas

- Additional Liquid benchmark cases for sibling quants.
- Generic Liquid-family benchmark documentation.
- Multi-turn or tool-use benchmark scenarios.

</deferred>

---
*Phase: 37-benchmark-and-docs-publication*
*Context gathered: 2026-03-31*
