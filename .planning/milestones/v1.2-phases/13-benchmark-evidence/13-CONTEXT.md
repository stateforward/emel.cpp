# Phase 13: Benchmark Evidence - Context

**Gathered:** 2026-03-22
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 13 publishes truthful flash-attention benchmark evidence for the canonical CPU-hosted
Llama-68M generation slice through the existing `tools/bench` compare workflow, maintained
snapshot artifacts, and generated benchmark docs. This phase stays inside the current benchmark
surface: no new runtime/API surface, no flash-only benchmark command, and no broadening beyond the
accepted canonical workload.

</domain>

<decisions>
## Implementation Decisions

### Benchmark Surface
- **D-01:** Phase 13 keeps `scripts/bench.sh --compare` and `scripts/bench.sh --compare-update`
  as the only publication path for benchmark evidence.
- **D-02:** The maintained BENCH-03 gate is
  `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`.
- **D-03:** The canonical compare row name stays unchanged so the existing snapshot/docs pipeline
  remains compatible.
- **D-04:** Durable flash proof must live in the normal compare output itself as comment metadata,
  not only in stderr, a sidecar artifact, or docs-only prose.

### Reference And Truth Policy
- **D-05:** `tools/bench` must use the configured fetched-or-pinned reference implementation and
  must not prefer a machine-local `tmp/llama.cpp` checkout.
- **D-06:** The normal compare output must publish the resolved reference identity as a concrete
  commit SHA so benchmark evidence is reproducible.
- **D-07:** Phase 13 compare execution fails if the canonical short EMEL case does not prove flash
  execution.
- **D-08:** Required proof for the canonical short case is positive flash-dispatch count together
  with zero EMEL decode/logits seam calls and zero reference decode/logits seam calls.

### Baseline Artifact And Improvement Proof
- **D-09:** The pre-flash baseline is preserved as a maintained artifact at
  `snapshots/bench/generation_pre_flash_baseline.txt`.
- **D-10:** That preserved baseline is seeded from committed pre-flash snapshot history
  (`git show 2acd4fe^:snapshots/bench/benchmarks_compare.txt`), not by reintroducing a non-flash
  runtime fallback.
- **D-11:** BENCH-03 is evaluated by comparing current EMEL short-case latency against the
  preserved pre-flash EMEL baseline; reference ns/op and ratio remain preserved as supporting
  context.
- **D-12:** If the current canonical short case is not faster than the preserved baseline, Phase 13
  is blocked rather than published as a warning-only result.

### Publication Gate And Docs Shape
- **D-13:** Checked-in snapshot or generated benchmark artifact changes require explicit user
  approval before execution proceeds.
- **D-14:** After approval, `snapshots/bench/benchmarks_compare.txt` is refreshed only if the
  checked-in snapshot is missing the new proof metadata; otherwise preserve the current compare
  numbers.
- **D-15:** Generated benchmark docs keep the full compare table and add a dedicated flash-evidence
  plus pre-flash-baseline section for the canonical short case.
- **D-16:** Benchmark docs continue to be produced through `tools/docsgen` /
  `scripts/generate_docs.sh`, not by hand editing `docs/benchmarks.md`.

### the agent's Discretion
- Exact proof-comment field names and formatting, as long as they stay machine-checkable and live
  on the normal compare output.
- Exact implementation shape for the baseline comparator CLI and supporting testdata.
- Exact generated-doc wording and section layout, as long as it preserves the full compare table
  and clearly publishes the canonical flash-vs-baseline evidence.

</decisions>

<specifics>
## Specific Ideas

- Mirror the deterministic reference-selection policy already proven in
  `tools/paritychecker/CMakeLists.txt`.
- Use the generator flash counters and seam-audit facts already surfaced in
  `tools/bench/generation_bench.cpp` and `src/emel/generator/sm.hpp` rather than inventing a new
  benchmark-only proof seam.
- Preserve the current compare row format for downstream tooling and add truth metadata as comment
  lines ahead of the rows.
- Keep the canonical short-case improvement workflow executable from repo-local files instead of
  manual `git show` arithmetic.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Milestone Scope And Rules
- `.planning/ROADMAP.md` — Phase 13 goal, success criteria, plan inventory, and milestone scope.
- `.planning/REQUIREMENTS.md` — `BENCH-01`, `BENCH-02`, and `BENCH-03`.
- `.planning/STATE.md` — current milestone position, blocker notes, and recent decisions.
- `AGENTS.md` — project engineering contract, including the explicit snapshot-consent rule.
- `docs/rules/sml.rules.md` — architecture constraints that still bound any runtime-adjacent
  changes.

### Prior Phase Context
- `.planning/phases/12-parity-and-verification-closure/12-CONTEXT.md` — deterministic reference
  sourcing and truth-proof expectations already locked in Phase 12.
- `.planning/phases/12.1-enforce-sml-tensor-lifecycle-orchestration/12.1-CONTEXT.md` — current
  runtime truth boundary that benchmark evidence must not bypass.
- `.planning/phases/13-benchmark-evidence/13-RESEARCH.md` — researched benchmark risks,
  recommended patterns, and historical baseline evidence.

### Benchmark And Docs Surfaces
- `tools/bench/CMakeLists.txt` — benchmark reference-selection policy and compile-time metadata.
- `tools/bench/bench_main.cpp` — compare output shape and comment metadata integration point.
- `tools/bench/generation_bench.cpp` — canonical generation benchmark case and flash/seam proof
  data source.
- `tools/bench/bench_cases.hpp` — maintained canonical generation case names.
- `scripts/bench.sh` — operator-facing compare and compare-update workflow.
- `docs/benchmarking.md` — benchmark runbook and publication policy.
- `snapshots/bench/benchmarks_compare.txt` — maintained compare snapshot consumed by docsgen.
- `tools/docsgen/docsgen.cpp` — generated benchmark-doc pipeline.
- `docs/templates/benchmarks.md.j2` — benchmark-doc rendering template.
- `docs/benchmarks.md` — generated published benchmark evidence surface.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/bench/generation_bench.cpp` already exposes the canonical short and long generation cases
  plus seam-audit helpers and generator-backed proof counters.
- `tools/paritychecker/CMakeLists.txt` already demonstrates the milestone-approved deterministic
  fetched-reference policy that `tools/bench` should match.
- `scripts/bench.sh` already owns the maintained `--compare` / `--compare-update` operator flow.
- `tools/docsgen/docsgen.cpp` and `docs/templates/benchmarks.md.j2` already generate
  `docs/benchmarks.md` from maintained snapshot artifacts.

### Established Patterns
- The repo already treats `snapshots/bench/benchmarks_compare.txt` as the durable publication
  source for benchmark docs.
- Prior phases already established that flash claims must stay truthful and must not rely on a
  hidden fallback or machine-local reference behavior.
- The maintained canonical workload contract is fixed to the checked-in Llama-68M fixture with the
  existing `generation/preloaded_request/...` row names.

### Integration Points
- `tools/bench/CMakeLists.txt` is the integration seam for deterministic reference sourcing and
  compile-time reference metadata.
- `tools/bench/bench_main.cpp` is the integration seam for durable proof comments on the normal
  compare output.
- `tools/bench/generation_bench.cpp` is the integration seam for canonical short-case flash proof.
- `scripts/bench.sh`, `snapshots/bench/benchmarks_compare.txt`, `tools/docsgen/docsgen.cpp`, and
  `docs/templates/benchmarks.md.j2` are the integration chain for maintained publication.

</code_context>

<deferred>
## Deferred Ideas

- A preserved long-case pre-flash baseline for `max_tokens=8` remains out of scope until the repo
  has a trustworthy maintained historical artifact for that case.
- Broader flash benchmark matrices, additional model fixtures, or backend-specific benchmark
  rollouts remain future work beyond this canonical Phase 13 scope.
- Repository-wide benchmark-policy hardening for noisy snapshot smoke checks remains a separate
  concern outside this milestone.

</deferred>

---
*Phase: 13-benchmark-evidence*
*Context gathered: 2026-03-22*
