# Phase 70: Reproducible Generation Workload Contract - Context

**Gathered:** 2026-04-20
**Status:** Completed

<domain>
## Phase Boundary

Phase 70 pins maintained generation compare workloads to explicit checked-in prompt and workload
manifests so the current EMEL and reference generation lanes stop relying on hardcoded prompt,
sampling, seed, and formatter assumptions inside `generation_bench.cpp`.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- Keep the scope in `tools/bench`, scripts, tests, and documentation.
- Reuse the existing maintained generation bench surface instead of creating a second runner.
- Make prompt identity and workload identity explicit repo-owned files, not shell-only or
  hardcoded local constants.
- Mark non-comparable single-lane workloads explicitly instead of letting them look like parity
  records.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/bench/reference_backends/*.json` and `tools/bench/reference_backends/README.md` already
  establish the checked-in manifest pattern for pluggable compare surfaces.
- `tools/bench/embedding_compare.py` shows the current repo pattern for loading manifests and
  preserving machine-readable provenance in compare outputs.
- `tools/generation_fixture_registry.hpp` and `tools/generation_formatter_contract.hpp` already
  pin the maintained generation model fixtures and supported formatter truth.

### Current Gaps
- `tools/bench/generation_bench.cpp` still hardcodes prompt text, token budgets, sampling id,
  stop id, and compare metadata in C++ constants.
- `generation_compare/v1` records currently expose lane metadata, formatter contract, and prompt
  id, but they do not yet point back to explicit workload-manifest or prompt-fixture files.
- The EMEL-only Gemma4 generation path is additive today but not explicitly labeled
  non-comparable in the machine-readable output.

</code_context>

<specifics>
## Specific Ideas

- Add checked-in `tools/bench/generation_prompts/*.json` fixtures and
  `tools/bench/generation_workloads/*.json` manifests.
- Load maintained generation workload metadata from those manifests before bench dispatch begins.
- Extend `generation_compare/v1` records with workload-manifest and prompt-fixture provenance plus
  explicit comparability markers.
- Keep compare mode restricted to comparable workloads while allowing EMEL-only workloads to emit
  truthful `single_lane` records in JSONL mode.

</specifics>

<deferred>
## Deferred Ideas

- External or Python-backed reference backend adapters remain Phase 71 scope.
- Unified compare wrapper/publication work remains Phase 72 scope.
- Milestone closeout traceability remains Phase 73 scope.

</deferred>
