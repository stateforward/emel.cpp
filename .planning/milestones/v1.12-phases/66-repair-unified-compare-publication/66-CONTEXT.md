# Phase 66: Repair Unified Compare Publication - Context

**Gathered:** 2026-04-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Repair the maintained compare publication path in `tools/bench/embedding_compare.py` so
machine-readable summaries preserve every emitted reference result for the maintained C++ compare
workflow.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- This is infrastructure/tooling work, so implementation details are at the agent's discretion as
  long as the resulting summary remains truthful and preserves all emitted maintained compare
  records.
- Keep the fix inside the compare tooling and tests; do not widen `src/` runtime scope to satisfy
  the publication repair.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/bench/embedding_compare.py` already owns summary generation and stdout rendering.
- `tools/bench/embedding_compare_tests.cpp` already exercises parity and baseline summary cases via
  temp JSONL fixtures.

### Established Patterns
- Bench-tool regression coverage lives in focused doctest cases under `tools/bench`.
- Maintained compare artifacts are published under `build/embedding_compare/` and summarized into
  `compare_summary.json`.

### Integration Points
- The `liquid_cpp` backend manifest and wrapper emit multiple baseline reference records that feed
  the compare driver.
- Phase verification should keep using the maintained compare commands already published in Phase
  `65`.

</code_context>

<specifics>
## Specific Ideas

- Reproduce the bug with a failing focused test before changing `embedding_compare.py`.
- Preserve summary truth for repeated `compare_group` values without hiding one backend behind
  another.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>
