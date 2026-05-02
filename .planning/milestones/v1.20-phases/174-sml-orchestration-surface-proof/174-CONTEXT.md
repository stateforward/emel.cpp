# Phase 174: SML Orchestration Surface Proof - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 174 closes SRC-03 by adding live proof that the migrated `stateforward::sml` surface covers
dispatch tables, logger policies, unexpected-event handling, and state inspection.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion
- Prefer focused doctest coverage in the existing `tests/sm` shard.
- Keep the proof local to SML surface behavior and avoid changing production actor behavior.
- Preserve destination-first transition-table style in new test models.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/sm/*` already covers project SML wrapper behavior.
- `CMakeLists.txt` owns the `EMEL_TEST_SOURCES` list and `emel_tests_sm` shard.

### Established Patterns
- Doctest is the unit-test framework.
- SML tests use `stateforward::sml::state<...>` and `machine.is(...)` for state inspection.

### Integration Points
- New SML surface proof belongs in the existing `tests/sm` shard.
- The focused verification command is `EMEL_ZIG_TEST_SHARDS=sm scripts/build_with_zig.sh` followed
  by `ctest --test-dir build/zig -R '^emel_tests_sm$' --output-on-failure`.

</code_context>

<specifics>
## Specific Ideas

No user-facing behavior. This is a focused infrastructure proof phase.

</specifics>

<deferred>
## Deferred Ideas

Legacy SML drift guardrails and full quality-gate repair are deferred to Phase 176.

</deferred>
