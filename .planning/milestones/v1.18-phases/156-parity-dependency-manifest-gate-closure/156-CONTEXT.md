# Phase 156: Parity Dependency Manifest Gate Closure - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Close `MANIFEST-01` and `MANIFEST-02` by wiring dependency-manifest emission and conservative
freshness semantics into maintained paritychecker and quality-gate entrypoints.

</domain>

<decisions>
## Implementation Decisions

### Production Manifest Path
- Keep the manifest schema and records source-owned by
  `tools/paritychecker/parity_dependency_manifest.*`.
- Expose manifest emission through the maintained paritychecker CLI so operators and scripts can
  produce `parity_dependency_manifest/v1` without reaching into test helpers.
- Add a checked-in maintained manifest baseline so the quality gate has a production comparison
  target.

### Freshness Gate Path
- Make the quality gate ask the paritychecker CLI whether manifest data is missing, stale, or
  uncertain before deciding whether automatic parity checks can be skipped.
- Treat missing, stale, and uncertain states as conservative escalation signals that force the full
  parity gate in the maintained quality-gate path.
- Keep ordinary parity execution semantics unchanged when the manifest is fresh.

### Tests
- Add CLI coverage for manifest emission and freshness decisions.
- Add source/behavior checks that the quality gate invokes the maintained CLI manifest path and
  handles missing, stale, and uncertain states conservatively.
- Preserve existing parity mode execution tests as the regression proof for unchanged behavior.

</decisions>

<code_context>
## Existing Code Insights

- `dependency_manifest::render()`, `write()`, and `requires_full_gate(...)` already exist but are
  only covered by tests and docs.
- `parity_runner.cpp` owns CLI parsing and validation after Phase 153, making it the right entry
  point for production manifest emission and freshness checks.
- `scripts/quality_gates.sh` currently decides whether to run parity from changed-file inference
  and `EMEL_QUALITY_GATES_PARITY`, with no dependency-manifest freshness override.
- `scripts/paritychecker.sh` already builds the maintained paritychecker binary under
  `build/paritychecker_zig`.

</code_context>

<specifics>
## Specific Ideas

Use an explicit manifest CLI mode rather than coupling manifest output to every parity run.

</specifics>

<deferred>
## Deferred Ideas

None.

</deferred>
