---
phase: 65
slug: unified-compare-workflow-and-publication
created: 2026-04-17
status: complete
---

# Phase 65 Context

## Phase Boundary

Phase 65 publishes the unified operator workflow and compare artifact surface across backend
languages. It ties together the shared contract, Python backends, C++ backends, tests, and
shell entrypoint into one maintained compare path.

## Implementation Decisions

### Scope
- Add one compare driver that can run EMEL plus a selected reference backend manifest.
- Add one operator shell entrypoint for the common workflow.
- Add focused bench-tool tests for compare summaries and Python backend emission.
- Publish compare artifacts with backend identity, fixture identity, and similarity evidence.

### Constraints
- Keep the workflow truthful about parity vs baseline comparisons.
- Keep artifact publication machine-readable and reproducible.

## Existing Code Insights

- The shared contract and manifests were already enough to run the pieces independently.
- What remained was one driver plus tests/docs so operators did not need to hand-stitch runners,
  JSONL files, and vector directories.
