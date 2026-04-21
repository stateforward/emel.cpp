# Phase 79: Embedding Variant Discovery Cutover - Context

**Gathered:** 2026-04-21
**Status:** Completed

<domain>
## Phase Boundary

Move maintained embedding benchmark case identity and metadata into discovered variant manifests.

</domain>

<decisions>
## Implementation Decisions

### Variant Ownership
- Store maintained embedding variant identity in `tools/bench/embedding_variants`.
- Let EMEL embedding benchmark output records consume case name, compare group, modality,
  comparison mode, and note from manifests.
- Let Python golden/live reference backends consume the same variant manifests.

### the agent's Discretion
Support only the existing maintained payload IDs in this milestone.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Existing TE text/image/audio benchmark payload construction remains valid.
- Python golden outputs already map cleanly by payload kind.

### Established Patterns
- The compare driver groups records by `compare_group`.

### Integration Points
- `embedding_generator_bench.cpp`
- `embedding_reference_python.py`
- `embedding_compare.py`

</code_context>

<specifics>
## Specific Ideas

Keep C++ and Python lane metadata aligned through shared variant files.

</specifics>

<deferred>
## Deferred Ideas

New payload implementations and remote reference engines are deferred.

</deferred>
