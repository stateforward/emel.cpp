# Phase 77: Benchmark Variant Registry Contract - Context

**Gathered:** 2026-04-21
**Status:** Completed

<domain>
## Phase Boundary

Define benchmark-owned manifest discovery and validation primitives shared by generation workload
manifests and embedding variant manifests.

</domain>

<decisions>
## Implementation Decisions

### Contract Shape
- Keep the contract in `tools/bench` because this is benchmark tooling, not runtime behavior.
- Use checked-in JSON manifests as the variant source of truth.
- Discover manifests in deterministic path order.
- Hard-fail duplicate IDs, missing schemas, and unsupported variant fields before publishing
  compare records.

### the agent's Discretion
Use small header-only helpers to avoid adding a new build target.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Existing generation workload manifests already contain the required provenance and comparability
  fields.
- Existing compare wrappers already pass selected workload/case filters through environment
  variables.

### Established Patterns
- `tools/bench` keeps benchmark-only contracts close to the runner and compare scripts.
- Tests use doctest binaries under `tools/bench`.

### Integration Points
- `generation_workload_manifest.hpp`
- New `embedding_variant_manifest.hpp`
- `embedding_compare.py` / `scripts/bench_embedding_compare.sh`

</code_context>

<specifics>
## Specific Ideas

Reduce the developer surface for adding new benchmark variants without widening runtime support.

</specifics>

<deferred>
## Deferred Ideas

Public plugin SDKs, remote reference engines, and new runtime/model support remain out of scope.

</deferred>
