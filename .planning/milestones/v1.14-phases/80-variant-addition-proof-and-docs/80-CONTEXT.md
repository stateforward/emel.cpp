# Phase 80: Variant Addition Proof And Docs - Context

**Gathered:** 2026-04-21
**Status:** Completed

<domain>
## Phase Boundary

Prove and document the data-only add path for maintained generation and embedding benchmark
variants.

</domain>

<decisions>
## Implementation Decisions

### Proof
- Use focused tests to prove discovery, duplicate rejection, and selected variant filtering.
- Document the files that ordinary additions touch.
- Explicitly identify runner/compare/test enumeration files as not part of ordinary additions.

### the agent's Discretion
Use existing fixtures and payloads rather than adding new runtime behavior.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Existing generation workload manifests and embedding payloads provide enough proof without new
  model support.

### Established Patterns
- README files under benchmark manifest directories document local contracts.

### Integration Points
- `tools/bench/generation_variants/README.md`
- `tools/bench/embedding_variants/README.md`
- `tools/bench/reference_backends/README.md`

</code_context>

<specifics>
## Specific Ideas

The milestone should close with auditable proof that variant additions are local and deterministic.

</specifics>

<deferred>
## Deferred Ideas

Broader plugin packaging remains future work.

</deferred>
