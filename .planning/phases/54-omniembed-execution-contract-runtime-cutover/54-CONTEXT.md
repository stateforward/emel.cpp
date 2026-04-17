---
phase: 54
slug: omniembed-execution-contract-runtime-cutover
created: 2026-04-14
status: ready
---

# Phase 54 Context

## Phase Boundary

Phase 54 closes the audit gap between the Phase 48 `omniembed` execution contract and the live
embedding runtime. The shipped TE slice already works end to end, but the generator still binds its
text, image, and audio runtimes directly from raw model metadata and tensor names instead of
consuming the explicit contract seam introduced in Phase 48. This phase keeps scope tight: cut the
runtime over to that contract, reject contract drift before requests execute, and preserve the
existing text/image/audio behavior.

## Implementation Decisions

### Runtime Scope
- Stay inside `src/emel/embeddings/generator` and `src/emel/model/omniembed`.
- Do not change the generator state-machine structure unless the existing graph makes the fix
  impossible.
- Treat this as a runtime-binding fix, not a broader architecture redesign.

### Contract Enforcement
- Require the embedding generator to build and retain a validated `omniembed` execution contract
  during setup.
- Fail runtime reservation when the maintained multimodal contract is incomplete or inconsistent.
- Make text/image/audio runtime binding consume the validated contract as the source of truth for
  shared embedding dimensions and required modality families.

### Proof Shape
- Start with a failing regression test that demonstrates the current gap.
- Keep the maintained text/image/audio proof surface green after the cutover.
- Preserve the current narrow milestone claim: one maintained TE-75M slice with all three
  modalities present.

## Existing Code Insights

### Relevant Surfaces
- `src/emel/model/omniembed/detail.hpp` already defines the `execution_contract` and builder.
- `src/emel/embeddings/generator/detail.hpp` currently hardcodes runtime binding for text, image,
  and audio.
- `src/emel/embeddings/generator/sm.hpp` reserves runtime scratch in the constructor before
  `initialize(...)` dispatch.

### Constraints
- Dispatch-time allocation is forbidden; any contract construction must remain outside
  `process_event(...)`.
- The runtime must stay synchronous and bounded.
- AGENTS requires a failing unit test before fixing a reported bug.

## Specific Ideas

- Add one regression test proving initialization rejects a TE fixture whose multimodal contract has
  been intentionally broken.
- Store the validated `execution_contract` in generator-owned context.
- Use the stored contract during runtime reservation and modality runtime binding so the generator
  no longer bypasses the explicit Phase 48 seam.

## Deferred Ideas

- Expanding the `execution_contract` to encode every internal tensor needed by modality runtimes
- Public embedding API work
- Additional TE quant slices or optional modality subsets

---
*Phase: 54-omniembed-execution-contract-runtime-cutover*
*Context gathered: 2026-04-14*
