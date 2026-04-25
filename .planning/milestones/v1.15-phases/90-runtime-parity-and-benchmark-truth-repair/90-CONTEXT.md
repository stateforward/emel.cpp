# Phase 90: Runtime Parity And Benchmark Truth Repair - Context

**Gathered:** 2026-04-23
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase replaces fabricated Sortformer parity and benchmark evidence with maintained EMEL
runtime output from the Phase 89 pipeline. It must not widen the supported model/profile, update
benchmark snapshots without approval, or claim real-audio external-engine parity beyond the
deterministic maintained proof fixture.

</domain>

<decisions>
## Implementation Decisions

### Parity Truth
- The parity test must call `emel::diarization::sortformer::pipeline::sm`, not
  `fill_reference_probabilities` plus `decode_segments`.
- The reference baseline may remain a fixed canonical segment baseline if it is explicitly the
  expected output for the deterministic maintained fixture and is compared against EMEL runtime
  output.
- The test must preserve fixture/profile provenance in its assertions and failure context.
- The parity test should verify probability/segment determinism from the runtime lane, not helper
  determinism alone.

### Benchmark Truth
- The canonical `diarization_sortformer` EMEL lane must measure `SortformerPipeline::process_event`
  using caller-owned output buffers.
- The EMEL lane must not fill fixed probabilities or run output decoding as the only measured work.
- Reference lane state must remain separate and must not share EMEL-owned model/runtime/cache
  objects.
- Existing unsnapshotted benchmark rows remain unsnapshotted until explicit user approval.

### Documentation
- Benchmark docs should state that the maintained EMEL lane now runs the Phase 89 pipeline.
- Docs must still avoid claiming broad real-audio GGUF parity or external-engine performance
  equivalence.
- Keep limitations explicit: deterministic maintained fixture, fixed profile, snapshot approval
  still pending.
- Do not repair roadmap/state validation ledgers in this phase; Phase 92 owns those artifacts.

### the agent's Discretion
The exact location for deterministic fixture builders is at the agent's discretion, as long as
they remain lane-owned test/benchmark setup and do not become production compute fallbacks.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 89 added `src/emel/diarization/sortformer/pipeline/sm.hpp`.
- Current parity test fills fixed probabilities in
  `tests/diarization/sortformer/parity/lifecycle_tests.cpp`.
- Current benchmark EMEL lane fills fixed probabilities in
  `tools/bench/diarization/sortformer_bench.cpp`.

### Established Patterns
- Benchmark lanes are appended through `append_emel_sortformer_diarization_cases` and
  `append_reference_sortformer_diarization_cases`.
- Bench metadata uses `lane`, `compare_group`, `model_id`, `fixture_id`, `workload_id`,
  `output_dim`, `output_checksum`, and `note`.
- Doctest parity tests use local deterministic fixtures and exact segment assertions.

### Integration Points
- Parity and benchmark paths should call the new pipeline actor through `event::run`.
- `docs/templates/benchmarks.md.j2` is the maintained source for generated benchmark docs.

</code_context>

<specifics>
## Specific Ideas

- Use a deterministic zero-weight maintained tensor fixture for parity/benchmark runtime proof so
  expected speaker segments are stable and cheap to reason about.
- Keep the reference baseline as four full-length speaker segments if the maintained fixture emits
  threshold-active `0.5` probabilities for all speakers.

</specifics>

<deferred>
## Deferred Ideas

- Phase 91 handles SML action-branching governance findings.
- Phase 92 backfills validation/frontmatter/ledger evidence and can consolidate repeated fixture
  setup if needed.

</deferred>
