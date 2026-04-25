# Phase 82: Diarization Request And Audio Frontend Contract - Context

**Gathered:** 2026-04-22
**Status:** Ready for planning
**Mode:** Autonomous smart discuss defaults

<domain>
## Phase Boundary

Phase 82 establishes the diarization-owned request and feature-extractor contract for the maintained
Sortformer profile. It accepts only caller-owned mono `float32` PCM at 16,000 Hz, validates output
capacity for the maintained chunk/profile shape, and writes a deterministic caller-owned feature
feature buffer. It does not run Sortformer tensors, decode speaker probabilities, publish segments,
or add parity/benchmark surfaces.

</domain>

<decisions>
## Implementation Decisions

### Request Surface
- Define a new `emel::diarization::request` component instead of adding Sortformer behavior to
  generic model data or embedding generator APIs.
- Keep the first request surface synchronous and in-memory: `std::span<const float>` PCM plus
  caller-owned `std::span<float>` feature output.
- Require explicit `sample_rate`, `channel_count`, and output-capacity fields so invalid media or
  buffer shape fails before runtime work.
- Keep public request events small and immutable except for required output references and optional
  same-RTC callbacks.

### Frontend Contract
- Use the Phase 81 Sortformer execution contract as the profile source: 16 kHz, 4 speakers,
  188 maintained frames, 80 ms frame shift, and 1 right-context frame.
- Derive a deterministic feature/input matrix natively in EMEL-owned code without file decode,
  resampling, channel mixing, NeMo, ONNX, Python, llama.cpp, or ggml.
- Treat this phase's feature extractor as a contract-proof input preparation path; full Sortformer tensor
  execution remains Phase 83.
- Require the caller-owned feature buffer to hold the maintained `frame_count * feature_bins`
  elements and return produced frame/bin counts.

### Error Routing
- Model sample-rate, channel-count, empty-payload, shape, and output-capacity rejection with
  explicit Boost.SML guards and error states.
- Keep request validation decisions in `guards.hpp`; actions only mark already-chosen outcomes or
  run bounded feature-extractor numeric work.
- Use one empty persistent context for this contract actor; dispatch-local data travels through
  typed runtime events only.
- Keep unexpected events explicit through `sml::unexpected_event`.

### the agent's Discretion
- Choose the smallest deterministic feature-bin count that preserves a useful feature-extractor contract for
  Phase 83 planning without pretending to be full Sortformer execution.
- Add focused doctest coverage under `tests/diarization/request/` and wire it into a dedicated
  shard.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/model/sortformer/detail.hpp` exposes the maintained Sortformer execution contract from
  Phase 81.
- `src/emel/embeddings/generator` shows existing in-memory audio request patterns and validation
  for 16 kHz `float32` PCM.
- `src/emel/gbnf/sampler` is a compact SML reference for destination-first transition tables,
  runtime events, explicit unexpected-event handling, and wrapper-owned `process_event`.

### Established Patterns
- Component directories map to namespaces and use `events.hpp`, `errors.hpp`, `guards.hpp`,
  `actions.hpp`, `context.hpp`, `sm.hpp`, and optional `detail.hpp`.
- Runtime behavior choice belongs in `guards.hpp` and transition rows, not action/detail helpers.
- Tests mirror subsystem structure and are added to `EMEL_TEST_SOURCES` plus a ctest shard.

### Integration Points
- Add the new component under `src/emel/diarization/request/`.
- Add tests under `tests/diarization/request/lifecycle_tests.cpp`.
- Add `emel::DiarizationRequest` as an additive top-level C++ alias in `src/emel/machines.hpp`.

</code_context>

<specifics>
## Specific Ideas

No additional user-specific API naming was supplied for Phase 82. The implementation should stay
small, deterministic, and isolated so later runtime phases can build on it without touching generic
model files.

</specifics>

<deferred>
## Deferred Ideas

- Sortformer encoder/cache/transformer execution remains Phase 83.
- Speaker activity probabilities and diarization segment records remain Phase 84.
- Reference parity, benchmark publication, and operator documentation remain Phase 85.

</deferred>
