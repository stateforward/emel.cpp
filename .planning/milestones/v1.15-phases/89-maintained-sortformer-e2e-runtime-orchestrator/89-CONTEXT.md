# Phase 89: Maintained Sortformer E2E Runtime Orchestrator - Context

**Gathered:** 2026-04-23
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase closes the runtime integration gap only. It must add one maintained EMEL-owned path
from validated mono 16 kHz PCM through request feature extraction, native encoder-frame
derivation, Sortformer executor hidden output, speaker-probability computation, and bounded
segment decoding. It must not repair parity/benchmark publication, SML action branching, or
milestone ledger artifacts; those are assigned to phases 90 through 92.

</domain>

<decisions>
## Implementation Decisions

### Orchestration Surface
- Add a component-local Boost.SML actor for the maintained Sortformer E2E path instead of another
  ad hoc test sequence.
- Own fixed-size scratch buffers in actor context so dispatch remains allocation-free after
  construction.
- Compose existing child actors through `process_event(...)` calls only; do not call child guards,
  actions, or implementation functions directly except for already-owned numeric kernels where no
  actor exists.
- Keep the public event small and reference-based, with caller-owned probability and segment
  outputs.

### Runtime Evidence
- Tests should prove real phase wiring: request actor output feeds encoder-frame computation,
  executor output feeds `compute_speaker_probabilities`, and segment decoding consumes those
  probabilities.
- Valid tests may use a deterministic in-memory model fixture, but not fixed output
  probabilities, synthetic feature projection, external runtime state, or reference-lane state.
- Invalid tests should cover input/capacity errors through the new actor rather than helper-only
  checks.
- Repeated dispatch on the same actor and fixture should produce byte-stable records.

### Scope Control
- Do not update benchmark or parity lanes in this phase.
- Do not update benchmark snapshots.
- Do not claim real-audio GGUF benchmark parity from this phase.
- Defer action-branching cleanup in existing request/executor actors to Phase 91.

### the agent's Discretion
The exact component name and event name are at the agent's discretion, provided the directory maps
to namespaces and files use the canonical component bases.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `emel::diarization::request::sm` validates the maintained request profile and writes
  deterministic `188 x 128` features.
- `encoder::detail::compute_encoder_frames_from_features` derives `188 x 512` encoder frames from
  features and a bound encoder contract.
- `emel::diarization::sortformer::executor::sm` consumes `188 x 512` encoder frames and emits
  deterministic `188 x 192` hidden frames.
- `output::detail::compute_speaker_probabilities` and `decode_segments` publish the maintained
  `188 x 4` probability matrix and bounded segments.

### Established Patterns
- Diarization request and executor actors use `event::*` trigger events, `events::*_done/error`
  outcome events, context aggregates, destination-first transition tables, and top-level aliases in
  `src/emel/machines.hpp`.
- Tests use deterministic in-memory tensor fixtures and doctest shards under
  `tests/diarization/...`.
- CMake currently lists diarization test sources explicitly.

### Integration Points
- New source files should live under `src/emel/diarization/sortformer/<component>/`.
- The maintained top-level alias should be additive in `src/emel/machines.hpp`.
- Focused tests should be added to the existing `emel_tests_diarization` shard via `CMakeLists.txt`.

</code_context>

<specifics>
## Specific Ideas

- Prefer a `sortformer/pipeline` actor name because it describes this phase's connection work
  without claiming a generic runtime dispatcher.
- Require segment capacity large enough for the bounded worst case so segment decoding cannot fail
  after the dispatch path has already committed to execution.

</specifics>

<deferred>
## Deferred Ideas

- Replace fabricated parity and benchmark lanes in Phase 90.
- Remove existing request/executor action-side runtime branching in Phase 91.
- Backfill validation/frontmatter/ledger evidence in Phase 92.

</deferred>
