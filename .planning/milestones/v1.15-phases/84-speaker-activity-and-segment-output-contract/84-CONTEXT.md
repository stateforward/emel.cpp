# Phase 84: Speaker Activity And Segment Output Contract - Context

**Gathered:** 2026-04-23
**Status:** Ready for planning
**Mode:** Autonomous smart discuss defaulted by prior user direction

<domain>
## Phase Boundary

Phase 84 owns deterministic Sortformer diarization outputs after native hidden-state execution. It
must publish a fixed-profile `T x 4` speaker-activity probability matrix and bounded segment
records with stable speaker labels and monotonic timestamps. It does not own reference parity,
benchmark publication, or ARM optimization.

</domain>

<decisions>
## Implementation Decisions

### Output Ownership
- Add output decoding under `src/emel/diarization/sortformer/output/`.
- Keep executor modules/cache/transformer behavior unchanged; output conversion consumes the
  hidden-state handoff and maintained module head tensors.
- Do not add JSON libraries, dynamic allocation, file I/O, or tool-only output conversion paths.

### Probability Contract
- Emit one probability per frame per maintained speaker: `188 x 4`.
- Use the maintained `mods.sh2s.*` speaker head over `188 x 192` hidden frames.
- Apply deterministic sigmoid normalization with bounded numeric input.

### Segment Contract
- Decode each speaker independently from the probability matrix, so overlap is explicit and
  allowed.
- Use stable speaker indexes `0..3`, corresponding to labels `speaker_0` through `speaker_3`.
- Store timestamps as frame indexes plus deterministic seconds derived from the 0.08-second frame
  shift.
- Require caller-owned segment storage and report failure if capacity is insufficient.

### the agent's Discretion
- The default threshold and exact fixed-record struct layout are at the agent's discretion, as long
  as repeated runs are deterministic and snapshot-friendly.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `modules::detail::contract` already binds `mods.sh2s.*` and `mods.h2s.*`.
- `executor::detail` defines the maintained frame count, hidden width, and speaker count.
- Existing Sortformer detail components use fixed-profile constants and caller-owned spans.

### Established Patterns
- Component-local detail files own numeric and data-plane helpers.
- Tests live under `tests/diarization/sortformer/<component>/lifecycle_tests.cpp`.
- CMake explicitly lists new component sources and test files.

### Integration Points
- Phase 84 output helpers consume Phase 83.4 hidden output.
- Phase 85 can use the snapshot-friendly probability and segment records for parity proof.

</code_context>

<specifics>
## Specific Ideas

- Add `compute_speaker_probabilities` for `hidden_frames -> 188 x 4` probabilities.
- Add `decode_segments` for independent multi-speaker active-region extraction.
- Add tests for deterministic probabilities, overlap allowance, monotonic segment timestamps,
  insufficient segment capacity, and invalid shape rejection.

</specifics>

<deferred>
## Deferred Ideas

- Reference comparison and canonical fixture proof remain Phase 85.
- ARM profiling and optimization remain Phases 86-88.

</deferred>
