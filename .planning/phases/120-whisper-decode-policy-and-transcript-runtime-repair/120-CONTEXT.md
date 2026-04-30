# Phase 120: Whisper Decode Policy And Transcript Runtime Repair - Context

**Gathered:** 2026-04-27T23:21:30Z
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase repairs the live Whisper decoder runtime and proof harness so `TOK-02` and
`POLICY-01` are source-backed by behavior, not just compare metadata. The work stays inside
speech-owned Whisper surfaces: tokenizer policy under `src/emel/speech/tokenizer/whisper`,
decoder runtime contracts/events/guards/actions under `src/emel/speech/decoder/whisper`, and
numeric decoder kernels under the existing speech encoder/decoder implementation path.

</domain>

<decisions>
## Implementation Decisions

### Policy Contract Wiring
- Carry the speech-owned ASR decode policy through the decoder public event rather than passing
  only `prompt_tokens`.
- Keep `emel::model::whisper` out of speech decoder events, context, guards, and actions.
- Use the existing `speech/tokenizer/whisper::asr_decode_policy` as the policy contract unless a
  smaller speech-owned runtime payload is necessary for allocation-free dispatch.
- Keep policy validation explicit and before selected decoder execution paths run.

### Runtime Behavior Choice
- Model timestamp/suppression behavior as explicit guards/states or compile-time selected action
  paths after policy acceptance.
- Do not move behavior selection into detail helper return values.
- Keep bulk numeric loops inside the existing allocation-free detail kernels once a policy path is
  already selected.
- Preserve exact `[C]` parity for the pinned Phase 99 audio/model pair.

### Transcript Ownership
- Remove the hardcoded public decoder `token:<id>` transcript output surface.
- Keep decoder output focused on token IDs, token count, confidence, and digest.
- Produce transcript text only through `speech/tokenizer/whisper::decode_token_ids` in the
  maintained publication path.
- Update tests so old token-placeholder assertions become token-output or tokenizer-publication
  assertions.

### Verification
- Add focused tests proving policy is actually passed into decoder runtime.
- Add regression coverage that decoder no longer writes a `token:<id>` transcript string.
- Keep benchmark/compare tests proving transcript mismatch fails and exact match passes.
- Run `scripts/check_domain_boundaries.sh` because this phase touches Whisper model-family runtime
  placement and speech-owned contracts.

### the agent's Discretion
The agent may choose exact type placement and naming as long as the public decoder event remains
speech-owned, SML routing owns behavior choice, and no model-family contract leaks into generic
public recognizer/generator/event/context headers.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/speech/tokenizer/whisper/any.hpp` exposes `asr_decode_policy`,
  `tiny_asr_decode_policy`, policy names, validation, and tokenizer-backed transcript decode.
- `src/emel/speech/decoder/whisper/events.hpp` owns the public decoder event and currently carries
  prompt-token spans and output buffers.
- `src/emel/speech/decoder/whisper/sm.hpp`, `guards.hpp`, and `actions.hpp` own decoder
  orchestration and variant routing.
- `tools/bench/whisper_emel_parity_runner.cpp` is the maintained EMEL compare/benchmark runner.

### Established Patterns
- Runtime behavior choice belongs in SML guards and transitions.
- Actions execute already-selected behavior paths and can call detail kernels for bounded numeric
  work.
- Public event payloads pass required fields by reference and keep mutable output buffers
  dispatch-local.
- Variant/model-family placement is protected by `scripts/check_domain_boundaries.sh`.

### Integration Points
- Decoder action currently calls `run_decoder_sequence` with prompt-token data only.
- The maintained runner gets the tokenizer policy from `speech/tokenizer/whisper`, then decodes
  generated token IDs into transcript text after decoder dispatch.
- Focused tests live under `tests/speech/decoder/whisper`, `tests/speech/tokenizer`, and
  `tools/bench/whisper_benchmark_tests.cpp`.

</code_context>

<specifics>
## Specific Ideas

Repair the exact contradictions from `.planning/milestones/v1.16-MILESTONE-AUDIT.md`: policy JSON
fields must describe runtime behavior actually consumed by decoder dispatch, and no public decoder
text surface may publish `token:<id>`.

</specifics>

<deferred>
## Deferred Ideas

No broader Whisper model-family or recognizer widening. Future variants require explicit
variant-clean phases and guards.

</deferred>
