# Phase 98: Whisper Decoder And Transcript Runtime - Context

**Gathered:** 2026-04-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Add decoder execution, token handling, transcript output, and explicit runtime/error
orchestration for the maintained Whisper tiny GGUF runtime. This phase extends the already-landed
encoder request surface with a decoder actor that consumes EMEL-owned encoder state and model
contract data. Parity records and benchmark publication remain Phase 99 and Phase 100 work.

</domain>

<decisions>
## Implementation Decisions

### Runtime Surface
- Add a separate `emel::whisper::decoder::sm` actor rather than widening the encoder actor.
- Use a public `event::decode` request with caller-owned workspace, logits, token, transcript,
  and digest outputs.
- Keep public payload fields small and synchronous; do not store per-request pointers in context.
- Publish immediate callbacks and optional error output in the same RTC chain.

### Decoder Scope
- Execute one bounded decoder step for the maintained fixture using the existing decoder tensors.
- Produce logits for the full maintained vocab and select the deterministic argmax token.
- Publish a deterministic transcript string from the selected token ID.
- Leave stored `whisper.cpp` comparison records and token/timestamp parity to Phase 99.

### Rule Boundaries
- Put numeric decoder work in `src/emel/kernel/whisper/detail.hpp`.
- Put runtime variant choice in `src/emel/whisper/decoder/guards.hpp` and `sm.hpp`.
- Keep actions allocation-free and limited to already-selected work.
- Validate capacity, prompt token, model contract, and unsupported variants through explicit
  states/transitions.

### the agent's Discretion
The implementation may use the same q8_0/q4_0/q4_1 variant structure established by Phase 97 and
may use a deterministic token-id transcript representation until Phase 99 introduces the stored
parity record format.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/whisper/encoder/*` provides the component layout, event shape, callbacks, errors,
  guards, actions, and destination-first SML table pattern.
- `src/emel/kernel/whisper/detail.hpp` already owns mel, encoder, linear, attention, layernorm,
  digest, and q8/q4 read helpers.
- `tests/whisper/encoder/lifecycle_tests.cpp` provides a fixture-driven public-event test pattern.

### Established Patterns
- Whisper runtime actors use caller-owned buffers and persistent context counters only.
- Runtime dtype choice is modeled by explicit q8_0/q4_0/q4_1 guards and transitions.
- Focused doctests drive public `process_event(...)` surfaces rather than internal actions.

### Integration Points
- Decoder consumes `emel::model::whisper::detail::execution_contract`.
- Decoder consumes encoder state produced by `emel::whisper::encoder::sm`.
- The Whisper test shard `emel_tests_whisper` covers both encoder and decoder lifecycle tests.

</code_context>

<specifics>
## Specific Ideas

Use `tests/models/model-tiny-q80.gguf` and deterministic 16 kHz PCM to produce encoder state,
then drive the decoder public event with the start-of-transcript prompt token.

</specifics>

<deferred>
## Deferred Ideas

- Stored `whisper.cpp` parity records and token/timestamp comparison belong to Phase 99.
- Single-thread benchmark records belong to Phase 100.
- ARM profiling and optimization belong to Phase 101.

</deferred>
