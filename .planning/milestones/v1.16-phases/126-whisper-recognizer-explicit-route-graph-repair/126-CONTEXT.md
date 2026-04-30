# Phase 126: Whisper Recognizer Explicit Route Graph Repair - Context

**Gathered:** 2026-04-28
**Status:** Ready for planning

<domain>
## Phase Boundary

Repair the public speech recognizer route boundary so route support, readiness, encode, decode,
and detokenize behavior are represented by explicit Boost.SML guards, states, transitions, and
compile-time route policy types instead of a runtime `runtime_backend` function-pointer bundle.

</domain>

<decisions>
## Implementation Decisions

### Infrastructure Scope
- Treat this as a source-backed SML rule repair, not a behavior expansion.
- Preserve the recognizer-backed `[C]` parity and single-thread benchmark contracts.
- Keep the generic recognizer public boundary model-family-free; variant-specific Whisper code
  stays under `speech/recognizer_routes/whisper`, `speech/encoder/whisper`,
  `speech/decoder/whisper`, `speech/tokenizer/whisper`, and `model/whisper`.
- Do not add fallback kernels, tool-only compute paths, or new model-family runtime roots.

### the agent's Discretion
- Choose the smallest route-policy structure that removes hidden runtime behavior selection while
  keeping existing caller-owned storage and public recognizer dispatch ergonomics.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/speech/recognizer/sm.hpp` already has explicit initialize and recognition phase
  states.
- `src/emel/speech/recognizer_routes/whisper/detail.cpp` already contains the maintained Whisper
  support/readiness and encode/decode/detokenize route logic.
- `tools/bench/whisper_emel_parity_runner.cpp` already drives `emel::speech::recognizer::sm`
  through public initialize and recognize events.

### Established Patterns
- Recognizer events carry caller-owned storage spans and synchronous callbacks.
- Route-specific code belongs outside the generic recognizer tree.
- SML transition tables use destination-first rows and typed completion propagation.

### Integration Points
- `src/emel/speech/recognizer/events.hpp`, `context.hpp`, `guards.hpp`, `actions.hpp`, and
  `sm.hpp`.
- `src/emel/speech/recognizer_routes/whisper/**`.
- `tests/speech/recognizer/lifecycle_tests.cpp`,
  `tests/speech/encoder/whisper/lifecycle_tests.cpp`, and
  `tools/bench/whisper_emel_parity_runner.cpp`.

</code_context>

<specifics>
## Specific Ideas

Use the latest source-backed audit as the acceptance contract: `scripts/check_sml_behavior_selection.sh`
must pass on the maintained recognizer path, and `CLOSE-01` must remain pending until a new audit
passes after this repair.

</specifics>

<deferred>
## Deferred Ideas

None - this phase is narrowly scoped to the explicit recognizer route graph repair.

</deferred>
