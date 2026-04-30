# Phase 123: Whisper Public Recognizer Runtime Wiring - Context

**Gathered:** 2026-04-28T01:10:03Z
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase makes `emel::speech::recognizer::sm` the public actor entrypoint for the maintained
Whisper ASR runtime path. The generic recognizer files must remain model-family clean: no
`whisper` names, no `emel::model::whisper` contract exposure, and no generic recognizer headers
that hardcode a variant. Variant-specific composition belongs in a clearly named speech-domain
route outside the generic recognizer tree.

</domain>

<decisions>
## Implementation Decisions

### Public Recognizer Boundary
- Keep `src/emel/speech/recognizer/**` generic and free of Whisper identifiers so
  `scripts/check_domain_boundaries.sh` remains a useful tripwire.
- Add a generic backend/route contract to recognizer public events, with explicit SML guards
  deciding when initialization and recognition can proceed.
- Store only the selected backend route pointer as persistent recognizer actor state after
  initialization; do not store per-request model, tokenizer, output, phase, or status data in
  context.
- Keep recognizer dispatch run-to-completion and avoid queues, re-entrancy, wall-clock reads, and
  allocation during dispatch.

### Whisper Route Ownership
- Place Whisper-specific recognizer composition in a variant-named speech route outside the generic
  recognizer directory.
- Let that route construct speech-owned encoder and decoder public events and call child
  `process_event(...)` synchronously inside the recognizer RTC chain.
- Keep model-family binding validation at the variant boundary and keep encoder/decoder public
  contracts speech-owned.
- Keep tokenizer policy and transcript publication under `speech/tokenizer/whisper`.

### Event And Storage Contract
- Add caller-owned dispatch-local scratch spans to the recognizer recognition event so the public
  recognizer path does not allocate in the hot path.
- Validate the pinned tokenizer identity through the route before initialization succeeds.
- Validate ASR decode-policy support before selected recognition execution.
- Publish transcript text from tokenizer-backed generated token IDs, not from tool-local direct
  tokenizer calls.

### Verification
- Replace tests that assert the maintained recognizer backend is unsupported with tests proving a
  backend route can initialize and recognize through public recognizer events.
- Add a Phase 99 Whisper fixture/audio/tokenizer test outside `tests/speech/recognizer` so the
  generic recognizer leak check remains strict.
- Run focused recognizer/Whisper tests, domain-boundary checks, and forbidden-root greps.
- Leave compare and single-thread benchmark cutover to Phase 124.

### the agent's Discretion
The exact route namespace and helper names are at the agent's discretion as long as the route is
truthfully variant-specific, the public recognizer actor is the entrypoint, and no tool-only
encoder/decoder/tokenizer orchestration is needed for recognition after this phase.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/speech/recognizer/sm.hpp` already models initialize and recognition as an SML actor,
  but model support and backend readiness guards currently return false.
- `src/emel/speech/encoder/whisper/sm.hpp` and `src/emel/speech/decoder/whisper/sm.hpp` expose
  public child actor events for encode/decode dispatch.
- `src/emel/speech/tokenizer/whisper/any.hpp` exposes the tiny ASR decode policy, tokenizer
  control-token validation, and token-id transcript publication.
- `tools/bench/whisper_emel_parity_runner.cpp` contains the current bypass orchestration that
  Phase 124 will cut over after this route exists.

### Established Patterns
- Runtime behavior choice belongs in `sm.hpp` transitions and pure guards.
- Actions execute already-selected paths and may call child actors through public
  `process_event(...)` interfaces.
- Public event payloads carry dispatch-local buffers and outputs by span/reference, not context.
- Variant/domain leak checks intentionally reject `whisper` inside generic recognizer files and
  tests.

### Integration Points
- Recognizer initialization should accept a tokenizer asset with model JSON and a pinned SHA-256.
- Recognition must receive caller-owned encoder workspace, encoder state, decoder workspace,
  logits, generated-token storage, transcript storage, and scalar outputs.
- Phase 124 will make the maintained compare/benchmark runner pass this route into
  `emel::speech::recognizer::sm`.

</code_context>

<specifics>
## Specific Ideas

The user's blocker is authoritative: parity and benchmark proof must drive through the recognizer,
not an ad hoc runner path. Phase 123 builds the recognizer-owned runtime route; Phase 124 removes
the bypass from the proof tools.

</specifics>

<deferred>
## Deferred Ideas

No broader model-family routing framework and no benchmark publication changes in this phase.
Those are Phase 124 responsibilities.

</deferred>
