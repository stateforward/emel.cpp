# Phase 127: Whisper Decoder Ownership Gap Closure - Context

**Gathered:** 2026-04-28
**Status:** Ready for planning
**Mode:** Auto-generated autonomous context

<domain>
## Phase Boundary

Move maintained Whisper decoder sequence, logits, timestamp-aware token selection, and
decode-loop execution out of `src/emel/speech/encoder/whisper/detail.hpp`. The public recognizer
path must continue to dispatch encoder, decoder, and detokenizer actors through public events, but
decoder runtime work must be owned by `src/emel/speech/decoder/whisper/**` or an explicitly
justified shared kernel-owned numeric surface.

</domain>

<decisions>
## Implementation Decisions

### Ownership Repair
- Treat `decoder/whisper/actions.hpp` and `decoder/whisper/guards.hpp` dependencies on
  `encoder/whisper/detail.hpp` as in-scope blockers.
- Keep recognizer SML route graph behavior unchanged unless source inspection proves it must
  change.
- Prefer decoder-owned runtime helpers for this phase because the blocker is decoder execution
  using encoder-owned detail code.
- Do not introduce new model-family roots or generic public Whisper leakage.

### Verification
- Add focused regression evidence that decoder action/guard code no longer includes or aliases
  encoder detail.
- Preserve exact `[C]` compare evidence and single-thread benchmark performance.
- Run SML behavior-selection and domain-boundary scans over the maintained Whisper path.
- Re-run source-backed milestone audit after phase execution.

### the agent's Discretion
Implementation details are at the agent's discretion within the ownership, SML, and no-fallback
contracts above.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/speech/decoder/whisper/detail.hpp` already owns decoder constants, contract binding,
  and workspace sizing.
- `src/emel/speech/decoder/whisper/sm.hpp` already models validation, variant selection, and
  success/error publication with explicit SML transitions.
- `src/emel/speech/recognizer_routes/whisper/actions.hpp` dispatches the decoder actor via public
  `decoder::whisper::event::decode`.

### Established Patterns
- Runtime variant choice belongs in `sm.hpp` guarded transitions; actions execute already-chosen
  variant paths.
- Speech encoder/decoder event contracts are speech-owned and must not expose model-family public
  contracts.
- Quality gates are scoped with `EMEL_QUALITY_GATES_CHANGED_FILES` and Whisper benchmark suites
  for this milestone.

### Integration Points
- `src/emel/speech/decoder/whisper/actions.hpp`
- `src/emel/speech/decoder/whisper/guards.hpp`
- `src/emel/speech/decoder/whisper/detail.hpp`
- `tests/speech/decoder/whisper/lifecycle_tests.cpp`
- `tools/bench/whisper_emel_parity_runner.cpp`

</code_context>

<specifics>
## Specific Ideas

- Remove the include of `emel/speech/encoder/whisper/detail.hpp` from decoder action and guard
  headers.
- Ensure greps for `encoder/whisper/detail` and `encoder::whisper::detail` under
  `src/emel/speech/decoder/whisper` return no matches.
- Keep compare and benchmark metadata on `emel.speech.recognizer.whisper`.

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope.

</deferred>
