---
phase: 124
status: pass
validated: 2026-04-28
---

# Validation 124

| Success Criterion | Status | Evidence |
|-------------------|--------|----------|
| Runner drives `emel::speech::recognizer::sm` through public recognizer events. | Pass | Runner now includes `emel/speech/recognizer/sm.hpp`, initializes with `recognizer_routes::whisper::backend()`, and dispatches `event::initialize` / `event::recognize`. |
| Runner no longer constructs encoder/decoder actors or calls tokenizer decode directly. | Pass | Source grep found no direct `encoder::whisper`, `decoder::whisper`, `speech/encoder/whisper`, `speech/decoder/whisper`, or `decode_token_ids` references in the runner. |
| Compare and benchmark metadata identify the public recognizer lane. | Pass | Compare and benchmark summaries both publish `backend_id=emel.speech.recognizer.whisper` and `runtime_surface=speech/recognizer+speech/recognizer_routes/whisper`. |
| Recognizer-backed compare exact-matches `[C]` and hard-fails drift. | Pass | Maintained compare returned `status=exact_match reason=ok`; benchmark tests still cover transcript mismatch hard failure. |
| Recognizer-backed benchmark records EMEL faster than the matched reference. | Pass | Quality-gate benchmark summary records EMEL mean `58,263,986 ns` and reference mean `60,507,152 ns`. |
| Domain boundaries remain intact. | Pass | `scripts/check_domain_boundaries.sh`, generic recognizer leak grep, and forbidden-root grep all passed. |

## Verdict

Phase 124 satisfies `REOPEN-01`, `PARITY-01`, and `PERF-03` for the reopened v1.16 gap scope. The
remaining milestone work is Phase 125 closeout rerun.
