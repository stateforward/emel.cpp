---
phase: 98-whisper-decoder-and-transcript-runtime
plan: 01
requirements-completed: [ASR-02, ASR-03, ASR-04]
completed: 2026-04-26
---

# Phase 98 Plan 01: Whisper Decoder And Transcript Runtime - Execution Summary

**Phase Goal:** Add decoder execution, token handling, transcript output, and explicit
runtime/error orchestration.

**Status:** Complete for Phase 98 scope.

## Outcomes

### Decoder Kernel Path

- Extended `src/emel/kernel/whisper/detail.hpp` with decoder constants, decoder workspace sizing,
  one-step decoder execution, full-vocab logits, argmax token selection, deterministic token-id
  transcript publication, and digest generation.
- Decoder execution uses the maintained Whisper decoder tensors:
  `model.decoder.embed_tokens.weight`, `model.decoder.embed_positions.weight`, each decoder layer's
  self-attention, encoder-attention, MLP, final layernorm tensors, and final decoder layernorm.
- The path consumes encoder state produced by the EMEL encoder actor and does not use
  `whisper.cpp` or ggml objects.

### Decoder Actor

- Added `src/emel/whisper/decoder/{context,errors,detail,events,guards,actions,sm}.hpp`.
- Added public `emel::whisper::decoder::event::decode` with caller-owned workspace, logits,
  transcript buffer, token output, confidence output, transcript-size output, digest output, and
  optional immediate callbacks.
- Runtime q8_0/q4_0/q4_1 variant selection is explicit in guards and destination-first SML
  transitions.
- Validation paths explicitly reject invalid model contract, encoder state, prompt token, logits
  capacity, transcript capacity, workspace capacity, and unsupported variants.

### Tests

- Added `tests/whisper/decoder/lifecycle_tests.cpp`.
- Tests load `tests/models/model-tiny-q80.gguf`, build the EMEL Whisper execution contract, run the
  existing encoder actor, then run the new decoder actor from the public event.
- Tests verify invalid prompt token and logits-capacity errors, deterministic q8_0 token/logit/
  transcript/digest output, and q8_0 dispatch attribution.

## Verification Commands

- `cmake --build build/audit-native --target emel_tests_bin` - passed.
- `build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/whisper/*'` -
  **5 cases, 5 passed, 1761 assertions, 0 failures**.
- `ctest --test-dir build/audit-native -R 'emel_tests_whisper|lint_snapshot' --output-on-failure` -
  **2/2 tests passed** after updating the approved lint snapshot baseline for new decoder files.
- `EMEL_QUALITY_GATES_CHANGED_FILES="<Phase 98 files>" scripts/quality_gates.sh` - passed
  build/scoped gate; unrelated coverage, parity, fuzz, bench, and docs lanes were skipped.

## Files Changed

| File | Change |
|------|--------|
| `src/emel/kernel/whisper/detail.hpp` | Decoder constants, workspace sizing, one-step decoder execution, logits, token selection, transcript publication. |
| `src/emel/whisper/decoder/*` | New decoder actor component. |
| `tests/whisper/decoder/lifecycle_tests.cpp` | Public-event decoder lifecycle tests. |
| `CMakeLists.txt` | Adds decoder lifecycle tests to `emel_tests_bin`. |
| `snapshots/lint/clang_format.txt` | Approved file-list snapshot update for decoder files. |

## Notes

- Transcript publication is deterministic and token-id based in this phase. Stored
  `whisper.cpp` parity records with transcript/token/timestamp comparison are Phase 99 work.
- The staged local fixture is q8_0. q4_0/q4_1 decoder routing is implemented and validated by
  source guards, but local end-to-end evidence remains q8_0 until larger fixtures are staged.
