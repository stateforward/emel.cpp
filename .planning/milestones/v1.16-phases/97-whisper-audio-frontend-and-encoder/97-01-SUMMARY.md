---
phase: 97-whisper-audio-frontend-and-encoder
plan: 01
requirements-completed: [ASR-01]
completed: 2026-04-26
---

# Phase 97 Plan 01: Whisper Audio Frontend And Encoder - Execution Summary

**Phase Goal:** Add the maintained audio request surface, mel feature preparation, and encoder
execution path.

**Status:** Complete for Phase 97 scope. The implementation is a real EMEL-owned encoder path,
not a skeleton, one-block shortcut, direct-DFT frontend, generated-filter substitute, or
tool-local compute path.

## Outcomes

### Public Request Surface

- Added `emel::whisper::encoder::sm` and top-level alias `emel::WhisperEncoder`.
- Added public `event::encode` with immutable PCM/model inputs, caller-provided bounded workspace
  and encoder-state buffers, output frame/width/digest references, optional error output, and
  immediate callbacks.
- Validation rejects invalid sample rate, channel count, PCM shape/finite values, undersized
  output, undersized workspace, malformed model contract, and unsupported encoder weight variant.

### Kernel-Owned Frontend And Encoder

- Added `src/emel/kernel/whisper/detail.hpp` as the Whisper-owned kernel surface for Phase 97
  numeric work.
- Implemented an exact 400-point frontend through a radix-2 FFT core using a 1024-point
  Bluestein convolution, so the code consumes the loaded Whisper `mel_filters` tensor
  (`201 x 80`) without direct DFT or changing the mel-filter contract.
- Implemented conv1/conv2 + GELU, positional embedding add, four full encoder blocks, attention,
  layernorm, feed-forward MLP, final layernorm, and deterministic FNV-style encoder-state digest.
- Linear weight routing is explicit in SML guards/transitions for `q8_0`, `q4_0`, and `q4_1`.
  The tested staged fixture routes through `q8_0`; q4_0/q4_1 fixture tests remain skipped because
  those larger model files are not staged locally.

### SML Orchestration

- Runtime choice lives in `guards.hpp`/`sm.hpp`:
  - q8_0 -> `effect_run_encoder_q8_0`
  - q4_0 -> `effect_run_encoder_q4_0`
  - q4_1 -> `effect_run_encoder_q4_1`
  - unsupported -> explicit error path
- Actions execute already-selected behavior and update dispatch counters.
- Per-dispatch request/output pointers stay in event payloads; context stores only persistent
  dispatch counters.
- Dispatch does not allocate; callers provide workspace and output buffers.

### Tests

- Added `tests/whisper/encoder/lifecycle_tests.cpp` with public-event tests:
  - invalid audio contracts reject through the actor and set explicit errors.
  - q8_0 pinned model fixture runs the full four-layer encoder and emits a deterministic digest.
  - mutating loaded `mel_filters` changes the encoder digest, proving the frontend consumes model
    filters rather than generated filters.
- Added `tests/whisper/fixtures/README.md` documenting the deterministic Phase 97 PCM fixture
  contract.

## Verification Commands

- `cmake --build build/audit-native --target emel_tests_bin` - passed.
- `build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/whisper/*'` -
  **3 cases, 3 passed, 1049 assertions, 0 failures**.
- `build/audit-native/emel_tests_bin --no-breaks --test-case='*whisper_encoder*,*model_whisper*,*kernel_aarch64_q4*,*kernel_aarch64_q8*'` -
  **23 cases, 23 passed, 1686 assertions, 0 failures**. q4_0/q4_1 real-fixture parse tests skip
  because `tests/models/whisper-tiny-q4_0.gguf` and `tests/models/whisper-tiny-q4_1.gguf` are not
  staged locally.
- `ctest --test-dir build/audit-native -R 'emel_tests_whisper|lint_snapshot' --output-on-failure` -
  **2/2 tests passed** after the explicitly approved `snapshots/lint/clang_format.txt` update.

## Files Changed

| File | Change |
|------|--------|
| `src/emel/kernel/whisper/detail.hpp` | Whisper-owned frontend/encoder numeric kernels and workspace sizing. |
| `src/emel/whisper/encoder/{context,events,guards,actions,sm,detail,errors}.hpp` | Public encoder actor, events, validation, explicit variant routing, and errors. |
| `tests/whisper/encoder/lifecycle_tests.cpp` | Public-event encoder tests against the staged q80 fixture. |
| `tests/whisper/fixtures/README.md` | Deterministic PCM fixture contract. |
| `CMakeLists.txt` | Adds Whisper encoder tests and test shard. |
| `.planning/phases/97-whisper-audio-frontend-and-encoder/*` | Context, plan, summary, and verification artifacts. |

## Notes

- The deterministic PCM fixture is intentionally small (320 mono 16 kHz samples) so unit tests run
  the full four-layer encoder without making the suite all-day. This is encoder-state evidence,
  not transcript or `whisper.cpp` parity evidence.
- Decoder, transcript output, token handling, and parity records remain Phase 98/99.
