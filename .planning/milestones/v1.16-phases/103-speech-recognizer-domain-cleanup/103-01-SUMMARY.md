---
phase: 103
plan: 1
status: complete
one-liner: Whisper runtime actors moved out of the top-level Whisper domain and into speech recognizer ownership.
requirements:
  - REOPEN-01
  - SPEECH-01
---

# Summary 103.1

## Completed

- Moved `src/emel/whisper/encoder/**` to
  `src/emel/speech/recognizer/whisper/encoder/**`.
- Moved `src/emel/whisper/decoder/**` to
  `src/emel/speech/recognizer/whisper/decoder/**`.
- Updated namespaces and includes from `emel::whisper::*` / `emel/whisper/...` to
  `emel::speech::recognizer::whisper::*` / `emel/speech/recognizer/whisper/...`.
- Updated Whisper tests and the EMEL parity runner to use the speech recognizer namespace.
- Removed the interrupted diagnostic hardcoded-token experiment before continuing.

## Verification

- `test ! -d src/emel/whisper`
- `rg -n "emel/whisper|emel::whisper|namespace emel::whisper" src tests tools CMakeLists.txt -S`
  returns no matches.
- `cmake --build build/audit-native --target emel_tests_bin --parallel`
- `build/audit-native/emel_tests_bin --no-breaks --source-file='*tests/whisper/*'`
  passed 12/12 test cases and 1813/1813 assertions.
- `cmake --build build/whisper_compare_tools --parallel --target whisper_emel_parity_runner`
- `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build` still reports
  `bounded_drift reason=transcript_mismatch`, preserving the reopened blocker for Phase 104/105.
