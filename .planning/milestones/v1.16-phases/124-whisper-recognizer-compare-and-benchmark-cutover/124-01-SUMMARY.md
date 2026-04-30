---
phase: 124
plan: 01
status: complete
completed: 2026-04-28
requirements-completed:
  - REOPEN-01
  - PARITY-01
  - PERF-03
---

# Summary 124.1

## Completed

- Cut `tools/bench/whisper_emel_parity_runner.cpp` over from direct Whisper
  encoder/decoder/tokenizer orchestration to `emel::speech::recognizer::sm` plus the
  `speech/recognizer_routes/whisper` backend.
- Added a branch-free generated-token-count output to the public recognizer recognition event so
  proof tools can report token count without reaching into decoder state.
- Updated `tools/bench/whisper_compare.py` and `tools/bench/whisper_benchmark.py` to publish the
  EMEL lane as `emel.speech.recognizer.whisper` with runtime surface
  `speech/recognizer+speech/recognizer_routes/whisper`.
- Added benchmark regression coverage that the runner source stays on public recognizer surfaces
  and the compare/benchmark summaries emit recognizer-backed metadata.

## Verification Highlights

- Focused recognizer tests passed: 7 test cases / 125 assertions.
- Focused Whisper public-recognizer fixture test passed: 1 test case / 356 assertions.
- Whisper benchmark tool tests passed: 10 test cases / 148 assertions.
- `scripts/check_domain_boundaries.sh` passed.
- Generic recognizer leak grep and forbidden-root grep returned no matches.
- Recognizer-backed compare passed with `status=exact_match reason=ok`.
- Recognizer-backed single-thread benchmark passed with `benchmark_status=ok reason=ok`; the
  quality-gate artifact records EMEL mean `58,263,986 ns` versus reference mean `60,507,152 ns`.
- Changed-file scoped `scripts/quality_gates.sh` passed with
  `EMEL_QUALITY_GATES_BENCH_SUITE=whisper_compare:whisper_single_thread`; changed recognizer
  header coverage was line `100.0%`.

## Residual Work

Phase 125 must rerun the source-backed closeout audit now that parity and performance evidence use
the public recognizer lane.
