---
phase: 125
plan: 01
status: complete
completed: 2026-04-28
requirements-completed:
  - CLOSE-01
---

# Summary 125.1

## Supersession Notice

This closeout result is historical. Phase 126 superseded it by removing hidden recognizer backend
dispatch, Phase 127 superseded it by closing decoder ownership, and Phase 128 stabilizes the
benchmark/evidence ledger before archive.

## Completed

- Reran the v1.16 source-backed closeout audit after recognizer-backed parity and benchmark
  evidence landed.
- Verified the maintained proof path enters `emel::speech::recognizer::sm`, uses the
  `speech/recognizer_routes/whisper` backend, and publishes recognizer-backed compare/benchmark
  metadata.
- Replaced the stale `gaps_found` audit with a passing audit that explicitly traces the maintained
  model/tokenizer/audio path through recognizer initialization, recognition, compare, benchmark,
  and metadata publication.
- Reran full-scope quality gates with Whisper compare and single-thread benchmark suites selected.

## Verification Highlights

- Full-scope quality gate passed: 12/12 test shards, line coverage `90.8%`, branch coverage
  `55.6%`, paritychecker, fuzz smoke, docsgen, recognizer-backed compare, and recognizer-backed
  benchmark.
- Maintained compare passed with `comparison_status=exact_match reason=ok` and EMEL backend
  `emel.speech.recognizer.whisper`.
- Maintained benchmark passed with `status=ok reason=ok`; latest full-gate output records EMEL
  mean `59,106,792 ns` versus reference mean `59,958,847 ns`.
- Domain-boundary script passed; generic recognizer leak grep and forbidden-root grep returned no
  matches.

## Residual Work

No work remains for Phase 125's original recognizer closeout rerun. Phase 127 is now the active
source-backed closeout truth, and Phase 128 owns the remaining evidence cleanup.
