---
phase: 114
plan: 01
status: complete
completed: 2026-04-27
requirements_completed:
  - SPEECH-01
  - TOK-01
  - TOK-02
  - POLICY-01
  - PARITY-01
---

# Phase 114 Summary

## Outcome

The maintained v1.16 Whisper ASR runtime surface is now explicitly documented and source-backed as:

```text
speech/encoder/whisper+speech/decoder/whisper+speech/tokenizer/whisper
```

The generic `speech::recognizer::sm` is not the maintained Whisper path for this milestone.
Keeping it unsupported avoids a variant leak into a generic recognizer component and matches
`scripts/check_domain_boundaries.sh`.

## Changes

- Updated `tools/bench/whisper_emel_parity_runner.cpp` to publish
  `backend_id: emel.speech.whisper.encoder_decoder` and the runtime surface above.
- Updated `tools/bench/whisper_compare.py` and `tools/bench/whisper_benchmark.py` so EMEL error
  and benchmark records use the same backend id and runtime surface.
- Reran compare and benchmark evidence after cleaning generated Whisper output directories.

## Evidence

- Compare summary: `comparison_status=exact_match`, EMEL transcript `[C]`, reference transcript
  `[C]`, EMEL backend `emel.speech.whisper.encoder_decoder`.
- Benchmark summary: `status=ok`, `reason=ok`, EMEL mean `56,901,792 ns`, reference mean
  `65,542,792 ns`.
- Domain boundary: `scripts/check_domain_boundaries.sh` passed.

## Requirement Impact

`SPEECH-01`, `TOK-01`, `TOK-02`, `POLICY-01`, and `PARITY-01` are complete through the selected
speech-owned runtime surface.
