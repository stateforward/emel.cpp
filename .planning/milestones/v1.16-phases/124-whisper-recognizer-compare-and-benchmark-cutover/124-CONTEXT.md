---
phase: 124
created: 2026-04-28
autonomous: true
source_gap: v1.16 milestone audit public recognizer bypass
---

# Context 124: Whisper Recognizer Compare And Benchmark Cutover

## Current State

Phase 123 added a public recognizer route for the maintained Whisper tiny ASR path. The route keeps
the generic recognizer model-family-free and exposes a variant-named backend under
`src/emel/speech/recognizer_routes/whisper/**`.

The remaining source-backed blocker is in the maintained proof tools. The EMEL compare and
benchmark lane still uses `tools/bench/whisper_emel_parity_runner.cpp`, which directly constructs
the Whisper encoder and decoder actors and calls tokenizer decode. That bypass means the exact
parity and performance artifacts do not prove the public recognizer actor path.

## Required Outcome

The maintained EMEL compare and benchmark lane must initialize `emel::speech::recognizer::sm` with
the Whisper recognizer route backend and run recognition through public recognizer events. Metadata
must name the recognizer-backed runtime surface so the closeout audit can trace the exact tool path.

## Constraints

- Do not let generic recognizer headers or tests expose Whisper names.
- Do not call Whisper encoder/decoder actors directly from the parity runner.
- Do not call tokenizer decode directly from the parity runner.
- Keep EMEL and reference lanes isolated; the reference lane may use `whisper.cpp` only for the
  comparison result.
- Preserve hard failure on transcript drift and benchmark performance regression.
- Run `scripts/check_domain_boundaries.sh` because this phase touches variant/runtime placement and
  recognizer evidence.
