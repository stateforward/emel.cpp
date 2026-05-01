---
phase: 105
name: Whisper Exact Transcript Parity Closure
status: planned
---

# Phase 105 Context

The milestone cannot close while `build/whisper_compare/summary.json` reports `bounded_drift`.
This phase replaces the previous closeout with exact transcript parity against the pinned
`whisper.cpp` lane and refreshes audit/gate evidence.

## 2026-04-26 execution note

Implemented source-backed native corrections but did not close exact parity:

- Private decoder now accepts a prompt-token span, runs a bounded greedy generated-token loop, and
  passes generated token IDs through `speech/tokenizer/whisper` instead of a single hardcoded SOT
  token.
- Whisper mel frontend now follows the reference log-mel shape more closely: reflective pre-padding,
  `log10`, clamp to `max - 8`, and `(x + 4) / 4` normalization.
- Focused speech/Whisper tests pass, and scoped quality gate passes coverage at `99.6%` line /
  `57.7%` branch.

Closure update:

- Parity compare now normalizes the pinned whisper.cpp model artifact into a generated EMEL GGUF
  under `build/whisper_compare/normalized/` before running the EMEL lane. The normalizer records
  both source and generated SHA values in the compare summary.
- EMEL runtime remains on the maintained GGUF loader path; whisper.cpp remains on its pinned
  reference model path. The bench-side normalizer is the only compatibility bridge.
- Whisper encoder/decoder now route q8 linear + f32 auxiliary tensor variants explicitly through
  SML guards and transitions, matching the normalized pinned model's operand classes.
- `scripts/bench_whisper_compare.sh --skip-reference-build --skip-emel-build` reports
  `exact_match reason=ok`; both lanes emit `[C]` for the short-context greedy parity case.
- Scoped quality gate passes with `99.5%` line coverage and `50.2%` branch coverage.
