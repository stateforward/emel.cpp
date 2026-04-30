---
phase: 107
status: research_complete
researched: 2026-04-27
---

# Phase 107 Research

## Findings

- `event::initialize` accepted tokenizer JSON without asset identity, so a caller could reach
  recognizer initialization without the pinned `tokenizer-tiny.json` checksum being represented
  in the maintained event contract.
- An earlier route-binding action allocated workspaces and child machines during SML initialize
  dispatch. This violated the no-allocation-during-dispatch rule.
- Whisper ASR prompt construction was a local action array. The policy existed implicitly as
  `[sot, language_en, transcribe]`, not as a named speech-domain contract with timestamp and
  suppression semantics.
- `scripts/bench_whisper_compare.sh` verified tokenizer SHA through setup only when the reference
  setup path ran. `--skip-reference-build` skipped that verification.

## Implementation Direction

- Add tokenizer asset identity to `event::tokenizer_assets` and validate it in tokenizer guards.
- Keep route storage allocation outside dispatch; Phase 114 ultimately selected the existing
  encoder/decoder/tokenizer surface instead of a generic recognizer route.
- Add an explicit `k_tiny_asr_decode_policy` with language, task, timestamp, suppression, and
  prompt sequence fields.
- Add checksum enforcement to the Whisper compare script and Python compare/benchmark entrypoints.
