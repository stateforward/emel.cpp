---
phase: 04-deterministic-generation-parity
plan: 02
subsystem: paritychecker
tags: [generation, parity, llama-cpp, reference]
requires: [04-01]
provides:
  - Truthful paritychecker-local decode callbacks backed by `llama_decode`
  - A direct reference generation path for the same prompt and deterministic settings
  - Parity success that depends on EMEL-vs-reference agreement rather than EMEL completion alone
affects: [paritychecker]
tech-stack:
  added: []
  patterns: [Tool-local reference bridge confined to `tools/paritychecker/`]
key-files:
  created: []
  modified:
    - tools/paritychecker/parity_runner.cpp
key-decisions:
  - "Kept all `llama_*` integration inside `tools/paritychecker/` and did not widen any `src/` API or machine surface."
  - "Matched the reference evidence to EMEL's current renderer semantics by comparing raw vocab-piece output instead of a separately detokenized string."
patterns-established:
  - "Pattern: parity success in generation mode now requires agreement between the EMEL path and a direct reference path over the same bounded request."
requirements-completed: [PARI-01]
duration: 10min
completed: 2026-03-08
---

# Phase 4 Plan 02 Summary

**The Phase 4 slice now compares EMEL generation against a real reference-side generation run**

## Accomplishments
- Replaced the earlier placeholder logits callback in [parity_runner.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.cpp) with a truthful paritychecker-local decode bridge that uses the loaded reference model and `llama_decode` to supply real logits for the EMEL generator path.
- Added a direct reference generation loop for the same prompt, token budget, and deterministic argmax selection used by the EMEL path.
- Changed generation mode so success is contingent on EMEL output matching the reference output for the pinned Llama-68M slice.

## Task Commits
- No commit created during this execution. The plan was completed locally on `next` while preserving unrelated workspace changes.

## Deviations from Plan
- I adjusted the comparison renderer after observing an initial whitespace-format mismatch (`" to"` vs `▁to`). The final parity record compares the raw vocab-piece representation because that is what the current EMEL text path emits for this slice.

## Verification
- `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1`
- `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1 | rg "parity"`

## Next Readiness
- Wave 3 could build on the dual-path result objects and focus only on compact structured evidence plus full repo verification.
