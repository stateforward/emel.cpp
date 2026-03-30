---
phase: 28
slug: qwen3-parity-and-regression
created: 2026-03-28
status: complete
---

# Phase 28 Context

## Objective

Convert the canonical Qwen3 maintained generation surface from temp-baseline write proof to stored
snapshot-backed parity proof, then lock that behavior behind regression coverage without regressing
the prior maintained Llama slice.

## Locked Decisions

- The canonical slice stays fixed to `tests/models/Qwen3-0.6B-Q8_0.gguf`.
- Maintained parity must use the primary GGUF `tokenizer.chat_template` through the explicit tool
  formatter contract; no raw fallback is allowed on the maintained path.
- Stored parity snapshots under `snapshots/parity/` require explicit user approval before they are
  added or refreshed.
- The prior maintained Llama slice stays protected by the existing shared `paritychecker_tests`
  lane; Phase 28 must not fork a Qwen-only acceptance surface.

## Execution Notes

- Compare-mode Qwen attribution initially failed because replay was not consuming the same
  conditioned prompt tokens, renderer output contract, and canonical Qwen3 Q/K RMS norm path as
  the shipped generator.
- Phase 28 closes that alignment gap and then converts the maintained Qwen tests from
  temp-baseline write mode to stored compare mode.
- Benchmark compare remains Phase 29 work and still fails truthfully at `prepare_emel_fixture` on
  the canonical Qwen fixture under the repo's warning-only benchmark policy.
