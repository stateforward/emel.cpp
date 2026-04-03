---
phase: 41-parity-and-regression-proof
verified: 2026-04-03T03:34:14Z
status: passed
score: 3/3 phase truths verified
---

# Phase 41 Verification Report

**Phase Goal:** Prove the maintained Bonsai slice against the pinned Prism reference lane while
keeping prior maintained anchors on truthful isolated regression lanes.
**Verified:** 2026-04-03T03:34:14Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Maintained generation fixtures now publish explicit per-model reference identity instead of relying on one global repo/ref pair. | ✓ VERIFIED | [generation_fixture_registry.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/tools/generation_fixture_registry.hpp) now records `reference_repository`, optional `reference_ref`, and per-fixture `generation_parity_contract`, including the pinned Prism tuple for `Bonsai-1.7B.gguf`. |
| 2 | The maintained parity workflow now builds and tests isolated reference lanes, so Bonsai proves against Prism without contaminating Qwen or Liquid. | ✓ VERIFIED | [scripts/paritychecker.sh](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/scripts/paritychecker.sh) now resolves fixture records into distinct lane builds and the full scripted parity sweep completed green across all configured lanes. |
| 3 | Repo-visible parity evidence now keeps Bonsai, Qwen, and Liquid on their truthful contracts while making the exact formatter and reference identity auditable. | ✓ VERIFIED | [parity_runner.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/tools/paritychecker/parity_runner.cpp) and [paritychecker_tests.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/tools/paritychecker/paritychecker_tests.cpp) now separate append-only baseline fixtures from live-reference fixtures and assert publication of formatter plus `repo=` and `ref=` metadata. |

**Score:** 3/3 phase truths verified

## Automated Checks

- `./scripts/paritychecker.sh`
- `./scripts/paritychecker.sh --implementation --model tests/models/Bonsai-1.7B.gguf --text hello --max-tokens 1`

## Current Results

- Passed:
  - `./scripts/paritychecker.sh`
  - `./scripts/paritychecker.sh --implementation --model tests/models/Bonsai-1.7B.gguf --text hello --max-tokens 1`

## Verification Notes

- The direct monolithic `build/paritychecker_zig/paritychecker` surface is no longer the truthful
  maintained verification contract for this phase, because Phase 41 intentionally split parity into
  isolated per-model reference lanes. The authoritative maintained proof surface is now
  [paritychecker.sh](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/bonsai/scripts/paritychecker.sh).
- The focused Bonsai implementation probe passed on the pinned Prism lane and printed the exact
  maintained formatter contract plus `repo=https://github.com/PrismML-Eng/llama.cpp.git` and
  `ref=f5dda7207ed5837f1c83c2f52f851ad9b933d2fd`.
- The full parity sweep completed green across the isolated `ggml-org/llama.cpp` and
  `PrismML-Eng/llama.cpp` lanes, which is the regression surface that protects Qwen, Liquid, and
  Bonsai simultaneously without cross-lane sharing.
