---
phase: 06-fixture-contract-hardening
verified: 2026-03-08T23:18:00Z
status: passed
score: 4/4 must-haves verified
---

# Phase 6 Verification Report

**Phase Goal:** Close the `HARN-02` audit gap by making the first Llama-68M generation slice accept
only one canonical fixture path and by aligning the CLI/help surface with that hardened contract.
**Verified:** 2026-03-08T23:18:00Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Generation mode now rejects a same-basename GGUF outside the canonical `tests/models/` path. | ✓ VERIFIED | [parity_runner.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.cpp) compares normalized canonical fixture paths, and [paritychecker_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/paritychecker_tests.cpp) exercises the temp-copy impostor failure through the subprocess surface. |
| 2 | The canonical `tests/models/Llama-68M-Chat-v1-Q2_K.gguf` path still succeeds through the real bounded generation parity flow. | ✓ VERIFIED | `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1` passed after the fixture hardening and still reported `generation parity ok`. |
| 3 | The CLI/help surface now truthfully describes the canonical fixture contract and no longer calls generation a reserved path. | ✓ VERIFIED | [parity_main.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_main.cpp) names the exact fixture path, and the new help subprocess doctest asserts that the old reserved-contract wording is absent. |
| 4 | The normal parity gate chain carries the hardened contract without adding a new repo gate. | ✓ VERIFIED | `scripts/paritychecker.sh` and `scripts/quality_gates.sh` both passed after the runtime and help-text changes, so the tightened contract is covered by the standard generation regression path. |

## Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| HARN-02: User can run the first slice against `tests/models/Llama-68M-Chat-v1-Q2_K.gguf` without ambiguous fixture selection. | ✓ SATISFIED | - |

## Automated Checks

- `cmake --build build/paritychecker_zig --target paritychecker_tests paritychecker`
- `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests`
- `build/paritychecker_zig/paritychecker --help`
- `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1`
- `scripts/paritychecker.sh`
- `scripts/quality_gates.sh`

## Verification Notes

- The new help-surface regression intentionally expects exit code `2` because the paritychecker CLI treats `--help` as an argument-parse usage exit rather than a successful normal-mode run.
- `scripts/quality_gates.sh` again reported benchmark snapshot regressions but still passed because the wrapper currently treats snapshot drift as non-blocking repo policy.
- No `src/` actor changes were required; the entire `HARN-02` fix stayed within the paritychecker boundary plus planning-state closeout.
