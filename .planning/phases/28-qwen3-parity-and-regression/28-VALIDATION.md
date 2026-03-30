---
phase: 28
slug: qwen3-parity-and-regression
status: validated
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-28
---

# Phase 28 — Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | doctest via CMake/CTest plus maintained subprocess proofs |
| Quick run command | `./build/paritychecker_zig/paritychecker_tests --test-case='*qwen3*generation*' --no-breaks` |
| Full suite command | `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests && scripts/quality_gates.sh` |
| Estimated runtime | ~1 minute quick lane, repo gate longer because of coverage/fuzz/docs |

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Automated Command | Status |
|---------|------|------|-------------|-------------------|--------|
| 28-01-01 | 01 | 1 | PAR-01 | `cmake --build build/paritychecker_zig --target paritychecker_tests paritychecker -j4 && ./build/paritychecker_zig/paritychecker_tests --test-case='*qwen3*generation*' --no-breaks` | ✅ green |
| 28-01-02 | 01 | 1 | PAR-01 | `./build/paritychecker_zig/paritychecker --generation --model tests/models/Qwen3-0.6B-Q8_0.gguf --text hello --max-tokens 1 --attribution` | ✅ green |
| 28-02-01 | 02 | 2 | VER-01 | `ctest --test-dir build/paritychecker_zig --output-on-failure -R paritychecker_tests` | ✅ green |
| 28-02-02 | 02 | 2 | VER-01 | `scripts/quality_gates.sh` | ✅ green* |

\* Repo gate completed with the known warning-only benchmark compare failure at
`prepare_emel_fixture` on the canonical Qwen fixture; parity, coverage, fuzz, and docs lanes
completed on the maintained policy path.

## Manual Review

- Confirmed Phase 28 only refreshed approved parity snapshots under `snapshots/parity/`.
- Confirmed operator-facing parity output still publishes the resolved formatter contract and
  canonical stored-baseline source without claiming benchmark completion.

## Validation Sign-Off

- [x] Every task has automated verification
- [x] Shared parity lane protects the prior Llama anchor
- [x] Stored snapshot update stayed within the user-approved parity scope
- [x] `nyquist_compliant: true`
