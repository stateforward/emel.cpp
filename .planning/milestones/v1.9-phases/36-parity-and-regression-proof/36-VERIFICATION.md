---
phase: 36-parity-and-regression-proof
verified: 2026-04-02T17:12:32Z
status: passed
score: 2/2 phase truths verified
---

# Phase 36 Verification Report

**Phase Goal:** Prove the maintained Liquid slice against the reference and protect prior
maintained anchors.  
**Verified:** 2026-04-02T17:12:32Z  
**Status:** passed

## Goal Achievement

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Maintained Liquid generation parity is stored for the official fixture and maintained conditioning contract. | ✓ VERIFIED | [snapshots/parity](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/snapshots/parity) contains maintained Liquid `1/10/100/1000` baselines for `generation_lfm2_5_1_2b_thinking_q4_k_m_*`. |
| 2 | Regression protection remains additive across maintained fixtures instead of dropping prior anchors. | ✓ VERIFIED | [generation_fixture_registry.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/generation_fixture_registry.hpp) keeps both Qwen and Liquid fixtures, and [paritychecker_tests.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/paritychecker/paritychecker_tests.cpp) includes maintained-generation regression coverage. |

## Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| PAR-02 | ✓ SATISFIED | - |
| VER-02 | ✓ SATISFIED | - |

## Automated Checks

- `ls snapshots/parity | rg 'generation_lfm2_5_1_2b_thinking_q4_k_m'`
- `rg -n "maintained generation|supported fixtures|append-only" tools/paritychecker/paritychecker_tests.cpp tools/generation_fixture_registry.hpp`

## Verification Notes

- This verification is reconstructed from shipped parity artifacts and maintained-fixture tests.
- The phase closes the parity and regression-proof surface only; benchmark publication remains
  Phase 37.

---
*Verified: 2026-04-02T17:12:32Z*
*Verifier: the agent*
