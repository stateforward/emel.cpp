---
phase: 36-parity-and-regression-proof
plan: 01
subsystem: parity
tags: [liquid, parity, regression, maintained-fixtures]
requires:
  - phase: 35
    provides: maintained Liquid runtime generation on the shipped path
provides:
  - maintained Liquid generation parity against the reference
  - additive regression protection for maintained fixtures
affects: [37, 39]
tech-stack:
  added: []
  patterns:
    - append-only maintained generation baselines
    - additive maintained fixture proof across Qwen and Liquid
key-files:
  created:
    - .planning/phases/36-parity-and-regression-proof/36-01-SUMMARY.md
  modified:
    - snapshots/parity/generation_lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1.txt
    - snapshots/parity/generation_lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_10.txt
    - snapshots/parity/generation_lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_100.txt
    - snapshots/parity/generation_lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1000.txt
    - tools/paritychecker/parity_runner.cpp
    - tools/paritychecker/paritychecker_tests.cpp
    - tools/generation_fixture_registry.hpp
key-decisions:
  - "Keep maintained generation baselines append-only across supported fixtures."
  - "Protect prior maintained anchors while widening to the Liquid slice."
patterns-established:
  - "Maintained parity proof is additive, not replacement-based."
requirements-completed: [PAR-02, VER-02]
duration: reconstructed
completed: 2026-04-02
---

# Phase 36 Plan 01: Parity And Regression Proof Summary

The maintained Liquid parity and regression-proof work shipped on the branch but never received
formal v1.9 closeout artifacts. This summary reconstructs the delivered proof surface.

## Accomplishments

- Added maintained Liquid generation baselines for `1/10/100/1000`.
- Kept the maintained fixture registry additive so Qwen and Liquid remain simultaneously covered.
- Added parity and maintained-generation tests that prove append-only baseline and fixture behavior.

## Evidence

- [snapshots/parity](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/snapshots/parity)
- [parity_runner.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/paritychecker/parity_runner.cpp)
- [paritychecker_tests.cpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/paritychecker/paritychecker_tests.cpp)
- [generation_fixture_registry.hpp](/Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/generation_fixture_registry.hpp)

---
*Phase: 36-parity-and-regression-proof*
*Completed: 2026-04-02*
