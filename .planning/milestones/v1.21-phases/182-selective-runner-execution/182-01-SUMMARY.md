---
phase: 182-selective-runner-execution
plan: 01
subsystem: quality-gates
tags:
  - selective-runners
  - paritychecker
  - benchmarks
duration: same-session
completed: 2026-05-02
requirements-completed:
  - RUNNER-01
  - RUNNER-02
  - RUNNER-03
---

# Phase 182 Summary

The paritychecker script now supports maintained selected-runner execution, and the quality gate can
pass manifest-selected parity runners without bypassing the maintained entrypoint.

## Changes

- Added `--runner=<name>` and `--mode=<name>` support to `scripts/paritychecker.sh`.
- Supported selected parity runners: `tokenizer`, `gbnf_parser`, `kernel`, `jinja`,
  `generation`, and `all`.
- Preserved full parity behavior when no runner filter is supplied.
- Wired `scripts/quality_gates.sh` to pass selected parity runners to `scripts/paritychecker.sh`.
- Kept benchmark selected execution through `scripts/bench.sh --suite=<runner>`.

## Evidence

- `scripts/paritychecker.sh --runner=kernel` passed one kernel parity doctest and skipped unrelated
  parity tests.
- `tools/bench/quality_gates_tests.cpp` covers selected parity runner support and benchmark suite
  selection contracts.
