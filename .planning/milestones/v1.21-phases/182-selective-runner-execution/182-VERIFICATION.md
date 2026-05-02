---
phase: 182
slug: selective-runner-execution
status: passed
verified: 2026-05-02
---

# Phase 182 Verification

## Requirements

- RUNNER-01: satisfied by `scripts/paritychecker.sh --runner=<name>` and quality-gate runner
  argument wiring.
- RUNNER-02: satisfied by the existing `scripts/bench.sh --suite=<runner>` benchmark path that
  remains selected from manifest decisions.
- RUNNER-03: satisfied because selected decisions come from checked-in manifest baselines and
  changed-file inputs, with no hidden state.

## Source Trace

- `scripts/paritychecker.sh` parses runner filters, normalizes `gbnf`, and runs runner-specific
  doctest cases from `paritychecker_tests`.
- `scripts/quality_gates.sh` builds `--runner=<name>` arguments from selected manifest runners.
- `scripts/bench.sh` remains the maintained selected benchmark entrypoint.

## Result

Verification passed for Phase 182.
