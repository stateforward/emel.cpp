---
phase: 178-v1-20-closeout-gate-and-evidence-repair
plan: 01
completed: 2026-05-02
commit: 794b52f
status: superseded
superseded_by: 179-v1-20-closeout-evidence-reproducibility-repair
requirements-addressed:
  - VAL-03
---

# Phase 178 Plan 01 Summary

Repaired the v1.20 closeout gate by expanding full benchmark scope through manifest suites instead
of the monolithic full runner, filtering host-specific and internal-only benchmark suites, and
running benchmarks before coverage/parity/fuzz/docs to reduce full-gate performance bias.

Fixed the memory benchmark build filter for reference-dependent suites, added focused bench tooling
tests for the gate behavior, and unrolled the AArch64 SQR kernel path that had regressed under the
closeout benchmark lane. With explicit user approval, updated benchmark snapshots for the remaining
full-gate measurements.

Final verification passed:

- `EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_COVERAGE_CLEAN=1 scripts/quality_gates.sh`

This artifact is historical after the later source-backed audit found the bench tooling validation
path was not reproducible from the current maintained build state. Phase 179 owns the
authoritative VAL-01 and VAL-03 closeout evidence.
