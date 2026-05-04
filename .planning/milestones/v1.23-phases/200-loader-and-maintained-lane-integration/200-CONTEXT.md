---
phase: 200-loader-and-maintained-lane-integration
status: complete
requirements:
  - LOAD-01
  - LOAD-02
created: 2026-05-04T01:10:00Z
---

# Phase 200 Context

Phase 200 wires `model/loader` to the public IO boundary while preserving maintained loader and
tool lane ownership. The loader may coordinate tensor and IO actors, but it must not implement
backend-specific byte access, mapping, staging, or loading strategy loops.

Locked decisions:

- `model/loader` can receive an optional IO actor pointer and strategy policy for same-RTC
  orchestration.
- Maintained benchmark, paritychecker, and embedded lanes must not include actor internals from
  IO, tensor, or loader components.
- Existing maintained loading behavior remains unchanged unless an IO strategy is explicitly
  requested.

Canonical refs:

- `src/emel/model/loader/events.hpp`
- `src/emel/model/loader/actions.hpp`
- `src/emel/model/loader/guards.hpp`
- `src/emel/model/loader/sm.hpp`
- `tools/bench/generation_bench.cpp`
- `scripts/check_domain_boundaries.sh`
