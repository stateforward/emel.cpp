# Phase 209 Context: Behavior Tests and Scope Guardrails

## Goals
Prove mmap behavior through public SML dispatch and explicitly fail closed on scope or ownership leaks. We must add tests that drive the actor via `process_event(...)` and inspect SML states and public events. We must maintain guardrails to prevent mmap logic from leaking into `model/loader`, `model/tensor`, or other surfaces outside the designated `io/mmap` actor.

## Requirements
- **VAL-01**: Doctest coverage proves supported mmap behavior and representative failure handling through `process_event(...)` and SML state inspection.
- **VAL-02**: Domain and source guardrails fail if mmap implementation leaks into `model/loader`, if tensor residency ownership moves out of `model/tensor`, or if staged read/copy/device/async strategies land in this milestone.

## SML Rules Context
According to `docs/rules/sml.rules.md` and `AGENTS.md`:
- Always use `process_event(...)` and public event interfaces for tests.
- Never reach into actor `actions.hpp`, `detail.hpp`, or `guards.hpp` helpers directly.
- Inspect state via `visit_current_states` or `is(...)`.
- Do not use test-only control fields or backdoors.
- Maintain strong component boundaries. `emel::io::mmap::sm` must own mmap strategy logic.

## Scope
- Doctests in `tests/io/mmap_tests.cpp` or equivalent.
- Guardrails in `scripts/check_domain_boundaries.sh`.
- Run scoped quality gates using `EMEL_QUALITY_GATES_CHANGED_FILES`.