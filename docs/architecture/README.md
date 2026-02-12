# Architecture Decisions

This directory stores generated architecture artifacts (for example PUML) derived from `src/`.
The source of truth for orchestration architecture is the Boost.SML code in `src/`.

## Why These Patterns

- Determinism: all control flow is represented in Boost.SML transitions, not ad-hoc branching.
- Explicit failure semantics: each step emits `_done` or `_error` events; errors are never implicit.
- Stable ownership boundaries: machines communicate only via `process_event(...)`.
- C-ABI safety: public API remains C-compatible while internal orchestration stays type-safe C++.
- Performance discipline: machine `data` holds runtime state only; avoid duplicated state and hidden coupling.

## Core Conventions

- Trigger events carry per-invocation external inputs.
  - Examples: target model pointer, capability/policy flags, owner machine pointer.
- Machine `data` holds machine-owned runtime state.
  - Examples: progress counters, selected mode, status fields.
- Do not store external service dependencies in persistent machine `data` unless required by lifecycle.
- Use explicit terminal dispatch.
  - `done`/`errored` entry actions dispatch terminal events to owner via `owner_sm->process_event(...)`.
- Prefer explicit event transitions over choice pseudostates for success/error routing.

## Practical Split

- Event payload: "inputs for this run"
- Machine data: "state produced/owned by this machine during this run"

This split keeps long-lived and RAII usage consistent, reduces duplicated dependency state,
and makes event traces sufficient for debugging/telemetry.

## Relationship To `tmp/llama.cpp`

Ported behavior is preserved while restructuring control flow into explicit state-machine contracts.
Transport, parse, and load decisions are modeled as events/states, not hidden function flow.

## What To Update Together

When changing a machine structure, update all of:

- the corresponding Boost.SML machine files under `src/`
- related parent/child machine code that consumes its events
- generated PUML outputs under `docs/architecture/puml/`
- `AGENTS.md` rules if a new cross-machine pattern is introduced
