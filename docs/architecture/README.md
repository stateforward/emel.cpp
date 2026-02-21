# architecture decisions

this directory stores generated architecture artifacts (markdown + mermaid) derived from `src/`.
the source of truth for orchestration architecture is the boost.SML code in `src/`.
WARNING: this architecture documentation is under active development and may change frequently.

## why these patterns

- determinism: all control flow is represented in boost.SML transitions, not ad-hoc branching.
- explicit failure semantics: each step emits `_done` or `_error` events; errors are never implicit.
- stable ownership boundaries: machines communicate only via `process_event(...)`.
- C-ABI safety: public API remains C-compatible while internal orchestration stays type-safe C++.
- performance discipline: machine `data` holds runtime state only; avoid duplicated state and hidden coupling.

## core conventions

- trigger events carry per-invocation external inputs.
  - examples: target model pointer, capability/policy flags, owner machine pointer.
- machine `data` holds machine-owned runtime state.
  - examples: progress counters, selected mode, status fields.
- do not store external service dependencies in persistent machine `data` unless required by lifecycle.
- use explicit terminal dispatch.
  - `done`/`errored` entry actions dispatch terminal events to owner via `owner_sm->process_event(...)`.
- prefer explicit event transitions over choice pseudostates for success/error routing.

## practical split

- event payload: "inputs for this run"
- machine data: "state produced/owned by this machine during this run"

this split keeps long-lived and RAII usage consistent, reduces duplicated dependency state,
and makes event traces sufficient for debugging/telemetry.

## relationship to reference implementations

ported behavior is preserved while restructuring control flow into explicit state-machine
contracts. transport, parse, and load decisions are modeled as events/states, not hidden function
flow.

## what to update together

when changing a machine structure, update all of:

- the corresponding boost.SML machine files under `src/`
- related parent/child machine code that consumes its events
- generated markdown under `docs/architecture/*.md`
- generated mermaid diagrams under `docs/architecture/mermaid/*.mmd`
- `AGENTS.md` rules if a new cross-machine pattern is introduced
