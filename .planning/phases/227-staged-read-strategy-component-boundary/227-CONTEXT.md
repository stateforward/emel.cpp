# Phase 227: Staged Read Strategy Component Boundary - Context

**Gathered:** 2026-05-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Establish `src/emel/io/staged_read/` as the canonical packaged Stateforward.SML
strategy actor mirroring `io/read` and `io/mmap` ownership (events, guards,
actions, errors, empty persistent `context`, destination-first transitions). Ship a
fail-closed or smoke dispatch so the actor is reachable before deeper staged
policy lands in later phases.

</domain>

<decisions>
## Implementation Decisions

### the agent's Discretion

All structural choices inside this scaffolding phase are at implementation
discretion provided STG-01 (`emel::io::staged_read::sm`) remains the canonical
alias, AGENTS.md / `docs/rules/sml.rules.md` are respected, mmap/device/async are
not introduced here, and no cooperative coroutine scheduling is wired.

### Locked constraints

- Match existing `src/emel/io/read/**` layout and `include`/`machines.hpp` alias
  patterns used for other I/O strategies.
- Do not move tensor residency ownership; strategy remains copy/stage semantics only.
- Prefer header-only actions like `io/read` unless a `.cpp` lane is already
  required for linkage (follow repo precedent).

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets

- `src/emel/io/read/**` — template for rtc-safe internal carriers, guards, empty
  `context`, public `event` vs `detail::*_runtime`.
- `src/emel/io/mmap/**` — reference for syscall-backed strategy structure (staging
  will differ but file layout analogous).
- `tests/io/read/lifecycle_tests.cpp` — pattern for doctest machine alias and `state_ready`.

### Established Patterns

- I/O actors own strategy only; filesystem work stays outside synchronous dispatch per
  established read-path discipline.
- `emel_core` includes `src/`; new headers under `src/emel/io/staged_read/`.
- CMake: `mmap` owns `actions.cpp`; `read` remains header-heavy — staged_read should
  start header-only unless a concrete symbol forces `.cpp`.

### Integration Points

- Future phases wire `model/tensor` and `io/loader`; Phase 227 may only expose
  `emel::io::staged_read::sm` and optional top-level alias in `machines.hpp`,
  parallel to `IoRead`.

</code_context>

<specifics>
## Specific Ideas

No specific UI or product behavior — infrastructure-only boundary scaffold.

</specifics>

<canonical_refs>
## Canonical References

- `docs/rules/sml.rules.md` — RTC, no-queue, guard vs action branching rules.
- `AGENTS.md` — orchestration conventions, canonical file bases, naming.
- `.planning/REQUIREMENTS.md` — STG-01 obligation for Phase 227.
- `.planning/research/SUMMARY.md` — phased roadmap expectations for staged read.

</canonical_refs>

<deferred>
## Deferred Ideas

- Stage sizing, tensor public events, loader reporting, guardrails, and publication
  evidence belong to Phases 228–236 per ROADMAP.

</deferred>

---
*Phase: 227-staged-read-strategy-component-boundary*
