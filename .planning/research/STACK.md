# Stack Research

**Domain:** EMEL C++ inference — I/O staging for constrained host memory
**Researched:** 2026-05-07
**Confidence:** HIGH for project constraints; MEDIUM for target chunking API shape (to be fixed in plan-phase)

## Existing Stack (do not replace)

- Stateforward.SML (`stateforward::sml`) for orchestration; RTC, no-queue invariant.
- Established `src/emel/io` module with shipped `mmap` and `read` strategy actors.
- `model/tensor` + public `emel/io` events for strategy handoff (#60).
- Zig/clang toolchain per AGENTS.md; doctest for tests.

## Stack Additions For v1.26

| Area | Addition | Rationale |
|------|----------|-----------|
| Actor | New `src/emel/io/staged_read` component | Isolates chunked policy from bulk `io/read` |
| Events | Staged request/result events (names TBD in plan-phase) | Typed handoff without context phase fields |
| Tests | `emel_tests_io` or sibling focused targets | Match existing I/O test layout |

## Explicit non-additions

- Cooperative coroutine runtimes, async schedulers, or syscall-from-action patterns that violate RTC rules.
- Duplicating `io/read` bulk copy inside `staged_read` actions without guard-modeled stage boundaries.

## Sources

- `.planning/milestones/v1.25-REQUIREMENTS.md` (deferred STAGED-01 precedent)
- `AGENTS.md`, `docs/rules/sml.rules.md`
- Issue #63 brief (manager message)

---
*Stack research for v1.26 staged read milestone*
