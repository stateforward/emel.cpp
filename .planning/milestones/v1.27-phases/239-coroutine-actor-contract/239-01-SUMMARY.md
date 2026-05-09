---
phase: 239
plan: 01
status: complete
requirements-completed:
  - CO-01
---

# Phase 239 Summary

## Completed

- Added the coroutine actor definition and `co_sm` rules to `docs/rules/sml.rules.md`.
- Mirrored operational `co_sm` guidance into `AGENTS.md` and `CLAUDE.md`.
- Preserved existing synchronous RTC/no-queue actor guidance while adding an explicit opt-in
  coroutine actor contract.
- Ran focused source checks and changed-file scoped quality gates.

## Verification

Phase 239 verification passed. Runtime `co_sm` implementation is intentionally deferred to Phase
240.
