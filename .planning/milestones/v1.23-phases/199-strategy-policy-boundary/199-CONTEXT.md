---
phase: 199-strategy-policy-boundary
status: complete
requirements:
  - TBOUND-03
  - POLICY-01
  - POLICY-02
  - POLICY-03
created: 2026-05-04T01:10:00Z
---

# Phase 199 Context

Phase 199 defines strategy policy injection and deterministic rejection behavior. The milestone is
strategy-framework-only, so concrete strategies must fail closed while leaving future mmap and
staged read strategy actors room to land later.

Locked decisions:

- Runtime strategy choice belongs in `sm.hpp` guards and transition rows.
- Actions may execute an already-selected path but must not branch on strategy kind.
- No queues, defer queues, mailboxes, sleeps, async scheduling, or post-for-later behavior are
  allowed.
- Failed IO strategy attempts must preserve precise error classification.

Canonical refs:

- `src/emel/io/loader/guards.hpp`
- `src/emel/io/loader/sm.hpp`
- `src/emel/model/tensor/guards.hpp`
- `src/emel/model/tensor/sm.hpp`
- `src/emel/model/loader/guards.hpp`
- `src/emel/model/loader/sm.hpp`
