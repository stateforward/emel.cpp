---
phase: 197-i-o-module-skeleton-and-ownership-contract
status: complete
requirements:
  - IO-01
  - IO-02
created: 2026-05-04T01:10:00Z
---

# Phase 197 Context

Phase 197 establishes `src/emel/io` as the first-class runtime owner for loading strategy and
transport boundaries. The phase is boundary-only: it must not implement mmap, staged read, copy,
chunked, device-specific, or cooperative async behavior.

Locked decisions:

- I/O owns loading strategy selection boundaries and future transport/staging strategy actors.
- `model/tensor` remains the tensor residency lifecycle owner.
- `model/loader` remains an orchestrator and must not regain backend byte-access logic.
- The new I/O component must follow the canonical Stateforward.SML layout and expose
  `emel::io::loader::sm`, `emel::io::sm`, and an additive top-level alias.

Canonical refs:

- `docs/rules/sml.rules.md`
- `AGENTS.md`
- `src/emel/gbnf`
- `src/emel/model/tensor`
- `src/emel/model/loader`
