---
phase: 204-mmap-strategy-component-boundary
status: in_progress
requirements:
  - MMAP-01
created: 2026-05-04T15:22:00Z
---

# Phase 204 Context

Phase 204 establishes `src/emel/io/mmap` as the canonical Stateforward.SML mmap strategy actor
under `src/emel/io`. The phase is boundary-only: it must not perform real mmap/munmap calls,
publish a mapped descriptor surface, model file/offset/length validation, or carry tensor
residency. Concrete validation, descriptors, lifetime, errors, tensor integration, public runtime
exposure, and tests are owned by Phases 205-209.

Locked decisions:

- `src/emel/io/mmap` is the canonical mmap-only Stateforward.SML component. Other strategies
  (staged read, copy, device, cooperative async) remain out of scope for v1.24.
- The component follows the canonical SML layout with component-local `context`, `events`,
  `errors`, `guards`, `actions`, `detail`, and `sm` files and exposes `emel::io::mmap::sm`.
- Tensor residency lifecycle ownership remains with `model/tensor`; the mmap component MUST NOT
  store dispatch-local request data or tensor-owned residency metadata in context.
- `model/loader` remains orchestration-only; mmap selection is reached through `emel/io` events
  in later phases, not by loader internals.
- This boundary phase does NOT introduce any platform-specific mmap call, descriptor publication,
  or tool-only mmap scaffold; those land in 205/206/208.

Canonical refs:

- `docs/rules/sml.rules.md`
- `AGENTS.md`
- `src/emel/gbnf`
- `src/emel/io/loader`
- `src/emel/model/tensor`
- `.planning/ROADMAP.md` (v1.24 active)
- `.planning/REQUIREMENTS.md` (v1.24)
