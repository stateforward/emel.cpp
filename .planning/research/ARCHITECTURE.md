# Architecture Research

**Domain:** `emel/io` staged read integration
**Researched:** 2026-05-07
**Confidence:** HIGH

## Major components

1. **`io/staged_read` actor** — Owns staged read state machine, validation guards, and stage execution effects; mirrors component layout of `io/read` and `io/mmap`.
2. **`model/tensor` orchestration** — Injects or selects staged strategy alongside existing mmap/read paths using explicit guards (no action-side strategy pick).
3. **`io/loader` or existing loader surfaces** — Continues to report/select strategies through public contracts only; no reach-through to actor `detail`.

## Data flow (intent)

- Request describes overall file span and staging constraints (max chunk size / window — exact fields to be decided in plan-phase).
- Each stage copies a sub-span into the tensor-owned target region for that window.
- Success commits residency only at tensor layer after final stage completes (pattern to align with existing read success semantics).

## Build order (for roadmap)

1. Component boundary + fail-closed scaffold
2. Validation / platform / chunk policy gates
3. Stage execution + errors + lifetime
4. Tensor integration + public reporting
5. Tests, guardrails, publication

## Pitfalls cross-links

- Do not use completion-driven loops as data-plane iteration per SML rules; bulk work stays in bounded kernels within a transition where applicable.
- Do not store dispatch-local request pointers or phase indices in `staged_read` context.

## Sources

- `AGENTS.md` architecture and composition section
- v1.25/v1.24 milestone archives for I/O patterns

---
*Architecture research for v1.26*
