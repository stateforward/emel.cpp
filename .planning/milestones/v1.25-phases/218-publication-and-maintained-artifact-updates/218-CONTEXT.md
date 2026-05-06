---
phase: 218-publication-and-maintained-artifact-updates
status: complete
created: 2026-05-05T18:53:13Z
requirements:
  - VAL-03
depends_on:
  - 217-behavior-tests-and-scope-guardrails
---

# Phase 218 Context

## Goal

Close the v1.25 publication gap by making public docs, generated architecture docs,
maintained snapshots, benchmark evidence, and planning truth match the implemented
read/copy runtime path.

## Source-Backed Truth

- `src/emel/io/read` is the read/copy actor. It copies from an event-provided source span
  into a caller-owned target buffer and does not own tensor residency.
- `src/emel/io/loader` dispatches `read_copy` to an injected `io/read` actor when present,
  and reports unsupported strategy otherwise.
- `src/emel/model/tensor` owns read-backed residency commit through `request_read_load`.
- `src/emel/model/loader` and maintained tool lanes select/report read/copy through
  public model-loader and I/O surfaces.
- Mmap remains a separate v1.24 strategy under `src/emel/io/mmap`; staged/chunked, async,
  and device-specific strategies remain deferred.

## Publication Risk

Stale docs or planning artifacts must not claim read/copy is still follow-on work, and must
not imply staged/chunked constrained-memory support shipped in v1.25.
