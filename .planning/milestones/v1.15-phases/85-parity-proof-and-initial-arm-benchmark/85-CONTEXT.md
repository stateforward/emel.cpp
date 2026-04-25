# Phase 85: Parity Proof And Initial ARM Benchmark - Context

**Gathered:** 2026-04-23
**Status:** Ready for planning
**Mode:** Auto-generated from ROADMAP.md under gsd-autonomous

<domain>
## Phase Boundary

Establish lane-isolated proof and an initial maintained ARM benchmark before optimization work
begins.

## Requirements

- PRF-01
- PRF-02
- BEN-01
- DOC-01
</domain>

<decisions>
## Implementation Decisions

- Add a repository-level deterministic Sortformer diarization parity proof for one canonical
  multi-speaker fixture.
- Keep the reference side as a fixed trusted baseline in test/benchmark code; do not call EMEL from
  the reference lane and do not introduce llama.cpp/ggml dependencies outside the existing bench
  reference tooling boundary.
- Add a maintained benchmark suite that reports fixture identity, fixed profile parameters, timing,
  and proof status without updating benchmark snapshots.
- Document the currently supported Sortformer model slice, input contract, output contract, and
  known pre-optimization limitations.
</decisions>

<constraints>
## Constraints

- No benchmark or parity snapshots may be updated without explicit user consent.
- Reference and EMEL lanes must stay visibly separated.
- New runtime work must remain in EMEL-owned code paths and avoid hot-path allocation.
</constraints>
