# Phase 87: ARM Kernel Optimization Loop - Context

**Gathered:** 2026-04-23
**Status:** Ready for planning
**Mode:** Auto-generated from ROADMAP.md under gsd-autonomous

<domain>
## Phase Boundary

Iterate on the measured ARM hotspots until additional optimizations are no longer material for the
maintained Sortformer profile.

## Requirements

- RUN-03
- BEN-01
</domain>

<decisions>
## Phase 86 Evidence

The suite-filtered Phase 86 benchmark identified the transformer stage as the dominant bounded
stage-profile hotspot:

- `transformer_ns`: about 2.55 ms
- `feature_ns`: about 0.24 ms
- `modules_cache_ns`: about 0.08 ms
- `output_ns`: about 0.001 ms

## Implementation Decision

- Start with the shared Sortformer dense f32 helper used by transformer, modules, encoder, and
  output stages.
- Keep the work under `src/emel/diarization/sortformer/detail.cpp` because the current helper is
  Sortformer-local and does not expose a general kernel tensor contract yet.
- Do not add tool-only compute, reference-lane dependency, or fallback paths.
- Stop after one validated optimization pass if the remaining profile does not expose a clear,
  low-risk, material follow-up candidate.
</decisions>

<constraints>
## Constraints

- Benchmark snapshots remain unchanged without explicit approval.
- Hot-path changes must stay allocation-free.
- Numeric behavior must stay equivalent to existing tests and the maintained benchmark checksum.
</constraints>
