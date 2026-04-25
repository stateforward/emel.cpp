# Phase 86: ARM Sortformer Profiling Baseline - Context

**Gathered:** 2026-04-23
**Status:** Ready for planning
**Mode:** Auto-generated from ROADMAP.md under gsd-autonomous

<domain>
## Phase Boundary

Build an ARM profiling baseline for the maintained Sortformer path with stable end-to-end and
per-stage measurements.

## Requirements

- BEN-01
- RUN-03
</domain>

<decisions>
## Implementation Decisions

- Extend the maintained `diarization_sortformer` benchmark with measurement-only stage attribution.
- Attribute stage timings to existing EMEL-owned functions for feature extraction, encoder work,
  modules/cache work, transformer work, and output conversion.
- Report ownership guidance in benchmark metadata so optimization work can decide whether hotspots
  belong in AArch64 kernels, shared kernels, or Sortformer-local helpers.
- Keep all production compute in `src/`; benchmark tooling only prepares fixed inputs, times calls,
  and reports measurements.
</decisions>

<constraints>
## Constraints

- Do not introduce tool-only optimized compute paths.
- Do not update benchmark snapshots without explicit approval.
- Keep profiling stable enough for ARM baseline comparison without making the full quality gate
  impractically slow.
</constraints>
