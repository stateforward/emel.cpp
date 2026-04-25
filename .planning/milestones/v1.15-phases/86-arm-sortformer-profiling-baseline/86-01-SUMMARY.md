---
requirements-completed: []
---

# Phase 86 Summary: ARM Sortformer Profiling Baseline

## Completed

- Extended the maintained `diarization_sortformer` benchmark suite with
  `diarization/sortformer/stage_profile_8f_profile_188x192x4`.
- Added measurement-only stage attribution metadata for feature extraction, encoder work,
  modules/cache work, transformer work, output conversion, and end-to-end timing.
- Reported hotspot owner hints in benchmark metadata so follow-up optimization can route work to
  AArch64 kernels, shared kernel surfaces, or Sortformer-local helpers.
- Updated generated benchmark documentation through `docs/templates/benchmarks.md.j2`.

## Notes

- The profiling row is intentionally measurement-only and uses existing EMEL-owned stage functions.
- Benchmark snapshots were not updated; the new profiling row remains unsnapshotted pending explicit
  approval.
