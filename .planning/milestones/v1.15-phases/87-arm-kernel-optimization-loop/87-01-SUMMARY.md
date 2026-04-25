---
requirements-completed: []
---

# Phase 87 Summary: ARM Kernel Optimization Loop

## Completed

- Optimized the Sortformer-local dense f32 helper used by the measured transformer, modules/cache,
  encoder, and output stages.
- Preserved allocation-free behavior and kept the change in the existing Sortformer-local helper
  without adding tool-only compute, fallback paths, or reference-lane dependencies.
- Recorded before/after benchmark evidence for the maintained `diarization_sortformer` profile.

## Evidence

- Phase 86 baseline sample:
  - `end_to_end_ns=2871626`
  - `transformer_ns=2554333`
- Phase 87 post-optimization sample:
  - `end_to_end_ns=2652040`
  - `transformer_ns=2398083`
  - Output checksum remained `15712531076325547939`.

## Stop Condition

After the dense helper pass, the transformer stage remains the dominant bounded profile cost.
Another material improvement would require a broader kernel-owned dense/matmul contract or
stage-level transformer restructuring. That is not a low-risk incremental loop item inside this
phase, so the optimization loop stops with one validated material pass and records the remaining
candidate for the closeout audit.
