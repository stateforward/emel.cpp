# Requirements: EMEL

**Status:** No active milestone requirements
**Last shipped milestone:** `v1.3 ARM Flash Optimizations`

The shipped v1.3 requirement set is archived in
`.planning/milestones/v1.3-REQUIREMENTS.md`.

Start the next requirements cycle with `$gsd-new-milestone`.

## Carry-Forward Candidates

- `GEN-03`: Optimize ARM generator-side RMSNorm, RoPE, residual-add, and SwiGLU math after the
  standalone AArch64 flash gain is measured.
- `FLASH-03`: Broaden flash attention beyond the canonical Llama-68M shape and workload
  contract.
- `MODEL-01`: Roll optimized ARM flash attention out to additional model fixtures after the
  canonical path remains correct and benchmarked.
- `BENCH-07`: Revisit whether noisy benchmark drift should become a blocking repo gate once ARM
  compare evidence stabilizes.

---
*Last updated: 2026-03-22 after shipping milestone v1.3*
