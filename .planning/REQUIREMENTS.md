# Requirements: EMEL

**Defined:** 2026-03-22
**Core Value:** Prove real end-to-end behavior with explicit SML orchestration and
parity-oriented verification before widening API surface or model scope.

## v1 Requirements

### Vectorized Quantized Kernels

- [ ] **PORT-04**: The canonical Llama-68M ARM generation slice can execute the maintained
  `q2_K x q8_K` dot-product hot path through an EMEL-owned vectorized AArch64 kernel instead of
  the current scalar row helper.
- [ ] **PORT-05**: The canonical Llama-68M ARM generation slice can execute the maintained
  `q3_K x q8_K` dot-product hot path through an EMEL-owned vectorized AArch64 kernel instead of
  the current scalar row helper.
- [ ] **PORT-06**: The canonical Llama-68M ARM generation slice can execute the maintained
  `q6_K x q8_K` dot-product hot path through an EMEL-owned vectorized AArch64 kernel instead of
  the current scalar row helper.
- [ ] **PORT-07**: The vectorized quantized kernels preserve zero-allocation hot-path behavior,
  consume the same effective operand class as the current shipped path, and do not replace the
  hot loop with dequantize-to-f32 fallbacks.

### Architecture Contract

- [ ] **ARCH-02**: The vectorized quantized kernel rollout remains a data-plane replacement
  inside the existing generator -> graph -> processor -> kernel chain and does not introduce
  queue-based orchestration, public API widening, or actor-structure rewrites.

### Verification

- [ ] **PAR-04**: `tools/paritychecker --generation` keeps the maintained `1/10/100/1000` token
  checks and proves the canonical ARM workload exercised the vectorized quantized path.
- [ ] **VER-03**: Kernel and regression tests cover `q2_K`, `q3_K`, and `q6_K` vectorized
  correctness against the existing scalar path plus deterministic behavior when a request is
  unsupported or forced off the optimized path.

### Benchmarking

- [ ] **BENCH-08**: `tools/bench` runs the maintained canonical ARM compare workload through the
  vectorized quantized path and publishes enough attribution to distinguish it from the current
  scalar row-helper path.
- [ ] **BENCH-09**: Maintained compare output republishes `1`, `10`, `100`, and `1000` token
  results and shows measurable end-to-end improvement over the current v1.3 baseline on at least
  one maintained generation length.

## v2 Requirements

### Generator Math

- `GEN-03`: Optimize ARM generator-side RMSNorm, RoPE, residual-add, and SwiGLU math after the
  vectorized quantized kernel gain is measured.

### Broader Coverage

- `FLASH-03`: Broaden flash attention beyond the canonical Llama-68M shape and workload
  contract.
- `MODEL-01`: Roll optimized ARM flash attention out to additional model fixtures after the
  canonical path remains correct and benchmarked.

### Benchmark Policy

- `BENCH-07`: Revisit whether noisy benchmark drift should become a blocking repo gate once ARM
  compare evidence stabilizes.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Generator-side ARM math outside the quantized row-dot hot path | Keep the milestone fixed on the dominant profiled bottleneck before widening optimization scope. |
| Non-ARM backend quantized kernel specialization | The canonical ARM Llama-68M slice remains the maintained truth anchor for this milestone. |
| Broader model or workload rollout | New fixtures and wider workload claims should wait until the canonical vectorized path is proven. |
| Dequantize-to-f32 or tool-only compute fallbacks in the shipped hot path | The milestone goal is native EMEL-owned vectorized quantized kernels. |
| State-machine or orchestration rewrites | Current profiling points at scalar data-plane cost, not the actor model. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PORT-04 | Phase 17 | Pending |
| PORT-05 | Phase 18 | Pending |
| PORT-06 | Phase 19 | Pending |
| PORT-07 | Phase 19 | Pending |
| ARCH-02 | Phase 20 | Pending |
| PAR-04 | Phase 20 | Pending |
| VER-03 | Phase 20 | Pending |
| BENCH-08 | Phase 21 | Pending |
| BENCH-09 | Phase 21 | Pending |

**Coverage:**
- v1 requirements: 9 total
- Mapped to phases: 9
- Unmapped: 0
- Coverage status: 100% mapped ✓

---
*Requirements defined: 2026-03-22*
*Last updated: 2026-03-22 after roadmap creation*
