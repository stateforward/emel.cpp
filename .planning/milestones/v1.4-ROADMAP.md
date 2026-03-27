# Roadmap

## Archived Milestones

- [x] [v1.0: EMEL Llama-68M Generation Slice](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.0-ROADMAP.md) — shipped 2026-03-08 with 7 phases and 15 plans; proved one canonical Llama-68M generation parity slice in `tools/paritychecker/`.
- [x] [v1.1: EMEL Llama-68M Generation Benchmark](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.1-ROADMAP.md) — shipped 2026-03-11 with 4 phases and 10 plans; added one truthful canonical Llama-68M generation benchmark in `tools/bench`, native EMEL decode benchmarking, compare output, and snapshot/docs integration.
- [x] [v1.2: Flash Attention](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.2-ROADMAP.md) — shipped 2026-03-22 with 5 phases and 13 plans; added an EMEL-owned flash-attention path to the canonical Llama-68M slice, hard-cut runtime tensor lifecycle through `emel::tensor::sm`, and published maintained benchmark evidence over a preserved pre-flash baseline.
- [x] [v1.3: ARM Flash Optimizations](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.3-ROADMAP.md) — shipped 2026-03-22 with 3 phases and 7 plans; delivered optimized AArch64 flash execution, maintained runtime/parity attribution, and preserved-baseline benchmark publication for the canonical ARM Llama-68M slice.

## Current Milestone

### v1.4 Full Vectorized Quantized Kernels

**Milestone Goal:** Close the remaining canonical ARM inference gap by replacing the shipped
scalar `q2_K/q3_K/q6_K x q8_K` hot path with EMEL-owned vectorized AArch64 kernels while
preserving the existing Boost.SML orchestration contract and maintained `1/10/100/1000` parity
and benchmark surfaces.

**Scope Guardrails:**
- Keep the milestone narrow to the canonical CPU-hosted Llama-68M ARM slice.
- Keep `tools/paritychecker` and `tools/bench` as the only acceptance surfaces.
- Preserve the existing generator -> graph -> processor -> kernel chain and current actor
  structure.
- Do not accept dequantize-to-f32 or tool-only compute fallbacks in the shipped hot path.

## Phases

- [x] **Phase 17: Vectorized q2_K Kernel** - Replace the maintained `q2_K x q8_K` scalar row (completed 2026-03-23)
  helper with an EMEL-owned vectorized AArch64 kernel.
- [x] **Phase 18: Vectorized q3_K Kernel** - Replace the maintained `q3_K x q8_K` scalar row (completed 2026-03-23)
  helper with an EMEL-owned vectorized AArch64 kernel.
- [x] **Phase 19: Vectorized q6_K Kernel And Hot-Path Contract** - Finish the `q6_K x q8_K` (completed 2026-03-23)
  cutover and lock zero-allocation operand fidelity across the maintained quantized path.
- [x] **Phase 20: Runtime Integration And Proof** - Adopt the full vectorized kernel set in the (completed 2026-03-23)
  shipped chain and prove supported plus fallback behavior without changing actor structure.
- [x] **Phase 21: Benchmark Attribution And Impact** - Publish maintained ARM compare evidence (completed 2026-03-23)
  that distinguishes the vectorized quantized path from the v1.3 scalar baseline.

## Phase Details

### Phase 17: Vectorized q2_K Kernel
**Goal**: Replace the maintained `q2_K x q8_K` scalar row helper with an EMEL-owned vectorized
AArch64 kernel on the canonical ARM workload.
**Depends on**: Phase 16
**Requirements**: PORT-04
**Success Criteria** (what must be TRUE):
  1. The canonical Llama-68M ARM workload can execute maintained `q2_K x q8_K` hot-path dot
     products through an EMEL-owned vectorized AArch64 kernel instead of the current scalar row
     helper.
  2. Maintained proof at the kernel seam can distinguish supported vectorized `q2_K` execution
     from the prior scalar helper on the canonical operand path.
**Plans**: 2 plans

Plans:
- [ ] 17-01: Land the maintained vectorized `q2_K x q8_K` AArch64 kernel through the existing
  quantized backend seam.
- [ ] 17-02: Prove vectorized `q2_K` path selection on the canonical operand path without
  widening the acceptance surface.

### Phase 18: Vectorized q3_K Kernel
**Goal**: Replace the maintained `q3_K x q8_K` scalar row helper with an EMEL-owned vectorized
AArch64 kernel on the canonical ARM workload.
**Depends on**: Phase 17
**Requirements**: PORT-05
**Success Criteria** (what must be TRUE):
  1. The canonical Llama-68M ARM workload can execute maintained `q3_K x q8_K` hot-path dot
     products through an EMEL-owned vectorized AArch64 kernel instead of the current scalar row
     helper.
  2. Maintained proof at the kernel seam can distinguish supported vectorized `q3_K` execution
     from the prior scalar helper on the canonical operand path.
**Plans**: 2 plans

Plans:
- [ ] 18-01: Land the maintained vectorized `q3_K x q8_K` AArch64 kernel through the existing
  quantized backend seam.
- [ ] 18-02: Prove vectorized `q3_K` path selection on the canonical operand path without
  widening the acceptance surface.

### Phase 19: Vectorized q6_K Kernel And Hot-Path Contract
**Goal**: Complete the `q6_K x q8_K` cutover and lock the maintained quantized hot path to the
same effective operand class with zero-allocation behavior.
**Depends on**: Phase 18
**Requirements**: PORT-06, PORT-07
**Success Criteria** (what must be TRUE):
  1. The canonical Llama-68M ARM workload can execute maintained `q6_K x q8_K` hot-path dot
     products through an EMEL-owned vectorized AArch64 kernel instead of the current scalar row
     helper.
  2. Supported maintained `q2_K/q3_K/q6_K x q8_K` hot-path requests stay zero-allocation during
     dispatch and keep the current effective operand class rather than dequantize-to-f32
     fallbacks.
  3. The maintained optimized quantized path no longer depends on the scalar `q2_K/q3_K/q6_K`
     row helpers for supported canonical ARM requests.
**Plans**: 2 plans

Plans:
- [ ] 19-01: Land the maintained vectorized `q6_K x q8_K` AArch64 kernel and retire scalar row
  helper dependence for supported optimized requests.
- [ ] 19-02: Prove zero-allocation dispatch and no dequantize-to-f32 fallback across the
  maintained `q2_K/q3_K/q6_K` optimized path.

### Phase 20: Runtime Integration And Proof
**Goal**: Adopt the complete vectorized quantized kernel set in the shipped runtime chain and
prove supported plus fallback behavior without changing public APIs or actor structure.
**Depends on**: Phase 19
**Requirements**: ARCH-02, PAR-04, VER-03
**Success Criteria** (what must be TRUE):
  1. The shipped generator -> graph -> processor -> kernel chain selects the maintained
     vectorized quantized path for supported canonical ARM requests without queue-based
     orchestration, public API widening, or actor-structure rewrites.
  2. `tools/paritychecker --generation` keeps the maintained `1/10/100/1000` token checks and
     publishes proof that the canonical ARM workload exercised the vectorized quantized path.
  3. Regression and kernel tests cover `q2_K`, `q3_K`, and `q6_K` vectorized correctness against
     the scalar path plus deterministic behavior when optimization is unsupported or forced off.
  4. Unsupported or forced-off requests publish explicit deterministic fallback behavior instead
     of silent optimized-path claims.
**Plans**: 3 plans

Plans:
- [ ] 20-01: Adopt the full vectorized `q2_K/q3_K/q6_K` kernel set in the shipped generator ->
  graph -> processor -> kernel chain without changing actor structure or API boundaries.
- [ ] 20-02: Publish maintained `1/10/100/1000` parity proof and deterministic supported or
  fallback behavior for the canonical ARM workload.
- [ ] 20-03: Add regression and kernel coverage for scalar equivalence and deterministic fallback
  behavior on AArch64.

### Phase 21: Benchmark Attribution And Impact
**Goal**: Publish maintained benchmark evidence that proves and measures the vectorized quantized
path against the current v1.3 scalar baseline.
**Depends on**: Phase 20
**Requirements**: BENCH-08, BENCH-09
**Success Criteria** (what must be TRUE):
  1. `tools/bench` runs the maintained canonical ARM compare workload through the vectorized
     quantized path and publishes attribution distinct from the scalar row-helper path.
  2. Maintained compare output republishes `1`, `10`, `100`, and `1000` token results against
     the current v1.3 baseline on the canonical workload.
  3. At least one maintained generation length shows measurable end-to-end improvement over the
     current v1.3 baseline without overstating unsupported cases.
**Plans**: 2 plans

Plans:
- [ ] 21-01: Run the maintained canonical ARM compare workflow through the vectorized quantized
  path and publish attribution distinct from scalar row helpers.
- [ ] 21-02: Refresh maintained `1/10/100/1000` compare evidence against the v1.3 baseline and
  publish measurable end-to-end improvement where it exists.

## Progress

**Execution Order:**
Phases execute in numeric order: 17 -> 18 -> 19 -> 20 -> 21

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 17. Vectorized q2_K Kernel | 0/2 | Complete    | 2026-03-23 |
| 18. Vectorized q3_K Kernel | 0/2 | Complete    | 2026-03-23 |
| 19. Vectorized q6_K Kernel And Hot-Path Contract | 0/2 | Complete    | 2026-03-23 |
| 20. Runtime Integration And Proof | 0/3 | Complete    | 2026-03-23 |
| 21. Benchmark Attribution And Impact | 0/2 | Complete    | 2026-03-23 |
