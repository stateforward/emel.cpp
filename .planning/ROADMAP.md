# Roadmap

## Archived Milestones

- [x] [v1.0: EMEL Llama-68M Generation Slice](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.0-ROADMAP.md) — shipped 2026-03-08 with 7 phases and 15 plans; proved one canonical Llama-68M generation parity slice in `tools/paritychecker/`.
- [x] [v1.1: EMEL Llama-68M Generation Benchmark](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.1-ROADMAP.md) — shipped 2026-03-11 with 4 phases and 10 plans; added one truthful canonical Llama-68M generation benchmark in `tools/bench`, native EMEL decode benchmarking, compare output, and snapshot/docs integration.
- [x] [v1.2: Flash Attention](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.2-ROADMAP.md) — shipped 2026-03-22 with 5 phases and 13 plans; added an EMEL-owned flash-attention path to the canonical Llama-68M slice, hard-cut runtime tensor lifecycle through `emel::tensor::sm`, and published maintained benchmark evidence over a preserved pre-flash baseline.
- [x] [v1.3: ARM Flash Optimizations](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.3-ROADMAP.md) — shipped 2026-03-22 with 3 phases and 7 plans; delivered optimized AArch64 flash execution, maintained runtime/parity attribution, and preserved-baseline benchmark publication for the canonical ARM Llama-68M slice.
- [x] [v1.4: Full Vectorized Quantized Kernels](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.4-ROADMAP.md) — shipped 2026-03-25 with 5 phases and 11 plans; delivered EMEL-owned vectorized q2/q3/q6 kernels, full maintained `1/10/100/1000` parity proof, and quantized benchmark attribution on the canonical ARM slice.

## Current Milestone

### v1.5 Full ARM Quantized Path

**Milestone Goal:** Prove the maintained canonical ARM generation slice stays on the intended
quantized operand path end to end, eliminate any disallowed f32/dequant detours that still exist,
and publish explicit proof where full quantized coverage still does not apply.

**Scope Guardrails:**
- Keep the milestone narrow to the canonical CPU-hosted Llama-68M ARM slice.
- Keep `tools/paritychecker` and `tools/bench` as the only acceptance surfaces.
- Preserve the existing generator -> graph -> processor -> kernel actor chain and current public
  API boundaries.
- Do not accept dequantize-to-f32 or tool-only compute fallbacks in the shipped canonical hot path
  without explicit user approval.

## Phases

- [ ] **Phase 22: Quantized Path Audit And Contract** - Inventory the maintained canonical ARM
  operand path and make explicit which branches are native quantized, approved dense-f32-by-
  contract, or disallowed fallback.
- [ ] **Phase 23: ARM Quantized Path Closure** - Remove any remaining disallowed f32 or
  dequantize-to-f32 widening on supported canonical ARM quantized requests without changing actor
  structure.
- [ ] **Phase 24: Quantized Path Proof And Regression** - Extend paritychecker and test coverage
  so maintained proof surfaces fail if the canonical ARM request regresses to disallowed fallback.
- [ ] **Phase 25: Quantized Attribution And Impact** - Publish benchmark attribution that shows the
  end-to-end impact of full quantized-path closure and isolates the next bottleneck honestly.

## Phase Details

### Phase 22: Quantized Path Audit And Contract
**Goal**: Inventory the maintained canonical ARM operand path and encode the supported versus
unsupported quantized-path contract explicitly.
**Depends on**: Phase 21
**Requirements**: AUD-01, PATH-02
**Success Criteria** (what must be TRUE):
  1. The canonical ARM generation chain has a maintained audit that identifies each quantized stage
     as native quantized, approved dense-f32-by-contract, or disallowed fallback.
  2. Unsupported or not-yet-ported quantized cases publish explicit no-claim behavior instead of
     silently routing through a misleading f32 fallback path.
  3. The audit output is grounded in the shipped runtime chain rather than tool-local assumptions.
**Plans**: 2 plans

Plans:
- [ ] 22-01: Audit the maintained canonical ARM generation chain and capture the operand-format
  contract for each quantized stage.
- [ ] 22-02: Encode explicit no-claim behavior and operator inventory surfaces for unsupported or
  not-yet-ported quantized branches.

### Phase 23: ARM Quantized Path Closure
**Goal**: Remove any remaining disallowed f32 or dequantize-to-f32 widening on supported canonical
ARM quantized requests.
**Depends on**: Phase 22
**Requirements**: PATH-01
**Success Criteria** (what must be TRUE):
  1. Supported canonical ARM quantized requests do not silently widen through disallowed whole-row
     or operator-level dequantize-to-f32 substitution in the shipped runtime path.
  2. The closure work preserves the existing generator -> graph -> processor -> kernel actor chain
     and public API boundaries.
  3. Any still-unsupported cases remain explicit no-claim paths instead of silent fallback.
**Plans**: 2 plans

Plans:
- [ ] 23-01: Close the remaining supported canonical ARM branches that still widen through
  disallowed f32 or dequantize-to-f32 pathing.
- [ ] 23-02: Prove the shipped runtime path preserves the approved operand contract after the
  closure work.

### Phase 24: Quantized Path Proof And Regression
**Goal**: Extend maintained proof and regression surfaces so they fail if the canonical ARM
request regresses away from the approved quantized path.
**Depends on**: Phase 23
**Requirements**: ATTR-01, VER-04, PAR-05
**Success Criteria** (what must be TRUE):
  1. Maintained paritychecker output proves the canonical ARM workload stays on the approved
     quantized path across `1`, `10`, `100`, and `1000` tokens.
  2. Kernel, runtime, and regression tests cover the audited quantized-path branches and detect
     regressions back to disallowed f32 fallback.
  3. Maintained proof surfaces publish enough attribution to distinguish approved contract stages
     from disallowed fallback.
**Plans**: 2 plans

Plans:
- [ ] 24-01: Extend paritychecker and runtime attribution so maintained proof fails on disallowed
  canonical ARM fallback.
- [ ] 24-02: Add kernel and regression coverage for the audited quantized-path branches and their
  explicit no-claim cases.

### Phase 25: Quantized Attribution And Impact
**Goal**: Publish maintained benchmark attribution that shows the end-to-end impact of full
quantized-path closure and honestly isolates the next bottleneck.
**Depends on**: Phase 24
**Requirements**: BENCH-10
**Success Criteria** (what must be TRUE):
  1. Maintained benchmark output attributes whether the canonical ARM request stayed on the
     approved quantized path.
  2. Published benchmark evidence isolates the end-to-end impact of the path-closure work from
     remaining generator-side ARM math cost.
  3. The generated docs and stored compare artifacts tell the truthful post-closure performance
     story without overstating unsupported cases.
**Plans**: 2 plans

Plans:
- [ ] 25-01: Refresh the maintained compare workflow to publish quantized-path attribution after
  path closure.
- [ ] 25-02: Regenerate stored benchmark evidence and docs that isolate the next post-closure
  bottleneck honestly.

## Progress

**Execution Order:**
Phases execute in numeric order: 22 -> 23 -> 24 -> 25

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 22. Quantized Path Audit And Contract | 0/2 | Not started | - |
| 23. ARM Quantized Path Closure | 0/2 | Not started | - |
| 24. Quantized Path Proof And Regression | 0/2 | Not started | - |
| 25. Quantized Attribution And Impact | 0/2 | Not started | - |
