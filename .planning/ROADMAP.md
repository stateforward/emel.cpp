# Roadmap

## Archived Milestones

- [x] [v1.0: EMEL Llama-68M Generation Slice](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.0-ROADMAP.md) — shipped 2026-03-08 with 7 phases and 15 plans; proved one canonical Llama-68M generation parity slice in `tools/paritychecker/`.
- [x] [v1.1: EMEL Llama-68M Generation Benchmark](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.1-ROADMAP.md) — shipped 2026-03-11 with 4 phases and 10 plans; added one truthful canonical Llama-68M generation benchmark in `tools/bench`, native EMEL decode benchmarking, compare output, and snapshot/docs integration.

## Current Milestone: v1.2 Flash Attention

**Milestone Goal:** Add an EMEL-owned flash-attention data path for the canonical CPU-hosted
Llama-68M generation slice, then prove it through the existing paritychecker and bench surfaces
without changing the Boost.SML orchestration or widening the runtime/API boundary.

**Status:** Roadmapped
**Phases:** 10-13
**Coverage:** 10/10 v1 requirements mapped

## Phases

- [x] **Phase 10: Flash Kernel Bring-Up** - Add the canonical EMEL-owned `op_flash_attn_ext` (completed 2026-03-22)
      operator and persistent hot-path workspace in `src/emel/kernel`.
- [x] **Phase 11: Generator Flash Adoption** - Route supported canonical generation requests (completed 2026-03-22)
      through flash attention inside the existing generator to kernel chain.
- [ ] **Phase 12: Parity And Verification Closure** - Prove the shipped flash path ran and stayed
      parity-stable on the canonical Llama-68M workload.
- [ ] **Phase 13: Benchmark Evidence** - Publish canonical flash-attention benchmark evidence
      through the maintained compare and artifact workflow.

## Phase Details

### Phase 10: Flash Kernel Bring-Up
**Goal**: The canonical Llama-68M attention step has a real EMEL-owned flash-attention kernel path
available in `src/emel/kernel` with reusable workspace semantics.
**Depends on**: v1.1 shipped benchmark slice
**Requirements**: FLASH-01, FLASH-02
**Success Criteria** (what must be TRUE):
  1. Canonical Llama-68M attention fixtures can execute through EMEL-owned `op_flash_attn_ext`
     code in `src/emel/kernel` rather than the old materialized attention path.
  2. Repeated canonical flash-attention kernel invocations reuse persistent workspace or buffers
     and do not introduce hot-path allocation churn.
**Plans**: TBD

### Phase 11: Generator Flash Adoption
**Goal**: The shipped canonical generation flow uses flash attention for supported Llama-68M
requests while keeping existing Boost.SML orchestration unchanged.
**Depends on**: Phase 10
**Requirements**: GEN-01, GEN-02
**Success Criteria** (what must be TRUE):
  1. The canonical CPU-hosted Llama-68M generation request completes through the existing
     generator -> graph -> processor -> kernel chain with flash attention selected on supported
     requests.
  2. Unsupported or non-selecting requests produce explicit deterministic non-flash behavior or
     failure reporting; the runtime never silently claims flash-path execution.
**Plans**: TBD

### Phase 12: Parity And Verification Closure
**Goal**: The canonical shipped flash-attention path is provably exercised and remains correct on
the accepted parity surfaces.
**Depends on**: Phase 11
**Requirements**: PAR-01, PAR-02, VER-01
**Success Criteria** (what must be TRUE):
  1. `tools/paritychecker --generation` can prove the EMEL flash-attention path executed on
     `tests/models/Llama-68M-Chat-v1-Q2_K.gguf` without introducing a new user-facing surface.
  2. The canonical flash-attention generation slice remains parity-stable against an aligned
     `llama.cpp` reference configuration for the maintained workload.
  3. The automated test suite covers shared-kernel correctness, generator flash adoption, and
     deterministic negative selection or fallback behavior.
**Plans**: TBD

### Phase 13: Benchmark Evidence
**Goal**: The existing benchmark workflow publishes truthful flash-attention performance evidence
for the canonical Llama-68M generation slice.
**Depends on**: Phase 12
**Requirements**: BENCH-01, BENCH-02, BENCH-03
**Success Criteria** (what must be TRUE):
  1. `tools/bench` can run the canonical EMEL flash-attention path through the existing compare
     workflow without adding a broader runtime or API surface.
  2. Maintained benchmark output and artifacts clearly identify flash-attention evidence separately
     from the prior non-flash baseline.
  3. At least one maintained canonical compare case shows measurable improvement over the current
     EMEL non-flash baseline.
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 10 → 11 → 12 → 13

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 10. Flash Kernel Bring-Up | 1/2 | Complete    | 2026-03-22 |
| 11. Generator Flash Adoption | 2/2 | Complete    | 2026-03-22 |
| 12. Parity And Verification Closure | 0/TBD | Not started | - |
| 13. Benchmark Evidence | 0/TBD | Not started | - |
