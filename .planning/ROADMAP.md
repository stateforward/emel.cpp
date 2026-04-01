# Roadmap

## Archived Milestones

- [x] [v1.0: EMEL Llama-68M Generation Slice](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.0-ROADMAP.md) - shipped 2026-03-08 with 7 phases and 15 plans; proved one canonical Llama-68M generation parity slice in `tools/paritychecker/`.
- [x] [v1.1: EMEL Llama-68M Generation Benchmark](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.1-ROADMAP.md) - shipped 2026-03-11 with 4 phases and 10 plans; added one truthful canonical Llama-68M generation benchmark in `tools/bench`, native EMEL decode benchmarking, compare output, and snapshot/docs integration.
- [x] [v1.2: Flash Attention](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.2-ROADMAP.md) - shipped 2026-03-22 with 5 phases and 13 plans; added an EMEL-owned flash-attention path to the canonical Llama-68M slice, hard-cut runtime tensor lifecycle through `emel::tensor::sm`, and published maintained benchmark evidence over a preserved pre-flash baseline.
- [x] [v1.3: ARM Flash Optimizations](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.3-ROADMAP.md) - shipped 2026-03-22 with 3 phases and 7 plans; delivered optimized AArch64 flash execution, maintained runtime/parity attribution, and preserved-baseline benchmark publication for the canonical ARM Llama-68M slice.
- [x] [v1.4: Full Vectorized Quantized Kernels](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.4-ROADMAP.md) - shipped 2026-03-25 with 5 phases and 11 plans; delivered EMEL-owned vectorized q2/q3/q6 kernels, full maintained `1/10/100/1000` parity proof, and quantized benchmark attribution on the canonical ARM slice.
- [x] [v1.5: Full ARM Quantized Path](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.5-ROADMAP.md) - shipped 2026-03-27 with 5 phases and 10 plans; closed the maintained ARM quantized-path contract and restored canonical flash publication.
- [x] [v1.6: Qwen3-0.6B Parity And Benchmark](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.6-ROADMAP.md) - shipped 2026-03-30 with 5 phases and 12 plans; brought one canonical Qwen3 slice up through the maintained generator, parity, and benchmark surfaces.
- [x] [v1.7: Generator Prefill Submachine Decomposition](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.7-ROADMAP.md) - shipped 2026-03-30 with 3 phases and 6 plans; extracted `generator/prefill`, collapsed prefill compute routing, and preserved maintained proof across the refactor.

## Current Milestone

### v1.8 Generator Initializer Submachine

**Milestone Goal:** Extract the generator's initialize pipeline into an explicit
`generator/initializer` machine while keeping same-RTC/no-queue semantics, typed request-scoped
handoff, and maintained Llama/Qwen proof unchanged.

**Scope Guardrails:**
- Keep the milestone narrow to one new generator-owned machine: `initializer`.
- Keep the per-token decode loop in the parent generator; do not retry decode child extraction in
  this milestone.
- Preserve explicit init behavior through guards, states, transitions, and typed outcome events.
- Preserve maintained generator, paritychecker, benchmark, snapshot, docs, and quality-gate proof
  across the refactor.
- Do not broaden into attention-family extraction, richer request surfaces, or benchmark gate
  policy.

## Phases

- [x] **Phase 33: Generator Initializer Submachine Extraction** - Move initialize orchestration
      into an explicit `src/emel/generator/initializer` machine without widening the generator
      boundary.
- [x] **Phase 34: Initializer Surface Shrink And Proof** - Reduce top-level
      initialize/publication boilerplate and close the milestone with maintained proof.

## Phase Details

### Phase 33: Generator Initializer Submachine Extraction
**Goal**: Move initialize orchestration into an explicit generator-owned `initializer` machine.
**Depends on**: Phase 32
**Requirements**: INIT-01, INIT-02, INIT-03
**Success Criteria** (what must be TRUE):
  1. `src/emel/generator/initializer` owns conditioner binding, renderer initialization, memory
     reserve, optional graph reserve, and sampling configuration as explicit states inside the
     generator domain.
  2. The parent generator delegates `initialize_run` through typed events and explicit outcomes
     rather than direct helper orchestration or context phase flags.
  3. Init route selection remains explicit and request-scoped data stays on typed runtime/internal
     events instead of generator context phase members.
**Plans**: 2 plans

Plans:
- [x] 33-01: Create the generator/initializer boundary, events, context seam, and transition
      table.
- [x] 33-02: Integrate the parent generator with generator/initializer while preserving initialize
      behavior and architecture docs.

### Phase 34: Initializer Surface Shrink And Proof
**Goal**: Reduce top-level generator boilerplate and close the milestone with maintained proof.
**Depends on**: Phase 33
**Requirements**: ARCH-02, VERIFY-02
**Success Criteria** (what must be TRUE):
  1. The top-level generator surface is materially smaller and easier to inspect after the
     initializer extraction, with initialize/publication-specific boilerplate reduced.
  2. Maintained generator, paritychecker, benchmark, and quality-gate coverage remain green on the
     current Llama and canonical Qwen slices.
  3. The milestone lands without broadening into decode submachine extraction, attention-family
     `sm_any`, or separate session/runtime redesign.
**Plans**: 2 plans

Plans:
- [x] 34-01: Shrink the parent generator initializer/publication surface and update docs/tests.
- [x] 34-02: Refresh maintained regression, benchmark, and gate proof for the initializer
      boundary.

## Progress

**Execution Order:**
Phases execute in numeric order: 33 -> 34

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 33. Generator Initializer Submachine Extraction | 2/2 | Completed | 2026-03-31 |
| 34. Initializer Surface Shrink And Proof | 2/2 | Completed | 2026-03-31 |
