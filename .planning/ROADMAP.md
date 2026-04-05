# Roadmap: EMEL

## Archived Milestones

- [x] v1.0: EMEL Llama-68M Generation Slice - shipped 2026-03-08 with 7 phases and 15 plans.
- [x] v1.1: EMEL Llama-68M Generation Benchmark - shipped 2026-03-11 with 4 phases and 10 plans.
- [x] v1.2: Flash Attention - shipped 2026-03-22 with 5 phases and 13 plans.
- [x] v1.3: ARM Flash Optimizations - shipped 2026-03-22 with 3 phases and 7 plans.
- [x] v1.4: Full Vectorized Quantized Kernels - shipped 2026-03-25 with 5 phases and 11 plans.
- [x] v1.5: Full ARM Quantized Path - shipped 2026-03-27 with 5 phases and 10 plans.
- [x] v1.6: Qwen3-0.6B Parity And Benchmark - shipped 2026-03-30 with 5 phases and 12 plans.
- [x] v1.7: Generator Prefill Submachine Decomposition - shipped 2026-03-30 with 3 phases and 6 plans.
- [x] v1.8: Truthful Qwen3 E2E Embedded Size - shipped 2026-04-02 with 6 phases and 8 plans.
- [x] v1.9: Liquid LFM2.5-1.2B Thinking ARM Slice - shipped 2026-04-02 with 8 phases and 9 plans.

## Current Milestone: v1.10 Planner Family AGENTS Hard Cutover

This milestone hard-cuts only `src/emel/batch/planner` and its child mode submachines over to the
`AGENTS.md` naming, file-base, and SML-machine contract while preserving maintained batching
behavior. It explicitly excludes generator child machines, broad repository cleanup, and ARM
benchmark or optimization claims.

## Phases

**Phase Numbering:**
- Integer phases continue from prior milestone history.
- v1.10 starts at Phase 40 because the previous highest completed phase was 39.

- [ ] **Phase 40: Planner Surface Cutover** - Canonicalize the top-level planner path, naming, and
      planner-owned orchestration surface.
- [ ] **Phase 41: Planner Mode Surface Cutover** - Canonicalize the `simple`, `sequential`, and
      `equal` child machine paths and allowed file bases.
- [ ] **Phase 42: Planner Event Boundaries** - Hard-cut planner and mode handoff to explicit typed
      machine dispatch and contract-aligned event naming.
- [ ] **Phase 43: Planner Rule Compliance** - Bring the planner family into destination-first
      transition and persistent-state compliance.
- [ ] **Phase 44: Behavior Preservation Proof** - Prove the cutover preserves maintained batching
      behavior on the current x86 validation host.

## Phase Details

### Phase 40: Planner Surface Cutover
**Goal**: Maintainers can find and invoke the top-level planner through one canonical
planner-owned surface under `src/emel/batch/planner/`.
**Depends on**: Phase 39
**Requirements**: PLAN-01, PLAN-02, PLAN-03
**Success Criteria** (what must be TRUE):
  1. Maintainer can find the planner machine entrypoint under `src/emel/batch/planner/` using only
     canonical file bases allowed by `AGENTS.md`.
  2. Planner machine exposes canonical `emel::batch::planner::sm` plus additive PascalCase public
     naming without legacy type-name ambiguity.
  3. Planner-owned orchestration logic is readable from planner-family files without chasing mixed
     helper surfaces outside the planner boundary.
**Plans**: TBD

### Phase 41: Planner Mode Surface Cutover
**Goal**: Maintainers can find each planner mode under a planner-owned path with only canonical
machine files exposed.
**Depends on**: Phase 40
**Requirements**: MODE-01, MODE-03
**Success Criteria** (what must be TRUE):
  1. `simple`, `sequential`, and `equal` each live under `src/emel/batch/planner/modes/` in
     planner-owned paths that match the hard-cut naming scheme.
  2. Each mode exposes only canonical machine, data, guard, action, event, error, and detail
     files, with no extra legacy surface left behind.
  3. A maintainer can open any mode directory and see its machine definition, state data, guards,
     actions, and events colocated in that family.
**Plans**: TBD

### Phase 42: Planner Event Boundaries
**Goal**: Planner and mode actors interact through explicit typed handoff instead of hidden
reach-through.
**Depends on**: Phase 41
**Requirements**: MODE-02, RULE-02
**Success Criteria** (what must be TRUE):
  1. Planner-to-mode dispatch occurs only through machine wrappers and typed events, with no direct
     cross-machine action, guard, or helper invocation.
  2. Mode outcomes return through explicit contract-aligned events, so a reviewer can trace handoff
     without direct context mutation across machines.
  3. Publicly exposed planner-family events remain small and immutable, while same-RTC internal
     handoff stays inside the planner family boundary.
**Plans**: TBD

### Phase 43: Planner Rule Compliance
**Goal**: Planner-family machines satisfy the AGENTS hard-cut rules for transition form and
persistent state ownership.
**Depends on**: Phase 42
**Requirements**: RULE-01, RULE-03
**Success Criteria** (what must be TRUE):
  1. Every touched planner-family transition table reads in destination-first form with explicit
     phase sections and no new source-first rows.
  2. Planner-family context carries only persistent actor-owned state; per-dispatch request, phase,
     status, and output data travel only on typed internal events.
**Plans**: TBD

### Phase 44: Behavior Preservation Proof
**Goal**: The hard cutover lands with behavior-preservation proof on the current x86 development
host.
**Depends on**: Phase 43
**Requirements**: PROOF-01, PROOF-02
**Success Criteria** (what must be TRUE):
  1. Focused planner-family tests fail if maintained batching behavior changes during the structural
     cutover.
  2. Required validation passes on x86 for this milestone without ARM publication, ARM optimization
     claims, or widened benchmark scope.
  3. Milestone evidence shows planner behavior preserved after the naming and rule cutover rather
     than relying on file moves alone.
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 40. Planner Surface Cutover | 0/1 | Blocked | - |
| 41. Planner Mode Surface Cutover | 0/TBD | Not started | - |
| 42. Planner Event Boundaries | 0/TBD | Not started | - |
| 43. Planner Rule Compliance | 0/TBD | Not started | - |
| 44. Behavior Preservation Proof | 0/TBD | Not started | - |
