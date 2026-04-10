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

- [x] **Phase 40: Planner Surface Cutover** - Canonicalize the top-level planner path, naming, and (completed 2026-04-05)
      planner-owned orchestration surface.
- [x] **Phase 41: Planner Mode Surface Cutover** - Canonicalize the `simple`, `sequential`, and (completed 2026-04-05)
      `equal` child machine paths and allowed file bases.
- [x] **Phase 42: Planner Event Boundaries** - Hard-cut planner and mode handoff to explicit typed (completed 2026-04-05)
      machine dispatch and contract-aligned event naming.
- [x] **Phase 43: Planner Rule Compliance** - Bring the planner family into destination-first (completed 2026-04-05)
      transition and persistent-state compliance.
- [x] **Phase 44: Behavior Preservation Proof** - Prove the cutover preserves maintained batching (completed 2026-04-05)
      behavior on the current arm64 validation host.
- [x] **Phase 45: Planner Audit Closeout** - Close the first audit-closeout tranche by (completed 2026-04-05)
      backfilling Phase 40 proof artifacts and removing the planner-wrapper rule warning.
- [x] **Phase 46: Planner Transient Unexpected-Event Closure** - Close the remaining non-blocking (completed 2026-04-05)
      transient-state unexpected-event audit finding without widening scope beyond the planner
      family.

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
**Goal**: The hard cutover lands with behavior-preservation proof on the current arm64 development
host.
**Depends on**: Phase 43
**Requirements**: PROOF-01, PROOF-02
**Success Criteria** (what must be TRUE):
  1. Focused planner-family tests fail if maintained batching behavior changes during the structural
     cutover.
  2. Required validation passes on the current arm64 host for this milestone without ARM
     publication, ARM optimization
     claims, or widened benchmark scope.
  3. Milestone evidence shows planner behavior preserved after the naming and rule cutover rather
     than relying on file moves alone.
**Plans**: TBD

### Phase 45: Planner Audit Closeout
**Goal**: Close the v1.10 milestone audit gaps without widening scope beyond the planner family.
**Depends on**: Phase 44
**Requirements**: PLAN-01, PLAN-02, PLAN-03, RULE-01
**Gap Closure**: Closes the Phase 40 verification-orphan gap and the planner-wrapper
                 runtime-branch finding from audit closeout. Leaves FINDING-01 for dedicated
                 follow-on closure.
**Success Criteria** (what must be TRUE):
  1. Phase 40 has formal verification evidence and summary metadata that explicitly prove
     PLAN-01, PLAN-02, and PLAN-03 rather than leaving those requirements orphaned.
  2. Planner mode `sm::process_event` wrappers no longer use runtime branching in member methods
     to choose done versus error outcomes.
  3. The milestone can be re-audited with only any explicitly deferred snapshot-consent work
     remaining.
**Plans**: TBD

### Phase 46: Planner Transient Unexpected-Event Closure
**Goal**: Close the remaining transient-state unexpected-event audit note in the planner family
without widening milestone scope.
**Depends on**: Phase 45
**Requirements**: None (audit finding only)
**Gap Closure**: Closes FINDING-01 from `.planning/v1.10-MILESTONE-AUDIT.md`.
**Success Criteria** (what must be TRUE):
  1. `state_simple_mode`, `state_equal_mode`, and `state_sequential_mode` have explicit
     contract-aligned handling for unexpected external events, or a planner-owned implementation
     makes the approved transient-state exemption explicit and uniform.
  2. The closure stays inside `src/emel/batch/planner` and focused planner-family proof surfaces,
     with no widened snapshot, generator, or broad repository cleanup scope.
  3. A follow-up milestone audit no longer reports FINDING-01 under `gaps.integration`.
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 40. Planner Surface Cutover | 1/1 | Complete    | 2026-04-05 |
| 41. Planner Mode Surface Cutover | 1/1 | Complete    | 2026-04-05 |
| 42. Planner Event Boundaries | 1/1 | Complete    | 2026-04-05 |
| 43. Planner Rule Compliance | 1/1 | Complete    | 2026-04-05 |
| 44. Behavior Preservation Proof | 1/1 | Complete    | 2026-04-05 |
| 45. Planner Audit Closeout | 1/1 | Complete    | 2026-04-05 |
| 46. Planner Transient Unexpected-Event Closure | 1/1 | Complete    | 2026-04-05 |
| 46.1 Rename planner-family wrapper names to remove ambiguity and wrapper indirection | 1/1 | Complete    | 2026-04-05 |

### Phase 46.1: Rename planner-family wrapper names to remove ambiguity and wrapper indirection (INSERTED)

**Goal:** Replace the planner family's remaining wrapper-era names with direct planner-owned event,
state, and callable names, including the child-mode action surfaces.
**Requirements**: None (user-inserted rename closeout)
**Depends on:** Phase 46
**Success Criteria** (what must be TRUE):
  1. The top-level planner trigger/runtime surface uses direct names like `plan_request`,
     `plan_scratch`, and `plan_runtime`, and maintained direct callers build on those names.
  2. The planner graph and callable surface use descriptive planner-owned names instead of
     wrapper-era `effect_*`, `guard_*`, ambiguous decision-state naming, and mode-action
     trampolines.
  3. Planner-family surface proof fails if the old wrapper-era names reappear in the planner
     family, including the mode action headers.
**Plans:** 1 plan

Plans:
- [x] 46.1-01 - Rename the planner-family surface in place, remove child-mode action trampolines,
      and lock it with planner-family regression proof.
