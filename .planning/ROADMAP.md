# Roadmap: EMEL v1.22 Weight Loading Ownership Cutover

**Milestone:** v1.22 Weight Loading Ownership Cutover
**Source:** GitHub issue #59, "Cut over weight loading ownership from model/weight_loader to
model/tensor"
**Started:** 2026-05-02

## Goal

Make `src/emel/model/tensor` the canonical owner of tensor load, bind, evict, and residency
transitions while removing `src/emel/model/weight_loader` from the primary runtime load path.

## Scope

This milestone is an ownership and orchestration cutover. It preserves existing external
model-loading behavior, keeps runtime behavior selection in SML guards/transitions, and prepares a
future `emel/io` strategy seam without implementing asynchronous loading or concrete I/O strategies.

## Phase Overview

| Phase | Name | Goal | Requirements |
|-------|------|------|--------------|
| 185 | Tensor Ownership Contract | Define the new tensor-owned loading boundary and future IO seam before moving behavior. | TENSOR-01, IO-01, IO-02 |
| 186 | Tensor-Owned Loading Runtime | Add tensor-owned load/residency events, states, guards, actions, and behavior-preservation tests. | TENSOR-02, TENSOR-03, TENSOR-04 |
| 187 | Loader-To-Tensor Cutover | Rewire model loader orchestration to target tensor-owned residency behavior and explicit errors. | LOAD-01, LOAD-02, LOAD-03, LOAD-04 |
| 188 | Weight Loader Path Retirement | Remove `model/weight_loader` as the primary runtime owner and bound any transitional adapter. | CUTOVER-01, CUTOVER-02, CUTOVER-03 |
| 189 | Ownership Guardrails And Closeout | Add source-backed guardrails, final behavior coverage, and closeout evidence for the ownership cutover. | CUTOVER-04 |

## Phases

### Phase 185: Tensor Ownership Contract

**Goal:** Establish the source-backed ownership contract for tensor loading before implementation
changes begin.

**Requirements:** TENSOR-01, IO-01, IO-02

**Success criteria:**
1. Current `model/tensor`, `model/weight_loader`, and `model/loader` responsibilities are traced
   from source, CMake, tests, and public aliases.
2. The planned tensor-owned loading boundary is documented in phase artifacts without introducing
   async loading or a concrete `emel/io` strategy.
3. The future `emel/io` seam is identified as below tensor ownership, not beside `model/tensor` or
   inside `model/loader`.
4. Phase plan names the exact files/components that may change during the cutover and the files that
   must not absorb loading-strategy ownership.

### Phase 186: Tensor-Owned Loading Runtime

**Goal:** Move load/residency semantics into `model/tensor` while preserving existing per-tensor
life-cycle behavior.

**Requirements:** TENSOR-02, TENSOR-03, TENSOR-04

**Success criteria:**
1. `src/emel/model/tensor` owns the events, context, guards, actions, and transition rows needed for
   load/residency transitions.
2. Tensor loading completion and error outcomes are modeled through explicit state-machine outcomes,
   not mirrored status fields or dispatch-local context.
3. Existing bind, evict, and capture-state behavior remains covered after adding tensor-owned
   loading behavior.
4. New or changed transition tables remain destination-first, bounded, non-allocating during
   dispatch, and compliant with `docs/rules/sml.rules.md`.

### Phase 187: Loader-To-Tensor Cutover

**Goal:** Rewire model loader orchestration so bulk model loading coordinates tensor-owned behavior
instead of owning the bulk weight-loading seam.

**Requirements:** LOAD-01, LOAD-02, LOAD-03, LOAD-04

**Success criteria:**
1. `src/emel/model/loader` no longer treats `load_weights_fn` as the long-term owner seam for
   residency transitions.
2. Loader orchestration dispatches or coordinates tensor-owned loading behavior without embedding
   backend-specific loading strategy logic.
3. Existing model-loading happy-path behavior remains equivalent and covered by focused tests.
4. Model-loading failure paths remain explicit, deterministic, and tested through error states or
   `_error` events.

### Phase 188: Weight Loader Path Retirement

**Goal:** Remove `model/weight_loader` as the primary runtime owner of weight loading and keep any
temporary bridge visibly transitional.

**Requirements:** CUTOVER-01, CUTOVER-02, CUTOVER-03

**Success criteria:**
1. The primary runtime model-loading path no longer depends on `src/emel/model/weight_loader` as the
   owner of weight-loading behavior.
2. CMake, includes, aliases, tests, and docs no longer present `model/weight_loader` as a parallel
   residency ownership layer.
3. Any compatibility shim is explicitly named as transitional, non-owning, and bounded to the
   minimum migration surface.
4. No new top-level runtime domain or model-family-specific loading owner is introduced.

### Phase 189: Ownership Guardrails And Closeout

**Goal:** Prove the ownership cutover is source-backed, behavior-preserving, and resistant to
regression before milestone closeout.

**Requirements:** CUTOVER-04

**Success criteria:**
1. Source checks or tests fail if a second model-weight residency owner is reintroduced under
   `model/weight_loader` or an equivalent shadow path.
2. Focused model tensor, model loader, and cutover tests pass through maintained public actor or
   wrapper interfaces rather than actor internals.
3. The default changed-file quality gate passes for the implementation files changed in this
   milestone.
4. Closeout evidence explicitly distinguishes this ownership cutover from the deferred future
   `emel/io` strategy milestone.

## Requirement Coverage

| Requirement | Phase | Status |
|-------------|-------|--------|
| TENSOR-01 | Phase 185 | Pending |
| TENSOR-02 | Phase 186 | Pending |
| TENSOR-03 | Phase 186 | Pending |
| TENSOR-04 | Phase 186 | Pending |
| LOAD-01 | Phase 187 | Pending |
| LOAD-02 | Phase 187 | Pending |
| LOAD-03 | Phase 187 | Pending |
| LOAD-04 | Phase 187 | Pending |
| CUTOVER-01 | Phase 188 | Pending |
| CUTOVER-02 | Phase 188 | Pending |
| CUTOVER-03 | Phase 188 | Pending |
| CUTOVER-04 | Phase 189 | Pending |
| IO-01 | Phase 185 | Pending |
| IO-02 | Phase 185 | Pending |

**Coverage:** 14/14 requirements mapped.

## Next Step

Start with Phase 185:

```bash
$gsd-discuss-phase 185
```

or plan directly:

```bash
$gsd-plan-phase 185
```
