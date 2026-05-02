# Requirements: EMEL v1.22 Weight Loading Ownership Cutover

**Defined:** 2026-05-02
**Core Value:** Prove real end-to-end behavior with explicit SML orchestration and parity-oriented
verification before widening API surface or model scope.
**Source:** GitHub issue #59, "Cut over weight loading ownership from model/weight_loader to
model/tensor"

## v1 Requirements

Requirements for this milestone. Each maps to exactly one roadmap phase.

### Tensor Ownership

- [ ] **TENSOR-01**: Maintainer can identify `src/emel/model/tensor` as the canonical owner of
  tensor load, bind, evict, and residency semantics.
- [ ] **TENSOR-02**: Model tensor runtime can perform bulk tensor residency transitions through
  tensor-owned events, guards, actions, and context instead of a separate weight-loader owner.
- [ ] **TENSOR-03**: Model tensor loading outcomes are represented with explicit `_done` and
  `_error` events or states rather than mirrored status fields or dispatch-local context.
- [ ] **TENSOR-04**: Existing per-tensor lifecycle behavior for bind, evict, and state capture is
  preserved after adding tensor-owned loading behavior.

### Loader Integration

- [ ] **LOAD-01**: Model loader orchestration no longer treats the current bulk `load_weights_fn`
  callback as the long-term architecture seam.
- [ ] **LOAD-02**: Model loader dispatches or coordinates tensor-owned residency transitions without
  directly owning backend-specific loading strategy logic.
- [ ] **LOAD-03**: Existing externally observable model-loading behavior remains equivalent or is
  intentionally updated with matching tests and documentation.
- [ ] **LOAD-04**: Model-loading failure paths remain explicit, deterministic, and covered through
  error events or error states.

### Weight Loader Retirement

- [ ] **CUTOVER-01**: The primary runtime model-loading path no longer depends on
  `src/emel/model/weight_loader` as the owner of weight-loading behavior.
- [ ] **CUTOVER-02**: Any transitional compatibility adapter is clearly named, bounded in scope, and
  documented as non-owning.
- [ ] **CUTOVER-03**: CMake wiring, includes, tests, and docs no longer present `model/weight_loader`
  as a parallel residency ownership layer.
- [ ] **CUTOVER-04**: Source-backed guardrails prevent reintroducing a second model-weight residency
  owner under `model/weight_loader` or an equivalent shadow path.

### Future IO Readiness

- [ ] **IO-01**: Tensor-owned loading boundaries are structured so a future `emel/io` module can
  provide loading strategies underneath tensor ownership.
- [ ] **IO-02**: The milestone leaves asynchronous loading and concrete I/O strategy implementation
  out of scope while documenting the intended follow-on seam.

## v2 Requirements

Deferred to future milestones. Tracked but not in the current roadmap.

### IO Strategies

- **IO-03**: `emel/io` provides one explicit state machine per loading strategy.
- **IO-04**: Tensor-to-IO orchestration supports cooperative asynchronous loading where appropriate.
- **IO-05**: Loading strategy selection is modeled with explicit guards and transitions, not hidden
  helper branching.

## Out of Scope

Explicitly excluded for this milestone.

| Feature | Reason |
|---------|--------|
| New `emel/io` runtime implementation | Issue #59 is the ownership cutover, not the strategy implementation PR. |
| Cooperative async loading | The issue explicitly excludes asynchronous scheduling for this cutover. |
| Backend-specific loading logic in `model/loader` | Loader must orchestrate tensor-owned behavior without becoming a loading-strategy owner. |
| New model-family or fixture support | The milestone must preserve existing loading behavior rather than widen model scope. |
| Performance claims for new loading strategies | No new strategy is implemented, so there is no new strategy benchmark claim to publish. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| TENSOR-01 | Phase 185 | Pending |
| IO-01 | Phase 185 | Pending |
| IO-02 | Phase 185 | Pending |
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

**Coverage:**
- v1 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0

---
*Requirements defined: 2026-05-02*
*Last updated: 2026-05-02 after roadmap creation*
