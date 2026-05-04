# Requirements: EMEL v1.23 I/O Loading Strategy Boundary

**Defined:** 2026-05-04
**Core Value:** Prove real end-to-end behavior with explicit SML orchestration and
parity-oriented verification before widening API surface or model scope.
**Source:** GitHub issue #60, "Add emel/io module and tensor-to-io orchestration boundary"

## v1 Requirements

Requirements for this milestone. Each maps to exactly one roadmap phase.

### I/O Module

- [ ] **IO-01**: Maintainer can identify `src/emel/io` as a first-class runtime module with
  component-local Stateforward.SML machine organization, context ownership, events, guards,
  actions, and canonical `emel::io::<component>::sm` aliases.
- [ ] **IO-02**: `emel/io` owns loading strategy, transport, mapping, staging, and
  device/resource-specific loading strategy behavior without owning tensor residency lifecycle
  semantics.
- [ ] **IO-03**: The new I/O boundary exposes explicit request, result, and error event contracts
  for tensor-backed loading without hidden shared state, retained dispatch-local payloads, or
  callback-stored context.

### Tensor Boundary

- [ ] **TBOUND-01**: `model/tensor` can request loading work through the I/O boundary while
  remaining the canonical owner of tensor load, bind, evict, and residency transitions.
- [ ] **TBOUND-02**: Tensor-to-I/O success and failure outcomes are represented with explicit
  `_done` and `_error` events or states rather than mirrored status fields, dispatch-local context,
  or action-selected callbacks.
- [ ] **TBOUND-03**: Existing tensor-owned residency behavior remains equivalent when no concrete
  I/O strategy is selected or when the boundary rejects a request deterministically.

### Strategy Policy

- [ ] **POLICY-01**: Loading strategy availability and policy injection points are explicit enough
  for future mmap, staged read, and copy-based strategies to land independently.
- [ ] **POLICY-02**: Runtime strategy choice is modeled with guards, choice states, and transition
  rows in `sm.hpp`, not branching in actions, detail helpers, or state-machine member functions.
- [ ] **POLICY-03**: The seam preserves room for future cooperative or resumable loading without
  implementing asynchronous scheduling, mailboxes, deferred queues, or post-for-later behavior in
  this milestone.

### Loader Ownership

- [ ] **LOAD-01**: `model/loader` remains a high-level orchestrator and does not regain
  backend-specific byte access, mapping, staging, or loading strategy implementation logic.
- [ ] **LOAD-02**: Maintained GGUF loader, benchmark, paritychecker, and embedded probe entrypoints
  continue to drive model loading through public runtime surfaces without reaching into I/O,
  tensor, or loader actor internals.

### Validation And Guardrails

- [ ] **VAL-01**: Tests cover supported tensor-to-I/O boundary behavior and representative
  deterministic failure handling through public event interfaces and SML state inspection.
- [ ] **VAL-02**: Domain and source guardrails fail if concrete strategy implementations land in
  this milestone, if `model/loader` regains low-level strategy logic, or if a shadow residency owner
  appears outside `model/tensor`.
- [ ] **VAL-03**: Public docs and planning artifacts describe the ownership split truthfully:
  `model/tensor` owns residency, `emel/io` owns strategy boundaries, and concrete mmap/read/copy
  strategies are follow-on work.

## v2 Requirements

Deferred to future milestones. Tracked but not in the current roadmap.

### Concrete Strategies

- **MMAP-01**: A dedicated mmap strategy state machine exists under `src/emel/io` and provides
  memory-mapped tensor residency where the platform and file layout allow it.
- **READ-01**: A staged or explicit read/copy strategy state machine exists under `src/emel/io`.
- **ASYNC-01**: Tensor-to-I/O orchestration supports cooperative or resumable loading while
  preserving the RTC actor model and no-queue invariant.
- **DEVICE-01**: Device/resource-specific loading strategies can be added behind the I/O boundary
  without changing tensor residency semantics.

## Out of Scope

Explicitly excluded for this milestone.

| Feature | Reason |
|---------|--------|
| Concrete mmap strategy implementation | Issue #60 defines the I/O boundary; issue #61 owns mmap strategy behavior. |
| Staged, chunked, or explicit read/copy strategy implementation | The milestone must create the strategy seam, not land every strategy behind it. |
| Cooperative async loading implementation | Issue #60 asks the seam to support future async work, not to implement scheduling now. |
| Backend-specific loading logic in `model/loader` | Loader must remain orchestration-only and must not regain low-level loading ownership. |
| Moving tensor residency lifecycle out of `model/tensor` | v1.22 made tensor the canonical residency owner; v1.23 must preserve that contract. |
| New model-family support or fixture widening | This is an architecture-boundary milestone, not a model-scope milestone. |
| New concrete strategy benchmark claims | No concrete strategy is implemented, so performance claims would be misleading. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| IO-01 | Phase 197 | Pending |
| IO-02 | Phase 197 | Pending |
| IO-03 | Phase 198 | Pending |
| TBOUND-01 | Phase 198 | Pending |
| TBOUND-02 | Phase 198 | Pending |
| TBOUND-03 | Phase 199 | Pending |
| POLICY-01 | Phase 199 | Pending |
| POLICY-02 | Phase 199 | Pending |
| POLICY-03 | Phase 199 | Pending |
| LOAD-01 | Phase 200 | Pending |
| LOAD-02 | Phase 200 | Pending |
| VAL-01 | Phase 201 | Pending |
| VAL-02 | Phase 201 | Pending |
| VAL-03 | Phase 201 | Pending |

**Coverage:**
- v1 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0

---
*Requirements defined: 2026-05-04*
*Last updated: 2026-05-04 after roadmap creation*
