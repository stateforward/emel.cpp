# Requirements: EMEL v1.25 I/O Read Loading Strategy

**Defined:** 2026-05-05
**Status:** GAP CLOSURE
**Core Value:** Prove real end-to-end behavior with explicit SML orchestration and
parity-oriented verification before widening API surface or model scope.
**Source:** GitHub issue #62, "Add io/read state machine for copy-based tensor loading"

## v1 Requirements

Requirements for this milestone. Each maps to exactly one active roadmap phase.

### Read Strategy

- [x] **READ-01**: Maintainer can identify a dedicated `src/emel/io/read` Stateforward.SML
  component with component-local `context`, `events`, `guards`, `actions`, `errors`, `sm`, and
  canonical `emel::io::read::sm` ownership.
- [x] **READ-02**: The read strategy validates request, platform, file, offset, length, layout,
  and target-buffer preconditions through explicit guards and transitions before any open or
  read attempt is accepted.
- [x] **READ-03**: The read strategy returns a deterministic copied-bytes outcome into the
  caller-provided owned target buffer on success, releases all transient OS resources before
  publishing `_done`, and stores no dispatch-local request data in `context`.

### Tensor-To-I/O Integration

- [x] **TIO-01**: `model/tensor` can request read-based (copy) tensor loading through the public
  `emel/io` boundary while remaining the owner of tensor load, bind, evict, and residency
  transitions.
- [x] **TIO-02**: Tensor-to-I/O read success, unsupported, validation failure, file open
  failure, and read failure outcomes are represented with explicit `_done` and `_error`
  events or states, not mirrored status fields, action-selected callbacks, or context
  phase flags.
- [x] **TIO-03**: `model/loader`, maintained benchmark lanes, paritychecker lanes, and
  embedded probes can select or report read-backed loading only through public runtime
  surfaces, with no low-level read logic or actor-internal reach-through.

### Platform And Lifetime

- [x] **PLAT-01**: Platform-specific file-read details are hidden behind the I/O abstraction
  boundary and fail closed on unsupported platforms or unsupported file/resource shapes.
- [x] **LIFE-01**: Read transient resource lifetime (file descriptor / handle) is deterministic,
  bounded, and tied to the actor-owned attempt; no kernel handle is held across `_done`
  publication. The caller-provided target buffer remains owned by `model/tensor`; the
  read strategy never takes residency ownership.
- [x] **ERR-01**: Read-specific failures surface deterministic error categories (invalid
  request, unsupported resource, unsupported platform, file open failed, file seek failed,
  file read failed, short read, internal error) with enough source-backed evidence for
  tests and diagnostics, without throwing exceptions across API or actor boundaries.

### Validation And Publication

- [x] **VAL-01**: Doctest coverage proves supported read behavior and representative failure
  handling through `process_event(...)` and SML state inspection (`visit_current_states`
  and/or `is(...)`).
- [x] **VAL-02**: Domain and source guardrails fail if read implementation leaks into
  `model/loader`, if tensor residency ownership moves out of `model/tensor`, or if mmap,
  staged/chunked constrained-memory, async, or device strategy behavior lands in this
  milestone.
- [ ] **VAL-03**: Public docs, generated architecture docs, planning artifacts, lint snapshots,
  benchmark snapshots, benchmark outputs, and model artifacts are updated from maintained
  commands when required and describe read-strategy support truthfully.
- [x] **VAL-04**: Maintained benchmark and parity evidence reports read-strategy usage only
  when the EMEL lane actually runs the read-backed runtime path and does not present
  unsupported fallback behavior as read-strategy parity or performance.

## v2 Requirements

Deferred to future milestones. Tracked but not in the current roadmap.

### Concrete Strategies

- **STAGED-01**: Staged or chunked constrained-memory read strategy state machine for
  hosts that cannot hold a full tensor span in a single owned buffer.
- **ASYNC-01**: Tensor-to-I/O orchestration supports cooperative or resumable loading while
  preserving the RTC actor model and no-queue invariant.
- **DEVICE-01**: Device/resource-specific loading strategies can be added behind the I/O
  boundary without changing tensor residency semantics.

## Out of Scope

Explicitly excluded for this milestone.

| Feature | Reason |
|---------|--------|
| Mmap strategy changes | v1.24 owns the mmap path; v1.25 must not touch `src/emel/io/mmap` runtime behavior. |
| Staged/chunked constrained-memory read policy | Issue #62 is the bulk read/copy strategy only; staged policy has different lifecycle and remains a separate milestone. |
| Cooperative async loading implementation | Async strategy work has different scheduling and RTC constraints and remains a separate milestone. |
| Device-specific loading strategies | Device/resource strategy work must not be folded into the first read strategy milestone. |
| Backend-specific loading logic in `model/loader` | Loader must remain orchestration-only and must not regain low-level loading ownership. |
| Moving tensor residency lifecycle out of `model/tensor` | v1.22 made tensor the canonical residency owner and v1.25 must preserve that contract. |
| New model-family support or fixture widening | This milestone is a loading-strategy milestone, not a model-scope milestone. |
| Tool-only read scaffolds or publication-only benchmarks | Read-strategy claims must come from maintained `src/` runtime behavior, not tool-local substitutes. |

## Traceability

Which phases cover which requirements.

| Requirement | Phase | Status |
|-------------|-------|--------|
| READ-01 | Phase 212 | Validated |
| READ-02 | Phase 213 | Validated |
| READ-03 | Phase 214.1 | Validated |
| PLAT-01 | Phase 222 | Validated |
| LIFE-01 | Phase 214.1 | Validated |
| ERR-01 | Phase 214.1 | Validated |
| TIO-01 | Phase 215 | Validated |
| TIO-02 | Phase 223 | Validated |
| TIO-03 | Phase 225 | Complete |
| VAL-04 | Phase 225 | Complete |
| VAL-01 | Phase 225 | Complete |
| VAL-02 | Phase 222 | Validated |
| VAL-03 | Phase 225 | Pending |

Phase 224 is cleanup-only. Phase 225 owns refreshed source-backed audit gap closure for
runtime validation, SML orchestration readiness, maintained read/copy evidence, and
closeout artifact path truth.

**Coverage:**
- v1 requirements: 13 total
- Mapped to phases: 13
- Validated: 9
- Pending: 4
- Unmapped: 0
