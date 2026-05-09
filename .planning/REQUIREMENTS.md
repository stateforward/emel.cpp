# Requirements: EMEL v1.26 I/O Staged Read Loading Strategy

**Defined:** 2026-05-07  
**Revised:** 2026-05-08 (milestone audit gap closure phases **237-238** added)  
**Status:** COMPLETE (v1.26 phases **227-238 complete**; `ESG-02B` deferred/future)  
**Core Value:** Prove real end-to-end behavior with explicit SML orchestration and
parity-oriented verification before widening API surface or model scope.  
**Source:** GitHub issue #63, "Add io/staged_read state machine for constrained-memory tensor loading"

## v1.26 Requirements

Each requirement is one independently testable obligation and maps to **exactly
one** roadmap phase. REQ-IDs are unique to this milestone (no overlap with
satisfied v1.25 identifiers).

### Staged read component and pre-I/O guards

- [x] **STG-01**: Maintainer can identify a dedicated `src/emel/io/staged_read`
  Stateforward.SML component with component-local `context`, `events`, `guards`,
  `actions`, `errors`, `sm`, and canonical `emel::io::staged_read::sm` ownership.

- [x] **STG-02**: Guards reject invalid overall source span or invalid
  staging/chunk contract (including offset/total-length/chunk sizing rules as
  defined by implementation) **before** any staged file I/O attempt is accepted.

- [x] **STG-03**: Guards reject invalid caller-owned target-buffer window or
  layout precondition for staged copy **before** any staged file I/O attempt is
  accepted.

- [x] **PLAT-02**: Unsupported platform hosts or unsupported staged resource
  shapes fail closed through explicit guarded transitions/states/events (not via
  unmodeled unchecked platform calls). *Satisfaction evidence is the compile-time
  platform guard plus the explicit SML unsupported branch on this milestone’s
  hosts; this ledger does **not** claim an executed CI/runtime proof on an
  “unsupported-host” OS image.*

### Staged execution semantics (actor body)

- [x] **STG-04**: On each accepted stage boundary, staged read copies the
  intended contiguous source byte sub-span into the designated caller-owned
  target region for that stage deterministically (verifiable by test vectors).
  *(Satisfied via single-dispatch tiling loop in **`effect_publish_staged_window_done`**
  backed by doctest vectors; staged “window” aligns with **`stage_chunk_bytes`**
  plus deterministic tail.)*

- [x] **STG-05**: A fully successful staged run covers the entire requested
  logical source byte span with strictly monotonic forward progress across
  stages (no gaps or backward contract motion in the logical span).
  *(Monotonicity evidenced by oracle memcmp covering the contiguous logical span
  (`tests/io/staged_read/lifecycle_tests.cpp`), including uneven chunk remainders.)*

- [x] **STG-06**: A fully successful staged run publishes a single terminal
  success outcome that corresponds to the complete logical span being copied per
  the staged-read contract.
  *(Exactly one **`staged_window_done`** with **`bytes_committed == logical_byte_length`**.)*

- [x] **STG-07**: `staged_read::context` does not retain dispatch-local request
  payload (no pointers, references, or mirrored handles to the current request
  stored in context for multi-event handoff that violates EMEL dispatch-local
  rules).

### Lifetime and ownership (staged actor)

- [x] **LIFE-02**: Transient OS resources acquired for a stage attempt are
  released before that stage’s terminal outcome is published, and no kernel
  handle is retained past the staged-read actor’s published `_done` for the
  overall request. *(Current Phase 230 source-backed caveat: staged_read
  consumes caller-provided `source_span` bytes and acquires no OS handle, so
  there is no staged_read-owned kernel handle to retain.)*

- [x] **SNR-01**: The staged-read strategy actor never claims tensor residency
  ownership of the destination buffer (copy-only semantics into caller/tensor
  windows).

### Error taxonomy

- [x] **ESG-01**: Pre-I/O guard failures map to named deterministic staged-read
  error categories. **Supported-host doctests cover the forceable validation
  categories** exercised in staged-read lifecycle tests. **Unsupported platform
  and unsupported resource-category outcomes remain source-backed via the
  compiled guard, transition, and action graph** (`sm.hpp`, `errors.hpp`).
  Unsupported-host runtime forcing for those terminals is deferred to future
  platform-targeted validation.

- [x] **ESG-02A**: Source-contract read-surface failures in the current
  source-span staged-read design (null source, `source_span_bytes` mismatch, and
  insufficient/short source span) map to named deterministic staged-read error
  categories observable in tests.

- [ ] **ESG-02B** (**Deferred/Future**): Real file open, seek, and read failures,
  including per-stage short read, map to named deterministic staged-read error
  categories observable in tests only after an approved file-backed staged-read
  source path owns handle/syscall behavior.

- [x] **ESG-03**: Staged sequencing or stage-contract violations map to a named
  deterministic staged-read error category observable in tests.

- [x] **ESG-04**: Staged-read handling does not throw exceptions across the
  staged-read actor boundary or public C ABI surfaces.

### Tensor integration

- [x] **TNX-01**: `model/tensor` initiates staged read loading only through
  public `emel/io` request events (no new private cross-actor calls).
  *(Phase 237 evidence: direct tensor `request_staged_load` dispatch still uses
  injected public `emel::io::staged_read::sm` via `process_event(...)` while
  applying the nonzero source window.)*

- [x] **TNX-02**: For tensors using staged reads, `model/tensor` remains the
  sole owner of load, bind, evict, and residency lifecycle transitions.

- [x] **TNX-03**: Staged-load success is visible to integrators through explicit
  terminal success `_done`/states in the tensor integration graph (not only
  through silent context mutation).
  *(Phase 237 evidence: `request_staged_load_done`, resident tensor state, and
  copied nonzero-offset bytes are asserted in public dispatch doctests.)*

- [x] **TNX-04**: Staged-load failure is visible through explicit terminal
  `_error`/error states/events in the tensor integration graph.
  *(Phase 237 evidence: direct staged-load validation failure maps to
  `request_staged_load_error` with explicit tensor and staged-read errors.)*

### Maintained surfaces

- [x] **PUB-01**: `model/loader` selects or reports staged-read usage only via
  public runtime contracts applicable to loaders (no `staged_read` detail
  includes).

- [x] **PUB-02**: Maintained benchmark entrypoints select or report staged-read
  usage only via public runtime contracts (no `staged_read` detail includes).

- [x] **PUB-03**: Maintained paritychecker entrypoints select or report
  staged-read usage only via public runtime contracts.

- [x] **PUB-04**: Maintained embedded-size probe entrypoints select or report
  staged-read usage only via public runtime contracts.

- [x] **PUB-05**: Maintained loaders and tools do not embed a second
  unconstrained POSIX-style staged read/copy loop duplicate of actor-owned file
  work (source scan or doctest-enforced invariant).

### Tests

- [x] **TST-01**: Doctest proves at least one fully successful staged-load path
  through `process_event(...)` with explicit SML state inspection.
  *(Phase 237 evidence: `model_tensor_request_staged_load_applies_nonzero_file_offset`
  covers direct staged-load success through public `process_event(...)` and `is(...)`.)*

- [x] **TST-02**: Doctest proves representative staged failure modes through
  `process_event(...)` including at least one pre-I/O guard failure path.
  *(Phase 237 evidence: direct staged-load validation-error doctest covers explicit
  failure publication through public dispatch.)*

### Guardrails (non-regression and scope)

- [x] **GRD-01**: Repository guardrail or focused test fails when staged-read
  file syscall loop ownership leaks into `model/loader`.

- [x] **GRD-02**: Repository guardrail or focused test fails when tensor residency
  lifecycle migrates out of `model/tensor` for the staged-load path.

- [x] **GRD-03**: Repository guardrail fails when cooperative coroutine staged
  scheduling scaffolding lands without documented separate approval.

- [x] **GRD-04**: Repository regression proof fails when shipped mmap strategy
  semantics regress.

- [x] **GRD-05**: Repository regression proof fails when shipped bulk `io/read`
  strategy semantics regress.

### Publication and artifact truth

- [x] **DOC-01**: Checked-in prose under maintained doc entrypoints accurately
  states whether staged constrained-memory loading is implemented and how it is
  reached.

- [x] **LNT-01**: `lint_snapshot` (or successor maintained lint gate) passes
  with an updated baseline when and only when staged-read-affected edits require
  it.

- [x] **BNH-01**: Benchmark snapshot deltas for suites affected by staged reads
  are refreshed only via the maintained benchmark snapshot/update workflow.

- [x] **EVI-01**: Maintained parity or benchmark artifacts do not label a run as
  staged-read-backed unless the staged runtime path actually executed.

## v2 Requirements (deferred)

Tracked for future milestones; not part of v1.26 roadmap.

### Async and device strategies

- **ASYNC-01**: Cooperative or resumable loading with explicit scheduling while
  preserving RTC invariants (only after dedicated milestone approval).

- **DEVICE-01**: Device/resource-specific loading strategies behind `emel/io`.

### Broader staging

- **STG-FUTURE-01**: Adaptive chunk sizing from runtime memory pressure signals.

## Out of Scope

Explicit exclusions for milestone v1.26:

| Feature | Reason |
|---------|--------|
| Cooperative coroutine scheduling / async suspension | Separate approval required per project brief |
| Device-specific transfers | Device-strategy milestone |
| Changing mmap semantics | v1.24 owns mmap |
| Replacing bulk `io/read` | Staged strategy is additive |
| Moving tensor residency out of `model/tensor` | Invariant |
| New model families / fixture widening | Strategy milestone only |
| Tool-only staged scaffolds without `src/` path | Evidence rule |

## Traceability

Each row appears **exactly once**. Phases continue after v1.25’s Phase 226.

| Requirement | Phase | Status |
|-------------|-------|--------|
| STG-01 | Phase 227 | Satisfied |
| STG-02 | Phase 228 | Satisfied |
| STG-03 | Phase 228 | Satisfied |
| PLAT-02 | Phase 228 | Satisfied |
| STG-04 | Phase 229 | Satisfied |
| STG-05 | Phase 229 | Satisfied |
| STG-06 | Phase 229 | Satisfied |
| STG-07 | Phase 230 | Satisfied |
| LIFE-02 | Phase 230 | Satisfied |
| SNR-01 | Phase 230 | Satisfied |
| ESG-01 | Phase 231 | Satisfied |
| ESG-02A | Phase 231 | Satisfied |
| ESG-02B | Deferred/Future (post-v1.26) | Deferred |
| ESG-03 | Phase 231 | Satisfied |
| ESG-04 | Phase 231 | Satisfied |
| TNX-01 | Phase 237 | Satisfied |
| TNX-02 | Phase 232 | Satisfied |
| TNX-03 | Phase 237 | Satisfied |
| TNX-04 | Phase 237 | Satisfied |
| PUB-01 | Phase 233 | Satisfied |
| PUB-02 | Phase 233 | Satisfied |
| PUB-03 | Phase 233 | Satisfied |
| PUB-04 | Phase 233 | Satisfied |
| PUB-05 | Phase 233 | Satisfied |
| TST-01 | Phase 237 | Satisfied |
| TST-02 | Phase 237 | Satisfied |
| GRD-01 | Phase 235 | Satisfied |
| GRD-02 | Phase 235 | Satisfied |
| GRD-03 | Phase 235 | Satisfied |
| GRD-04 | Phase 235 | Satisfied |
| GRD-05 | Phase 235 | Satisfied |
| DOC-01 | Phase 236 | Satisfied |
| LNT-01 | Phase 236 | Satisfied |
| BNH-01 | Phase 236 | Satisfied |
| EVI-01 | Phase 236 | Satisfied |

**Verification evidence — Phase 228 closeout (STG-02, STG-03, PLAT-02)**  
(manager-scoped run; do not extrapolate beyond this record):

- `scripts/quality_gates.sh` finished with exit **0**; exit snapshot:
  **`/tmp/emel_phase228_quality_gates_final.exit`** contains **`0`**.
- **`emel_tests_io`** passed during that gated run.
- Scoped **`staged_read` coverage reporting:** **91.9%** lines /
  **100.0%** branches (per gate output summarized by manager).

**Coverage:**

- v1.26 requirements: 35 total  
- Mapped to phases: 35  
- Satisfied: 34 active requirements  
- Pending gap closure: 0 active requirements  
- Deferred/future: 1 requirement (`ESG-02B`)  
- Unmapped: 0  
- Duplicate mappings: 0

---
*Requirements defined: 2026-05-07*  
*Last updated: 2026-05-08 Phase 237 source repair closed reopened requirement gaps*
