# Roadmap: EMEL

## Milestones

- ✅ **v1.0 EMEL Llama-68M Generation Slice** — shipped 2026-03-08
- ✅ **v1.1 EMEL Llama-68M Generation Benchmark** — shipped 2026-03-11
- ✅ **v1.2 Flash Attention** — shipped 2026-03-22
- ✅ **v1.3 ARM Flash Optimizations** — shipped 2026-03-22
- ✅ **v1.4 Full Vectorized Quantized Kernels** — shipped 2026-03-25
- ✅ **v1.5 Full ARM Quantized Path** — shipped 2026-03-27
- ✅ **v1.6 Qwen3-0.6B Parity And Benchmark** — shipped 2026-03-30
- ✅ **v1.7 Generator Prefill Submachine Decomposition** — shipped 2026-03-30
- ✅ **v1.8 Truthful Qwen3 E2E Embedded Size** — shipped 2026-04-02
- ✅ **v1.9 Liquid LFM2.5-1.2B Thinking ARM Slice** — shipped 2026-04-02
- ✅ **v1.11 TE-75M GGUF Trimodal Embedding Runtime** — shipped 2026-04-15
- ✅ **v1.12 Pluggable Reference Parity Bench Architecture** — shipped 2026-04-18
- ✅ **v1.13 Pluggable Generative Parity Bench** — shipped 2026-04-21
- ✅ **v1.14 Benchmark Variant Organization** — shipped 2026-04-21
- ✅ **v1.15 ARM Sortformer Diarization GGUF Slice** — shipped 2026-04-25
- ✅ **v1.16 ARM Whisper GGUF Parity And Performance** — shipped 2026-04-28
- ✅ **v1.17 Text Generator Domain Alignment** — shipped 2026-04-30
- ✅ **v1.18 Parity Tool Boundary Refactor** — shipped 2026-05-01
- ✅ **v1.19 Benchmark Tool Pluggable Runner Refactor** — shipped 2026-05-01
- ✅ **v1.20 SML Dependency And Namespace Migration** — shipped 2026-05-02
- ✅ **v1.21 Quality Gate Selective Runner Optimization** — shipped 2026-05-02
- ✅ **v1.22 Weight Loading Ownership Cutover** — shipped 2026-05-03
- ✅ **v1.23 I/O Loading Strategy Boundary** — shipped 2026-05-04
- ✅ **v1.24 I/O Mmap Loading Strategy** — shipped 2026-05-04 (Phases 204-211)
- ✅ **v1.25 I/O Read Loading Strategy** — shipped 2026-05-06 (Phases 212-226 + 214.1)
- ✅ **v1.26 I/O Staged Read Loading Strategy** — completed 2026-05-08
  (12 / 12 phases complete; issue #63; `ESG-02B` deferred/future)

## Phases

### ✅ v1.26 I/O Staged Read Loading Strategy (Phases 227-238) — COMPLETE 2026-05-08

Source: GitHub issue #63, "Add io/staged_read state machine for constrained-memory tensor loading".
Adds `src/emel/io/staged_read` for bounded chunked/windowed reads under tensor-owned residency.
Depends on the tensor-to-I/O boundary from issue #60. Cooperative coroutine scheduling is out of
scope unless separately approved. Shipped mmap (`io/mmap`) and bulk read/copy (`io/read`) must not
regress.

Execution order: 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238.

**Milestone progress (v1.26):** **12 / 12** phases recorded **Complete** in the table below.
The source-backed milestone audit found a direct tensor staged-load offset-contract gap plus
closeout artifact debt; Phases 237-238 closed those gaps. `ESG-02B` remains deferred/future
because file-backed staged-read source ownership is out of scope.

- [x] Phase 227: Staged Read Strategy Component Boundary (STG-01)
- [x] Phase 228: Span, Target-Window, and Platform Gating (STG-02, STG-03, PLAT-02)
- [x] Phase 229: Staged Copy Progress and Completion Semantics (STG-04, STG-05, STG-06)
- [x] Phase 230: Context Cleanness and Per-Attempt Lifetime (STG-07, LIFE-02, SNR-01)
- [x] Phase 231: Deterministic Error Taxonomy (ESG-01, ESG-02A, ESG-03, ESG-04; ESG-02B deferred)
- [x] Phase 232: Tensor-Owned Integration Graph (TNX-01, TNX-02, TNX-03, TNX-04)
- [x] Phase 233: Public Loader and Maintained Entrypoints (PUB-01, PUB-02, PUB-03, PUB-04, PUB-05)
- [x] Phase 234: Public Dispatch Tests (TST-01, TST-02)
- [x] Phase 235: Scope and Non-Regression Guardrails (GRD-01, GRD-02, GRD-03, GRD-04, GRD-05)
- [x] Phase 236: Publication and Evidence Truthfulness (DOC-01, LNT-01, BNH-01, EVI-01)
- [x] Phase 237: Direct Tensor Staged Offset Contract Repair (TNX-01, TNX-03, TNX-04, TST-01, TST-02)
- [x] Phase 238: Audit Artifact and Probe Reporting Cleanup (cleanup-only)

#### Phase 227: Staged Read Strategy Component Boundary

**Goal:** Locate canonical `src/emel/io/staged_read` with standard I/O component layout.
**Depends on:** Phase 226
**Requirements:** STG-01

**Success criteria:**

1. `src/emel/io/staged_read` exists with canonical `emel::io::staged_read::sm` alias.
2. Component scope excludes mmap, device transfer, or cooperative async runtime.
3. Initial fail-closed or smoke dispatch proves actors are wired like sibling I/O strategies.

#### Phase 228: Span, Target-Window, and Platform Gating

**Goal:** All staged preconditions enforced in guards/transitions before any file work.
**Depends on:** Phase 227
**Requirements:** STG-02, STG-03, PLAT-02

**Success criteria:**

1. Invalid source staging contract rejected solely via guard-modeled transitions.
2. Invalid target window/layout rejected solely via guard-modeled transitions.
3. Unsupported hosts/resources fail closed with explicit unsupported terminal shape.

#### Phase 229: Staged Copy Progress and Completion Semantics

**Goal:** Prove per-stage deterministic copy plus full-span monotone completion.
**Depends on:** Phase 228
**Requirements:** STG-04, STG-05, STG-06

**Success criteria:**

1. Test vectors observe correct bytes per staged window.
2. Completeness tests cover entire logical span order.
3. Terminal success aligns with copied full span per contract.

#### Phase 230: Context Cleanness and Per-Attempt Lifetime

**Goal:** Bounded handles and residency clarity for the staged actor.
**Depends on:** Phase 229
**Requirements:** STG-07, LIFE-02, SNR-01

**Success criteria:**

1. Static or dynamic review shows zero forbidden dispatch-local context mirrors.
2. Handle lifetime tests/tools show release-before-done semantics.
3. Tests confirm strategy never asserts tensor residency commits.

#### Phase 231: Deterministic Error Taxonomy

**Goal:** Errors are categorical, observable, exception-free.
**Depends on:** Phase 230
**Requirements:** ESG-01, ESG-02A, ESG-03, ESG-04 (`ESG-02B` deferred)

**Success criteria:**

1. At least one doctest per taxonomy family (pre-I/O guard, source-contract read-surface, sequencing/contract) demonstrates deterministic categories through `process_event(...)`.
2. Source-backed docs explicitly defer `ESG-02B` file open/seek/read + per-stage short-read categories until approved file-backed staged-read ownership exists.
3. ABI boundary scans show noexcept expectations for surfaced API.

#### Phase 232: Tensor-Owned Integration Graph

**Goal:** Integrate staged loads through explicit tensor+I/O graphs.
**Depends on:** Phase 231
**Requirements:** TNX-01, TNX-02, TNX-03, TNX-04

**Closeout ledger (verified):** Manager-scoped **`scripts/quality_gates.sh`** for Phase 232
changed-file corpus exited **2** (red — **not** exit 0). **`232-VERIFICATION.md`** records **bench_snapshot**
suite regressions unrelated to staged tensor-integration files and a **paritychecker** failure outside
Phase 232 scope. Phase 232 completion is ledger-approved **without** claiming a passing full-repo gate run.

**Success criteria:**

1. Requests flow only via public tensors↔IO events.
2. Residency proofs remain tensor-owned (`model/tensor` retains lifecycle ownership).
3. Success/failure each have explicit observable terminal representations.

#### Phase 233: Public Loader and Maintained Entrypoints

**Goal:** Strategies observable without actor detail reach-through or duplicate POSIX loops in tools.
**Depends on:** Phase 232
**Requirements:** PUB-01, PUB-02, PUB-03, PUB-04, PUB-05

**Closeout (2026-05-08):** **`PUB-01`–`PUB-05`** satisfied per **`233-VERIFICATION.md`** (manager validation +
**phase233-navigator final review PASS**). Public **`staged_read`** access is through **`io::loader`** and maintained
tool entrypoints with **`io_staged_read`** wiring; **`tests/model/loader/lifecycle_tests.cpp`** covers the
storage-backed **`staged_read`** route and include guards.

**Residual:** **`scripts/quality_gates.sh`** was **not** run on a Phase **233** changed-file corpus in
this closeout slice — **no Phase 233 scoped gate pass is claimed** (full-repo gate truth unchanged from
Phase **232** ledger where applicable).

**Success criteria:**

1–4. Each lane (loader/bench/parity/probe) has independent proof of public-contract-only access.
5. Source scan enforcement or doctest proves no duplicated unconstrained staged read shim in tools.

#### Phase 234: Public Dispatch Tests

**Goal:** Core success/failure behavior demonstrated through `process_event`.
**Depends on:** Phase 233
**Requirements:** TST-01, TST-02

**Success criteria:**

1. Passing success-path doctest with `visit_current_states` or equivalent.
2. Passing failure-path doctest for guard rejection.

#### Phase 235: Scope and Non-Regression Guardrails

**Goal:** Freeze architecture invariants relative to loaders, mmap, and read strategies.
**Depends on:** Phase 234
**Requirements:** GRD-01, GRD-02, GRD-03, GRD-04, GRD-05

**Success criteria:** Each of GRD-01, GRD-02, GRD-03, GRD-04, and GRD-05 has either a deterministic script failure mode or a narrowed regression doctest proving the invariant holds.

#### Phase 236: Publication and Evidence Truthfulness

**Goal:** Align docs and frozen artifacts with real staged/runtime usage.
**Depends on:** Phase 235
**Requirements:** DOC-01, LNT-01, BNH-01, EVI-01

**Success criteria:**

1. Doc diff review verifies accurate staged-read wording.
2. Lint snapshot regeneration path documented/passing.
3. Benchmark snapshot regeneration obeys policy.
4. Parity/compare metadata never mislabels unstaged workloads as staged.

**Closeout (2026-05-08):** **`DOC-01`–`EVI-01`** satisfied per
**`236-VERIFICATION.md`**. Serial full quality gate passed:
`EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_PARALLEL=0 scripts/quality_gates.sh`
(exit **0**, ended `2026-05-08T21:21:42.028Z`). Benchmark defaults now use bounded routine
settings (`100` iterations, `3` runs, `10` warmup iterations) with bounded generation and
diarization defaults.

#### Phase 237: Direct Tensor Staged Offset Contract Repair

**Goal:** Repair direct `model/tensor` staged-load nonzero-offset source-window behavior and prove it through public dispatch.
**Depends on:** Phase 236
**Requirements:** TNX-01, TNX-03, TNX-04, TST-01, TST-02
**Gap Closure:** Closes `.planning/v1.26-MILESTONE-AUDIT.md` findings
`direct-tensor-staged-offset-contract` and `direct-tensor-staged-nonzero-offset`.

**Success Criteria:**

1. A public `model/tensor::event::request_staged_load` doctest fails before repair and passes after
   repair for a nonzero `file_offset` against a whole-file source buffer.
2. Direct tensor staged-load source-span construction is aligned with `io/loader` or the
   pre-windowed-source contract is explicitly documented and enforced by validation/tests.
3. Direct tensor staged-load success and failure outcomes remain explicit `_done` / `_error`
   publications through public `process_event(...)` dispatch and SML state inspection.
4. Changed-file quality gates for `model/tensor`, `io/staged_read`, and affected tests pass without
   benchmark-regression override.
5. If implementation changes maintained model or snapshot artifacts, those artifacts are refreshed
   only through maintained workflows; model artifact updates are approved for this gap-closure work.

**Closeout (2026-05-08):** Phase 237 completed with a failing-first public
`request_staged_load` nonzero-offset doctest, repaired source-window dispatch in
`model/tensor`, and passing scoped validation:
`./build/emel_tests_bin --test-case="model_tensor_request_staged_load_*"`,
`ctest --test-dir build -R '^emel_tests_model_and_batch$' --output-on-failure`,
and changed-file `scripts/quality_gates.sh` (exit `0`). Reopened requirements
`TNX-01`, `TNX-03`, `TNX-04`, `TST-01`, and `TST-02` are satisfied by
`237-VERIFICATION.md`.

#### Phase 238: Audit Artifact and Probe Reporting Cleanup

**Goal:** Reconcile audit artifacts and probe reporting truth after the Phase 237 source repair.
**Depends on:** Phase 237
**Requirements:** none — cleanup-only; all reopened requirement closure belongs to Phase 237
**Gap Closure:** Closes `.planning/v1.26-MILESTONE-AUDIT.md` tech-debt items for missing
`requirements-completed` SUMMARY frontmatter and embedded-size probe reporting clarity.

**Success Criteria:**

1. Phase summaries for 232–236 expose accurate `requirements-completed` frontmatter or an explicit
   cleanup rationale so the three-source audit matrix no longer needs manual reconciliation.
2. Embedded-size probe evidence either prints the executed load strategy when appropriate or the
   maintained docs/audit explain why captured `used_io_strategy` is the authoritative evidence
   surface.
3. REQUIREMENTS, ROADMAP, STATE, and the milestone audit are refreshed from source-backed evidence
   after Phase 237.
4. Focused lint/docs/audit commands pass; no maintained benchmark, model, or snapshot artifact is
   updated unless the implementation actually requires it.

**Closeout (2026-05-08):** Phase 238 completed summary frontmatter reconciliation,
embedded probe reporting truth documentation, and refreshed `v1.26-MILESTONE-AUDIT.md`
to `status: passed`. Changed-file `scripts/quality_gates.sh` passed with no benchmark,
coverage, parity, fuzz, or docsgen-affecting lanes required.

---
### ✅ v1.25 I/O Read Loading Strategy (Phases 212-226 + 214.1) — SHIPPED 2026-05-06

Source: GitHub issue #62, "Add io/read state machine for copy-based tensor loading".
Adds a dedicated `src/emel/io/read` Stateforward.SML actor for explicit read/copy tensor
loading beneath tensor-owned residency. Mmap, staged/chunked constrained-memory, async,
and device strategies remain out of scope.

- [x] Phase 212: Read Strategy Component Boundary (1/1 plans) — completed 2026-05-05
- [x] Phase 213: Read Validation and Platform Gating (1/1 plans) — completed 2026-05-05
- [x] Phase 214: Read Execution, Errors, and Lifetime (1/1 plans) — completed 2026-05-05; audit found RTC compliance gap
- [x] Phase 214.1: RTC-Safe Read Execution Boundary Repair (1/1 plans) — gap closure
- [x] Phase 215: Tensor-Owned Read Integration (1/1 plans) — completed 2026-05-05
- [x] Phase 216: Public Runtime and Evidence Surfaces (1/1 plans) — completed 2026-05-05
- [x] Phase 217: Behavior Tests and Scope Guardrails (1/1 plans) — completed 2026-05-05
- [x] Phase 218: Publication and Maintained Artifact Updates (1/1 plans) — completed 2026-05-05
- [x] Phase 219: Maintained Read Source Provenance (1/1 plans) — completed
  2026-05-05; source-backed benchmark/parity/probe read_copy provenance
- [x] Phase 220: Explicit Tensor Read Outcome Graph (1/1 plans) — completed
  2026-05-05; tensor read outcomes selected by explicit same-RTC result graph
- [x] Phase 221: Read Closeout Truth Reconciliation — superseded planning stub
  closed 2026-05-06; Phase 223 owns final closeout
- [x] Phase 222: Public Read Source Contract Repair (1/1 plans) — completed
  2026-05-06; actor-detail reach-through removed from maintained lanes
- [x] Phase 223: Read Closeout Truth And Validation Reconciliation (1/1 plans) —
  completed 2026-05-06; final closeout truth and validation reconciled
- [x] Phase 224: Read Closeout Tech Debt Cleanup — completed 2026-05-06;
  refreshed audit ambiguity closed with fresh passing `emel_tests_io` evidence
- [x] Phase 225: Read Closeout Runtime Validation And SML Repair — completed
  2026-05-06; refreshed source-backed audit gaps closed with dyld fallback evidence
- [x] Phase 226: Read Batch Cap And Closeout Evidence Refresh — completed
  2026-05-06; refreshed audit tech debt closed

Archived closeout artifacts:
- `.planning/milestones/v1.25-ROADMAP.md`
- `.planning/milestones/v1.25-REQUIREMENTS.md`
- `.planning/milestones/v1.25-MILESTONE-AUDIT.md`
- `.planning/milestones/v1.25-phases/`

**Execution Order:** Phases execute in numeric order:
212 -> 213 -> 214 -> 214.1 -> 215 -> 216 -> 217 -> 218 -> 219 -> 220 -> 222 -> 223 -> 224 -> 225 -> 226.
Phase 221 is a completed superseded closeout planning stub and Phase 223 owns final
source-backed closeout truth. Phase 224 is cleanup-only; Phase 225 owns the refreshed
2026-05-06 audit gaps before archive. Phase 226 closes the post-audit nonblocking
tech-debt items before final closeout.

#### Phase 212: Read Strategy Component Boundary
**Goal**: Maintainers can identify `io/read` as the canonical read/copy strategy actor under
`src/emel/io`.
**Depends on**: Phase 211
**Requirements**: READ-01
**Success Criteria** (what must be TRUE):
  1. Maintainer can inspect `src/emel/io/read` and find component-local `context`, `events`,
     `guards`, `actions`, `errors`, and `sm` ownership.
  2. Maintainer can use canonical `emel::io::read::sm` ownership and public aliases without
     reaching into actor internals.
  3. Maintainer can confirm the component is read/copy-only and contains no mmap, staged or
     chunked constrained-memory, cooperative async, device-specific, loader-owned byte access,
     model-family widening, or tool-only read scaffold behavior.
**Plans**: 01 — Validated 2026-05-05; established canonical `io/read` boundary actor
and lifecycle tests.

#### Phase 213: Read Validation and Platform Gating
**Goal**: The read actor accepts read attempts only after explicit request, platform, file,
offset, length, layout, and target-buffer preconditions pass.
**Depends on**: Phase 212
**Requirements**: READ-02, PLAT-01
**Success Criteria** (what must be TRUE):
  1. Caller sees invalid request, file, offset, length, layout, or target-buffer preconditions
     rejected before any open or read attempt is accepted.
  2. Caller sees unsupported platforms and unsupported file/resource shapes fail closed
     deterministically through the I/O abstraction boundary.
  3. Maintainer can inspect SML guards and transitions and see validation outcomes modeled
     before the open/read attempt.
  4. Supported requests reach a read-attempt state only after all read preconditions are true.
**Plans**: 01 — Validated 2026-05-05; added explicit read validation and platform
gating before the read-attempt placeholder.

#### Phase 214: Read Execution, Errors, and Lifetime
**Goal**: Successful read requests deliver deterministic copied bytes into the caller-owned
target buffer with deterministic transient-resource lifetime and deterministic failure
outcomes, without taking tensor residency ownership.
**Depends on**: Phase 213
**Requirements**: READ-03, LIFE-01, ERR-01
**Success Criteria** (what must be TRUE):
  1. Caller receives a deterministic copied-bytes outcome on success with the requested byte
     span written into the caller-provided owned target buffer; the read strategy never claims
     residency ownership.
  2. Read failures surface deterministic error categories (invalid request, unsupported
     resource, unsupported platform, file open failed, file seek failed, file read failed,
     short read, internal error) instead of thrown exceptions or ambiguous status mirroring.
  3. Transient OS resources (file descriptor / handle) are released through the actor-owned
     attempt before `_done` is published; no kernel handle is held across publication.
  4. Maintainer can verify dispatch-local request data is not stored in `read::context` and
     tensor residency semantics remain owned by `model/tensor`.
**Plans**: 01 — Validated 2026-05-05; added concrete read execution, copied-byte
success, deterministic read errors, and close-before-done lifetime behavior.
2026-05-05 milestone audit found this phase superseded by unverified Phase 214.1 repair
work; Phase 214.1 owns source-backed RTC validation and artifact reconciliation.

#### Phase 214.1: RTC-Safe Read Execution Boundary Repair
**Goal**: The read actor preserves copied-byte success, deterministic errors, and
close-before-done lifetime evidence without performing blocking or input-dependent
filesystem work inside SML dispatch.
**Depends on**: Phase 214
**Requirements**: READ-03, PLAT-01, LIFE-01, ERR-01
**Gap Closure**: Closes v1.25 audit gaps for missing Phase 214.1 artifacts, stale Phase
214 planning truth, and source-backed Nyquist validity after the read actor moved to
caller-provided source spans.
**Success Criteria** (what must be TRUE):
  1. `src/emel/io/read` no longer calls platform open, seek, read, close, or equivalent
     filesystem APIs from guards, actions, transition helpers, or functions called by them.
  2. The read actor still accepts only validated read/copy attempts and publishes copied-byte
     `_done` plus deterministic `_error` outcomes through explicit states/events.
  3. The caller-owned target buffer remains caller-owned, dispatch-local request data is not
     stored in `read::context`, and no transient OS handle is retained or hidden in context.
  4. Tests prove the repaired behavior through public `process_event(...)` dispatch and SML
     state inspection, including validation failure, unsupported/resource failure, read
     failure, short read, and copied-byte success.
  5. Phase 214.1 SUMMARY.md, VERIFICATION.md, and VALIDATION.md reconcile ROADMAP.md,
     STATE.md, REQUIREMENTS.md, and generated architecture docs with the source-buffer based
     implementation and do not claim maintained benchmark/parity evidence.
**Plans**: 01 — Validated 2026-05-05; reconciled read actor evidence with the
source-buffer based implementation, confirmed no dispatch-time filesystem calls, and
updated requirement/state artifacts for the Phase 214.1 gap closure.

#### Phase 215: Tensor-Owned Read Integration
**Goal**: `model/tensor` can request and consume read-backed I/O through the public `emel/io`
boundary while retaining load, bind, evict, and residency orchestration.
**Depends on**: Phase 214.1
**Requirements**: TIO-01, TIO-02
**Gap Closure**: Closes v1.25 audit gaps for partial tensor-owned read integration and
callback/status-mediated read outcomes.
**Success Criteria** (what must be TRUE):
  1. Tensor load flow can request read-based (copy) loading through public `emel/io` events
     without direct low-level read calls.
  2. Tensor bind, residency, and evict transitions remain in `model/tensor` and consume read
     success outcomes that reference the caller-owned target buffer.
  3. Read success, unsupported, validation failure, file open failure, and file read failure
     are visible as explicit `_done` and `_error` events or states.
  4. Maintainer can verify no callback-selected outcomes, mirrored status fields, or context
     phase flags decide tensor-to-I/O outcomes for read-backed loading.
  5. Existing source/test progress through `model/loader -> model/tensor -> io/loader ->
     io/read -> tensor apply` is preserved or replaced by a stricter explicit outcome path
     with equivalent public-dispatch tests.
**Plans**: 01 — Validated 2026-05-05; added tensor-owned
`request_read_load` public events, explicit read outcome states, and tests for read
success, unsupported I/O actor, validation failure, file open failure, and file read
failure.

#### Phase 216: Public Runtime and Evidence Surfaces
**Goal**: Runtime entrypoints and maintained tool lanes can select or report read-backed
loading only through public surfaces, and evidence reflects the actual EMEL runtime path.
**Depends on**: Phase 215
**Requirements**: TIO-03, VAL-04
**Gap Closure**: Closes v1.25 audit gaps for maintained benchmark, paritychecker, and
embedded probe lanes bypassing the read-backed runtime path and for runtime reporting that
currently exposes only mmap usage.
**Success Criteria** (what must be TRUE):
  1. `model/loader`, maintained benchmark lanes, paritychecker lanes, and embedded probes can
     select or report read-backed loading only through public tensor and I/O runtime contracts.
  2. Maintained benchmark, paritychecker, and embedded probe lanes avoid actor-internal
     reach-through and contain no low-level read logic.
  3. Benchmark and parity output reports read-strategy usage only when the EMEL lane executed
     the read-backed runtime path.
  4. Unsupported or fallback behavior is reported as unsupported or non-read-strategy, not as
     read-strategy parity or performance evidence.
  5. Runtime done/error evidence distinguishes mmap, read/copy, unsupported, and non-I/O
     loading paths without relying on tool-only scaffolds.
**Plans**: 01 — Validated 2026-05-05; added public model-loader load-strategy
evidence, maintained tool strategy binding, load-strategy output notes, and
source-backed tests proving benchmark/parity/embedded lanes avoid callback-time
actor reach-through.

#### Phase 217: Behavior Tests and Scope Guardrails
**Goal**: Tests and guardrails prove read behavior through public dispatch and prevent scope
or ownership leaks.
**Depends on**: Phase 216
**Requirements**: VAL-01, VAL-02
**Gap Closure**: Closes v1.25 audit gaps for missing full-scope read behavior tests,
domain/source guardrails, and former ambiguous read-strategy naming relative to the
out-of-scope staged/chunked policy.
**Success Criteria** (what must be TRUE):
  1. Doctests drive supported read behavior through `process_event(...)` and inspect SML states
     via `visit_current_states` and/or `is(...)`.
  2. Doctests cover representative unsupported, validation failure, file open failure, and file
     read failure outcomes through public events.
  3. Guardrails fail if read implementation leaks into `model/loader` or tensor residency
     ownership moves out of `model/tensor`.
  4. Guardrails fail if mmap, staged or chunked constrained-memory, cooperative async,
     device-specific, model-family widening, loader-owned byte access, or tool-only read
     scaffold behavior enters this milestone.
  5. Source guardrails clarify or eliminate any public naming that could present the v1.25
     read/copy path as staged/chunked constrained-memory support.
**Plans**: 01 — Validated 2026-05-05; renamed the copy strategy to
`read_copy`, added public-dispatch behavior guardrails, tensor-residency ownership
guardrails, and maintained tool/model-loader no-reach-through source checks.

#### Phase 218: Publication and Maintained Artifact Updates
**Goal**: Maintained docs, snapshots, benchmark outputs, model artifacts, and planning truth
describe read-strategy support exactly as implemented.
**Depends on**: Phase 217
**Requirements**: VAL-03
**Gap Closure**: Closes v1.25 audit gaps for stale planning truth, stale generated docs,
and missing maintained artifact updates. User approved updating snapshots, benchmarks, and
models as needed during this gap closure command.
**Success Criteria** (what must be TRUE):
  1. Public docs and generated architecture docs describe the read/copy strategy path,
     ownership boundaries, and deferred strategies (mmap shipped in v1.24; staged/async/device
     remain deferred) truthfully.
  2. Lint snapshots, benchmark snapshots, benchmark outputs, and model artifacts are updated
     from maintained commands when the implementation changes them.
  3. Planning artifacts record final requirement coverage, validation evidence, and any
     approved artifact updates for v1.25.
  4. Closeout artifacts do not claim read-strategy support beyond source-backed maintained
     runtime behavior.
  5. Any snapshot, benchmark, or model artifact changes are produced by maintained commands
     and explicitly tied to source-backed read/copy runtime behavior.
**Plans**: 01 — Validated 2026-05-05; updated public docs, README template,
generated architecture docs, benchmark snapshots, planning truth, and final closeout
audit from maintained commands. The closing full quality gate passed with
`EMEL_QUALITY_GATES_SCOPE=full EMEL_QUALITY_GATES_PARALLEL=never
scripts/quality_gates.sh`.

#### Phase 219: Maintained Read Source Provenance
**Goal**: Maintained benchmark, paritychecker, and embedded probe lanes prove read/copy
strategy usage from a maintained `src`-owned source contract instead of tool-local full-file
read scaffolds.
**Depends on**: Phase 218
**Requirements**: PLAT-01, TIO-03, VAL-04
**Gap Closure**: Closes v1.25 audit gaps where generation, Sortformer diarization,
embedded probe, and paritychecker lanes report `read_copy` after tool-local
`read_file_bytes` helpers create the source span.
**Success Criteria** (what must be TRUE):
  1. Maintained benchmark, paritychecker, and embedded probe lanes no longer own low-level
     file slurp helpers as the source of `read_copy` evidence.
  2. A maintained `src`-owned loading/source contract feeds `model/loader -> model/tensor ->
     io/loader -> io/read` for read/copy tool evidence.
  3. `read_copy` benchmark/parity/probe output is emitted only when the EMEL lane actually
     consumed the maintained source contract and executed the public runtime path.
  4. Unsupported or fallback source behavior is reported as unsupported or non-read-strategy,
     never as read-strategy parity or performance evidence.
  5. Tests and source guardrails fail on tool-local substitutes for the maintained read/copy
     source path.

#### Phase 220: Explicit Tensor Read Outcome Graph
**Goal**: Tensor-owned read/copy integration exposes success and failure outcomes through
explicit state/event routing without callback/status-mediated behavior selection.
**Depends on**: Phase 219
**Requirements**: TIO-02
**Gap Closure**: Closes v1.25 audit gap where `model/tensor` represents final outcomes
with explicit states/events but still uses callback-mutated runtime status inspected by
guards to select the read outcome path.
**Success Criteria** (what must be TRUE):
  1. `model/tensor` read success, unsupported, validation failure, file open failure, and
     file read failure outcomes are selected by explicit guards/transitions over typed
     same-RTC events, not by callback-mutated status fields.
  2. Any same-RTC callbacks used for immediate replies do not decide which tensor outcome
     path runs.
  3. No mirrored status fields, context phase flags, or callback-selected outcomes remain in
     the read-backed tensor outcome path.
  4. Public doctests prove all representative read success and error outcomes through
     `process_event(...)` and SML state inspection.

#### Phase 221: Read Closeout Truth Reconciliation
**Goal**: Maintained docs, generated architecture docs, planning artifacts, snapshots,
benchmark outputs, model artifacts, and the milestone audit describe read/copy support
exactly as implemented after gap closure.
**Depends on**: Phase 220
**Requirements**: superseded by Phase 223
**Gap Closure**: Closes v1.25 audit gap where closeout artifacts overstated maintained
read/copy path truth while tool-local source spans still fed the reported lane. User
approved updating model artifacts, snapshots, and benchmarks as needed during this gap
closure command.
**Success Criteria** (what must be TRUE):
  1. Public docs, generated architecture docs, ROADMAP, REQUIREMENTS, STATE, PROJECT,
     MILESTONES, and the milestone audit describe the maintained read/copy path truthfully.
  2. Lint snapshots, benchmark snapshots, benchmark outputs, and model artifacts are updated
     from maintained commands when implementation changes require it.
  3. Phase 214 historical artifacts are reconciled or explicitly marked superseded so they no
     longer conflict with the Phase 214.1 source-buffer truth.
  4. A source-backed milestone audit passes without relying on tool-only source scaffolds.
  5. The closing quality gate is run with the appropriate full or changed-file scope and no
     benchmark-regression override unless explicitly documented as transitional.
**Plans**: 01 — Ready only. 2026-05-06 audit found an additional source-contract
blocker in Phase 219/216 maintained lanes, so Phase 221 is superseded by the
Phase 222 source-contract repair and Phase 223 closeout truth plan.
**Summary**: Superseded 2026-05-06 with no source or requirement claims.

#### Phase 222: Public Read Source Contract Repair
**Goal**: Maintained benchmark, paritychecker, and embedded probe lanes obtain read/copy
source bytes through an allowed public or non-actor-internal EMEL-owned contract instead of
including `emel/io/read/detail.hpp`.
**Depends on**: Phase 220
**Requirements**: PLAT-01, TIO-03, VAL-02, VAL-04
**Gap Closure**: Closes v1.25 audit gaps where maintained lanes replaced tool-local
`read_file_bytes` helpers with direct actor-detail reach-through, causing paritychecker
guardrails and maintained read/copy evidence to fail.
**Success Criteria** (what must be TRUE):
  1. Maintained generation, Sortformer diarization, embedded probe, and paritychecker lanes
     no longer include or call `emel/io/read/detail.hpp` or any actor `detail.hpp` helper for
     benchmark/parity source loading.
  2. Source-byte loading for maintained read/copy evidence is exposed through an allowed
     EMEL-owned public/runtime/setup contract that does not violate the actor model,
     benchmark/parity harness rules, or `detail.hpp` ownership rules.
  3. Maintained lanes still report `read_copy` only when the EMEL lane executes the public
     `model/loader -> model/tensor -> io/loader -> io/read` runtime path.
  4. Guardrails fail on actor-internal reach-through, tool-local read substitutes, and any
     unsupported fallback reported as read/copy evidence.
  5. Focused paritychecker and maintained generation evidence passes without benchmark
     regression override.
**Plans**: 01 — Validated 2026-05-06; moved maintained source-byte loading to
`emel::io::source::load_file_bytes`, removed `io/read/detail.hpp` reach-through
from maintained lanes, and restored paritychecker/generation guardrail evidence.

#### Phase 223: Read Closeout Truth And Validation Reconciliation
**Goal**: Final v1.25 closeout truth, generated artifacts, snapshots, benchmark outputs,
model artifacts, requirements, roadmap state, and milestone audit reflect the post-Phase 222
maintained read/copy runtime path.
**Depends on**: Phase 222
**Requirements**: TIO-02, VAL-01, VAL-03
**Gap Closure**: Closes v1.25 audit gaps for stale Phase 220 roadmap state, unvalidated
Phase 221/VAL-03 closeout truth, dyld-blocked test rerun evidence, and final source-backed
milestone audit truth.
**Success Criteria** (what must be TRUE):
  1. ROADMAP, REQUIREMENTS, STATE, PROJECT, MILESTONES, public docs, generated architecture
     docs, and the milestone audit no longer claim stale Phase 218/221 closeout truth.
  2. Phase 220 progress-table state is reconciled with its completed SUMMARY,
     VERIFICATION, and VALIDATION artifacts.
  3. Public behavior doctests and maintained guardrails are rerun or the dyld/libSystem launch
     blocker is explicitly captured with source-backed substitute evidence approved for the
     phase.
  4. Lint snapshots, benchmark snapshots, benchmark outputs, and model artifacts are updated
     only through maintained commands when the repaired implementation changes them.
  5. A source-backed milestone audit reports every active v1.25 requirement satisfied, with
     no actor-detail reach-through or tool-only maintained-path evidence.
**Plans**: 01 — Validated 2026-05-06; reconciled final planning truth, generated
docs checks, lint snapshot checks, public-dispatch doctests, paritychecker
guardrails, maintained generation compare evidence, repaired batch planner
benchmark evidence, the full closeout quality gate, and the source-backed
milestone audit.

#### Phase 224: Read Closeout Tech Debt Cleanup
**Goal**: Close the nonblocking tech-debt items from the refreshed v1.25 milestone audit
before archive.
**Depends on**: Phase 223
**Requirements**: none — all v1.25 requirements remain satisfied; this phase is cleanup only
**Gap Closure**: Addresses audit tech debt without resetting any validated requirement:
historical Phase 214 supersession noise, public tensor read event maintained-lane coverage shape,
and fresh `emel_tests_io` evidence after the local dyld/libSystem launch blocker is resolved.
**Success Criteria** (what must be TRUE):
  1. Phase 214 historical artifacts are either further reconciled or explicitly confirmed as
     intentionally superseded by Phase 214.1 without creating closeout ambiguity.
  2. Maintainers can tell whether `model::tensor::event::request_read_load` should gain a
     maintained direct-lane coverage path or remain a public tested route while maintained
     model-loader lanes use `model/tensor` plan/apply plus `io/loader -> io/read`.
  3. Fresh `emel_tests_io` evidence is captured from a healthy local environment, or the
     dyld/libSystem launch blocker is captured with an explicit archive-time decision.
  4. The milestone audit is rerun and either passes or reports only explicitly accepted
     nonblocking debt.
**Plans**: 01 — Validated 2026-05-06; Phase 214 supersession clarity,
`request_read_load` maintained-lane decision evidence, fresh passing
`emel_tests_io` evidence, and final milestone audit refresh.

#### Phase 225: Read Closeout Runtime Validation And SML Repair
**Goal**: Close refreshed v1.25 audit gaps by restoring executable model/batch validation,
moving maintained read/copy per-tensor I/O orchestration out of model-loader action loops,
and reconciling closeout artifact paths.
**Depends on**: Phase 224
**Requirements**: VAL-01, TIO-03, VAL-04, VAL-03
**Gap Closure**: Closes `.planning/v1.25-MILESTONE-AUDIT.md` findings: current
`emel_tests_model_and_batch` dyld launch failure, model-loader action-loop
`io_loader->process_event(...)` SML readiness risk, and stale archived closeout path
references.
**Success Criteria** (what must be TRUE):
  1. `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch`
     runs to completion or the dyld/libSystem launch blocker is eliminated with a
     source-backed maintained substitute explicitly recorded in validation.
  2. Maintained read/copy `model/loader -> io/loader` orchestration no longer relies on an
     action loop calling `io_loader->process_event(...)`; runtime choice and per-phase
     orchestration are represented with explicit SML guards/states/transitions.
  3. The maintained read/copy path still reports `used_io_strategy` only after public
     runtime execution through `model/loader -> model/tensor -> io/loader -> io/read`.
  4. Closeout artifact paths in active and archived roadmap/requirements/audit docs point
     at files that exist after the v1.25 archive layout.
  5. Focused model-loader, model/tensor, io/loader, io/read, domain-boundary, consistency,
     and changed-file quality gates pass without benchmark-regression override.
**Plans**: 6 plans — completed 2026-05-06
Plans:
- [x] `225-01-PLAN.md` — Add the owning `io/read` batch copy surface and public-dispatch tests.
- [x] `225-02-PLAN.md` — Route one `io/loader` read_copy batch to `io/read` with same-RTC result callbacks.
- [x] `225-03-PLAN.md` — Replace model-loader per-tensor I/O dispatch with one public batch dispatch.
- [x] `225-04-PLAN.md` — Wire maintained callers and guardrails to request-owned `io_load_spans`.
- [x] `225-05-PLAN.md` — Reconcile active and archived closeout path and plan traceability.
- [x] `225-06-PLAN.md` — Publish validation, summary, and active/archived audit evidence.

#### Phase 226: Read Batch Cap And Closeout Evidence Refresh
**Goal**: Close the nonblocking tech-debt items from `.planning/v1.25-MILESTONE-AUDIT.md`
by bounding the public read/copy batch API independently and refreshing closeout evidence
to match current executable validation.
**Depends on**: Phase 225
**Requirements**: none — all v1.25 requirements remain satisfied; this phase is cleanup only
**Gap Closure**: Closes audit tech debt for the uncapped public `io/read`
`read_tensor_batch` span and stale dyld-fallback closeout wording after current focused
CTest passed.
**Success Criteria** (what must be TRUE):
  1. Public `io/read::event::read_tensor_batch` dispatch rejects over-large spans before
     iterating or copying, with the cap owned by a public/read-side contract rather than
     relying only on maintained model-loader callers.
  2. Doctests prove accepted boundary-size batches and rejected over-large batches through
     public `process_event(...)` dispatch and SML state inspection.
  3. Active and archived closeout evidence distinguishes historical dyld fallback evidence
     from current direct `build/zig` focused CTest evidence.
  4. If the repaired implementation changes maintained snapshots, benchmark outputs,
     benchmark snapshots, or model artifacts, those artifacts are updated only through
     maintained commands. User permission for those updates was granted with this phase.
  5. Changed-file quality gates pass without benchmark-regression override, and the
     refreshed milestone audit reports no blockers.
**Plans**: 01 — Validated 2026-05-06; public `io/read` batch cap added,
exact-cap and over-cap doctests passed, closeout evidence refreshed, and
changed-file quality gate passed.

#### Coverage

| Requirement | Phase |
|-------------|-------|
| READ-01 | Phase 212 |
| READ-02 | Phase 213 |
| PLAT-01 | Phase 222 |
| READ-03 | Phase 214.1 |
| LIFE-01 | Phase 214.1 |
| ERR-01 | Phase 214.1 |
| TIO-01 | Phase 215 |
| TIO-02 | Phase 223 |
| TIO-03 | Phase 225 |
| VAL-04 | Phase 225 |
| VAL-01 | Phase 225 |
| VAL-02 | Phase 222 |
| VAL-03 | Phase 225 |

Mapped: 13/13 v1 requirements; validated 13, pending 0. Phases 224 and 226 are
cleanup-only; Phase 225 closed refreshed closeout gaps for VAL-01, TIO-03, VAL-04,
and VAL-03.

<details>
<summary>✅ v1.24 I/O Mmap Loading Strategy (Phases 204-211) — SHIPPED 2026-05-04</summary>

- [x] Phase 204: Mmap Strategy Component Boundary (1/1 plans) — completed 2026-05-04
- [x] Phase 205: Mmap Validation and Platform Gating (1/1 plans) — completed 2026-05-04
- [x] Phase 206: Mapped Descriptor, Errors, and Lifetime (1/1 plans) — completed 2026-05-04
- [x] Phase 207: Tensor-Owned Mmap Integration (1/1 plans) — completed 2026-05-04
- [x] Phase 208: Public Runtime and Evidence Surfaces (1/1 plans) — completed 2026-05-04
- [x] Phase 209: Behavior Tests and Scope Guardrails (1/1 plans) — completed 2026-05-04
- [x] Phase 210: Publication and Maintained Artifact Updates (1/1 plans) — completed 2026-05-04
- [x] Phase 211: Phase Verification Artifact Backfill (1/1 plans) — completed 2026-05-04 (gap closure)

Archive:
- `.planning/milestones/v1.24-ROADMAP.md`
- `.planning/milestones/v1.24-REQUIREMENTS.md`
- `.planning/milestones/v1.24-MILESTONE-AUDIT.md`
- `.planning/milestones/v1.24-phases/{204..210}-*` (Phase 211 backfill artifacts live alongside their parent phase dirs)

</details>

<details>
<summary>✅ v1.23 I/O Loading Strategy Boundary (Phases 197-203) — SHIPPED 2026-05-04</summary>

Archive:
- `.planning/milestones/v1.23-ROADMAP.md`
- `.planning/milestones/v1.23-REQUIREMENTS.md`
- `.planning/milestones/v1.23-MILESTONE-AUDIT.md`
- `.planning/milestones/v1.23-phases/`

</details>

### 📋 Milestone backlog

Older “next milestone” staging notes are superseded by **v1.26** (issue #63) in active planning
artifacts (`REQUIREMENTS.md`, `STATE.md`). Future milestones after v1.26 continue via
`$gsd-new-milestone`.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 227. Staged Read Strategy Component Boundary | v1.26 | 1/1 | Complete | 2026-05-07 |
| 228. Span, Target-Window, and Platform Gating | v1.26 | 1/1 | Complete | 2026-05-07 |
| 229. Staged Copy Progress and Completion Semantics | v1.26 | 1/1 | Complete | 2026-05-07 |
| 230. Context Cleanness and Per-Attempt Lifetime | v1.26 | 1/1 | Complete | 2026-05-07 |
| 231. Deterministic Error Taxonomy | v1.26 | 1/1 | Complete | 2026-05-07 |
| 232. Tensor-Owned Integration Graph | v1.26 | 1/1 | Complete | 2026-05-07 |
| 233. Public Loader and Maintained Entrypoints | v1.26 | 1/1 | Complete | 2026-05-08 |
| 234. Public Dispatch Tests | v1.26 | 1/1 | Complete | 2026-05-08 |
| 235. Scope and Non-Regression Guardrails | v1.26 | 1/1 | Complete | 2026-05-08 |
| 236. Publication and Evidence Truthfulness | v1.26 | 1/1 | Complete | 2026-05-08 |
| 237. Direct Tensor Staged Offset Contract Repair | v1.26 | 1/1 | Complete | 2026-05-08 |
| 238. Audit Artifact and Probe Reporting Cleanup | v1.26 | 1/1 | Complete | 2026-05-08 |
| 212. Read Strategy Component Boundary | v1.25 | 1/1 | Validated | 2026-05-05 |
| 213. Read Validation and Platform Gating | v1.25 | 1/1 | Validated | 2026-05-05 |
| 214. Read Execution, Errors, and Lifetime | v1.25 | 1/1 | Validated | 2026-05-05 |
| 214.1. RTC-Safe Read Execution Boundary Repair | v1.25 | 1/1 | Validated | 2026-05-05 |
| 215. Tensor-Owned Read Integration | v1.25 | 1/1 | Validated | 2026-05-05 |
| 216. Public Runtime and Evidence Surfaces | v1.25 | 1/1 | Validated | 2026-05-05 |
| 217. Behavior Tests and Scope Guardrails | v1.25 | 1/1 | Validated | 2026-05-05 |
| 218. Publication and Maintained Artifact Updates | v1.25 | 1/1 | Validated | 2026-05-05 |
| 219. Maintained Read Source Provenance | v1.25 | 1/1 | Validated | 2026-05-05 |
| 220. Explicit Tensor Read Outcome Graph | v1.25 | 1/1 | Validated | 2026-05-05 |
| 221. Read Closeout Truth Reconciliation | v1.25 | 1/1 | Superseded | 2026-05-06 |
| 222. Public Read Source Contract Repair | v1.25 | 1/1 | Validated | 2026-05-06 |
| 223. Read Closeout Truth And Validation Reconciliation | v1.25 | 1/1 | Validated | 2026-05-06 |
| 224. Read Closeout Tech Debt Cleanup | v1.25 | 1/1 | Complete    | 2026-05-06 |
| 225. Read Closeout Runtime Validation And SML Repair | v1.25 | 6/6 | Complete   | 2026-05-06 |
| 226. Read Batch Cap And Closeout Evidence Refresh | v1.25 | 1/1 | Validated | 2026-05-06 |
| 204. Mmap Strategy Component Boundary | v1.24 | 1/1 | Complete | 2026-05-04 |
| 205. Mmap Validation and Platform Gating | v1.24 | 1/1 | Complete | 2026-05-04 |
| 206. Mapped Descriptor, Errors, and Lifetime | v1.24 | 1/1 | Complete | 2026-05-04 |
| 207. Tensor-Owned Mmap Integration | v1.24 | 1/1 | Complete | 2026-05-04 |
| 208. Public Runtime and Evidence Surfaces | v1.24 | 1/1 | Complete | 2026-05-04 |
| 209. Behavior Tests and Scope Guardrails | v1.24 | 1/1 | Complete | 2026-05-04 |
| 210. Publication and Maintained Artifact Updates | v1.24 | 1/1 | Complete | 2026-05-04 |
| 211. Phase Verification Artifact Backfill | v1.24 | 1/1 | Complete | 2026-05-04 |
