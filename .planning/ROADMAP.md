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
- 🚧 **v1.25 I/O Read Loading Strategy** — active, planned 2026-05-05 (Phases 212-218)

## Phases

### 🚧 v1.25 I/O Read Loading Strategy (Phases 212-218) — ACTIVE

Source: GitHub issue #62, "Add io/read state machine for copy-based tensor loading".
Adds a dedicated `src/emel/io/read` Stateforward.SML actor for explicit read/copy tensor
loading beneath tensor-owned residency. Mmap, staged/chunked constrained-memory, async,
and device strategies remain out of scope.

- [x] Phase 212: Read Strategy Component Boundary (1/1 plans) — completed 2026-05-05
- [ ] Phase 213: Read Validation and Platform Gating (0/1 plans)
- [ ] Phase 214: Read Execution, Errors, and Lifetime (0/1 plans)
- [ ] Phase 215: Tensor-Owned Read Integration (0/1 plans)
- [ ] Phase 216: Public Runtime and Evidence Surfaces (0/1 plans)
- [ ] Phase 217: Behavior Tests and Scope Guardrails (0/1 plans)
- [ ] Phase 218: Publication and Maintained Artifact Updates (0/1 plans)

Active artifacts:
- `.planning/REQUIREMENTS.md` (v1.25 active requirements)

**Execution Order:** Phases execute in numeric order: 212 -> 213 -> 214 -> 215 -> 216 -> 217 -> 218.

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
**Plans**: TBD (run `$gsd-plan-phase 213`).

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
**Plans**: TBD (run `$gsd-plan-phase 214`).

#### Phase 215: Tensor-Owned Read Integration
**Goal**: `model/tensor` can request and consume read-backed I/O through the public `emel/io`
boundary while retaining load, bind, evict, and residency orchestration.
**Depends on**: Phase 214
**Requirements**: TIO-01, TIO-02
**Success Criteria** (what must be TRUE):
  1. Tensor load flow can request read-based (copy) loading through public `emel/io` events
     without direct low-level read calls.
  2. Tensor bind, residency, and evict transitions remain in `model/tensor` and consume read
     success outcomes that reference the caller-owned target buffer.
  3. Read success, unsupported, validation failure, file open failure, and file read failure
     are visible as explicit `_done` and `_error` events or states.
  4. Maintainer can verify no callback-selected outcomes, mirrored status fields, or context
     phase flags decide tensor-to-I/O outcomes for read-backed loading.
**Plans**: TBD (run `$gsd-plan-phase 215`).

#### Phase 216: Public Runtime and Evidence Surfaces
**Goal**: Runtime entrypoints and maintained tool lanes can select or report read-backed
loading only through public surfaces, and evidence reflects the actual EMEL runtime path.
**Depends on**: Phase 215
**Requirements**: TIO-03, VAL-04
**Success Criteria** (what must be TRUE):
  1. `model/loader`, maintained benchmark lanes, paritychecker lanes, and embedded probes can
     select or report read-backed loading only through public tensor and I/O runtime contracts.
  2. Maintained benchmark, paritychecker, and embedded probe lanes avoid actor-internal
     reach-through and contain no low-level read logic.
  3. Benchmark and parity output reports read-strategy usage only when the EMEL lane executed
     the read-backed runtime path.
  4. Unsupported or fallback behavior is reported as unsupported or non-read-strategy, not as
     read-strategy parity or performance evidence.
**Plans**: TBD (run `$gsd-plan-phase 216`).

#### Phase 217: Behavior Tests and Scope Guardrails
**Goal**: Tests and guardrails prove read behavior through public dispatch and prevent scope
or ownership leaks.
**Depends on**: Phase 216
**Requirements**: VAL-01, VAL-02
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
**Plans**: TBD (run `$gsd-plan-phase 217`).

#### Phase 218: Publication and Maintained Artifact Updates
**Goal**: Maintained docs, snapshots, benchmark outputs, model artifacts, and planning truth
describe read-strategy support exactly as implemented.
**Depends on**: Phase 217
**Requirements**: VAL-03
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
**Plans**: TBD (run `$gsd-plan-phase 218`).

#### Coverage

| Requirement | Phase |
|-------------|-------|
| READ-01 | Phase 212 |
| READ-02 | Phase 213 |
| PLAT-01 | Phase 213 |
| READ-03 | Phase 214 |
| LIFE-01 | Phase 214 |
| ERR-01 | Phase 214 |
| TIO-01 | Phase 215 |
| TIO-02 | Phase 215 |
| TIO-03 | Phase 216 |
| VAL-04 | Phase 216 |
| VAL-01 | Phase 217 |
| VAL-02 | Phase 217 |
| VAL-03 | Phase 218 |

Mapped: 13/13 v1 requirements; validated 1, pending 12.

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

### 📋 Next Milestone

After v1.25 ships, the next milestone selection happens via `$gsd-new-milestone`.
Staged/chunked constrained-memory, cooperative async, and device-specific loading
strategies remain deferred follow-on work below the `emel/io` boundary.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 212. Read Strategy Component Boundary | v1.25 | 1/1 | Validated | 2026-05-05 |
| 213. Read Validation and Platform Gating | v1.25 | 0/1 | Pending | — |
| 214. Read Execution, Errors, and Lifetime | v1.25 | 0/1 | Pending | — |
| 215. Tensor-Owned Read Integration | v1.25 | 0/1 | Pending | — |
| 216. Public Runtime and Evidence Surfaces | v1.25 | 0/1 | Pending | — |
| 217. Behavior Tests and Scope Guardrails | v1.25 | 0/1 | Pending | — |
| 218. Publication and Maintained Artifact Updates | v1.25 | 0/1 | Pending | — |
| 204. Mmap Strategy Component Boundary | v1.24 | 1/1 | Complete | 2026-05-04 |
| 205. Mmap Validation and Platform Gating | v1.24 | 1/1 | Complete | 2026-05-04 |
| 206. Mapped Descriptor, Errors, and Lifetime | v1.24 | 1/1 | Complete | 2026-05-04 |
| 207. Tensor-Owned Mmap Integration | v1.24 | 1/1 | Complete | 2026-05-04 |
| 208. Public Runtime and Evidence Surfaces | v1.24 | 1/1 | Complete | 2026-05-04 |
| 209. Behavior Tests and Scope Guardrails | v1.24 | 1/1 | Complete | 2026-05-04 |
| 210. Publication and Maintained Artifact Updates | v1.24 | 1/1 | Complete | 2026-05-04 |
| 211. Phase Verification Artifact Backfill | v1.24 | 1/1 | Complete | 2026-05-04 |
