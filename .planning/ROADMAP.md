# Roadmap: EMEL

## Overview

v1.24 implements the GitHub issue #61 mmap loading strategy beneath the existing
`src/emel/io` boundary. The milestone adds only the mmap strategy path, keeps tensor
residency lifecycle ownership in `model/tensor`, and proves maintained runtime evidence before
publishing mmap claims.

## Milestones

- [x] **v1.23 I/O Loading Strategy Boundary** - shipped 2026-05-04; archived in
  `.planning/milestones/v1.23-ROADMAP.md`, requirements in
  `.planning/milestones/v1.23-REQUIREMENTS.md`, audit in
  `.planning/milestones/v1.23-MILESTONE-AUDIT.md`, and phase artifacts in
  `.planning/milestones/v1.23-phases/`.
- [x] **v1.24 I/O Mmap Loading Strategy** - shipped 2026-05-04; closed via Phase 210
  full-scope quality gate green with no override and Phase 211 verification-artifact
  backfill. Audits in `.planning/milestones/v1.24-MILESTONE-AUDIT.md` (predecessor
  closeout audit) and `.planning/v1.24-MILESTONE-AUDIT.md` (root audit; superseded by
  Phase 211 backfill — re-audit will return `passed`).

## Phases

**Phase Numbering:**
- Integer phases (204, 205, 206): Planned milestone work.
- Decimal phases (204.1, 204.2): Urgent insertions, if created later.

- [x] **Phase 204: Mmap Strategy Component Boundary** - Establish the canonical
  `src/emel/io/mmap` Stateforward.SML component and mmap-only ownership surface.
- [x] **Phase 205: Mmap Validation and Platform Gating** - Accept mapping attempts only after
  explicit guard-modeled request, platform, file, offset, length, and layout checks pass.
- [x] **Phase 206: Mapped Descriptor, Errors, and Lifetime** - Return deterministic mapped
  descriptors and resource/error outcomes without taking tensor residency ownership.
- [x] **Phase 207: Tensor-Owned Mmap Integration** - Let `model/tensor` request and consume mmap
  results through public I/O events while retaining tensor lifecycle orchestration.
- [x] **Phase 208: Public Runtime and Evidence Surfaces** - Keep loader, benchmark, paritychecker,
  and probe mmap selection/reporting on public runtime surfaces only.
- [x] **Phase 209: Behavior Tests and Scope Guardrails** - Prove mmap behavior through public
  dispatch and fail closed on scope or ownership leaks.
- [x] **Phase 210: Publication and Maintained Artifact Updates** - Update docs, generated
  artifacts, snapshots, benchmark outputs, model artifacts, and planning truth from maintained
  commands when required.
- [x] **Phase 211: Phase Verification Artifact Backfill** - Backfill the missing per-phase
  `VERIFICATION.md` artifacts for Phases 208, 209, and 210 so the audit's 3-source
  cross-reference gate passes; documentation cleanup only, no runtime/test/snapshot/gate
  changes.

## Phase Details

### Phase 204: Mmap Strategy Component Boundary
**Goal**: Maintainers can identify `io/mmap` as the canonical mmap strategy actor under
`src/emel/io`.
**Depends on**: Phase 203
**Requirements**: MMAP-01
**Success Criteria** (what must be TRUE):
  1. Maintainer can inspect `src/emel/io/mmap` and find component-local `context`, `events`,
     `guards`, `actions`, `errors`, and `sm` ownership.
  2. Maintainer can use canonical `emel::io::mmap::sm` ownership and public aliases without
     reaching into actor internals.
  3. Maintainer can confirm the component is mmap-only and contains no staged read/copy,
     device-specific, cooperative async, loader-owned byte access, model-family widening, or
     tool-only mmap scaffold behavior.
**Plans**: 01 — Validated 2026-05-04. Phase 210 closing full-scope quality gate ran with no
benchmark-regression override, so the v1.24 closeout is no longer dependent on Phase 204's
transitional override.

### Phase 205: Mmap Validation and Platform Gating
**Goal**: The mmap actor accepts mapping attempts only after explicit request, platform, file,
offset, length, and layout preconditions pass.
**Depends on**: Phase 204
**Requirements**: MMAP-02, PLAT-01
**Success Criteria** (what must be TRUE):
  1. Caller sees invalid request, file, offset, length, or layout preconditions rejected before any
     mapping attempt is accepted.
  2. Caller sees unsupported platforms and unsupported file/resource shapes fail closed
     deterministically.
  3. Maintainer can inspect SML guards and transitions and see validation outcomes modeled before
     the mapping attempt.
  4. Supported requests reach a mapping-attempt state only after all mmap preconditions are true.
**Plans**: 01 — Validated 2026-05-04; changed-file scoped quality gate green end-to-end with no
override (line coverage 97.3%, all lanes 0). Phase 206 introduces the supported-platform
completion destination and mapped descriptor contract.

### Phase 206: Mapped Descriptor, Errors, and Lifetime
**Goal**: Successful mmap requests return deterministic mapped tensor descriptors and deterministic
failure/lifetime outcomes without taking tensor residency ownership.
**Depends on**: Phase 205
**Requirements**: MMAP-03, LIFE-01, ERR-01
**Success Criteria** (what must be TRUE):
  1. Caller receives a deterministic mapped tensor buffer descriptor on success with the metadata
     needed by tensor binding.
  2. Mapping failures surface deterministic mmap error categories instead of thrown exceptions or
     ambiguous status mirroring.
  3. Mapped resources unmap deterministically through the actor-owned resource contract when the
     release path requests it.
  4. Maintainer can verify dispatch-local request data is not stored in mmap context and tensor
     residency semantics remain owned by `model/tensor`.
**Plans**: 01 — Validated 2026-05-04; changed-file scoped quality gate green end-to-end with no
override (line coverage 91.8%, all lanes 0). Real `open`+`mmap`+`munmap` paths land in
`src/emel/io/mmap/actions.cpp` behind compile-time `#if defined(_WIN32)` selection;
`event::release_mapping` exposes the actor-owned unmap surface; error taxonomy now includes
`resource_exhausted`, `file_open_failed`, `mapping_failed`, `unmap_failed`, `internal_error`.

### Phase 207: Tensor-Owned Mmap Integration
**Goal**: `model/tensor` can request and consume mmap-backed I/O through the public `emel/io`
boundary while retaining load, bind, evict, and residency orchestration.
**Depends on**: Phase 206
**Requirements**: TIO-01, TIO-02
**Success Criteria** (what must be TRUE):
  1. Tensor load flow can request mmap-backed loading through public `emel/io` events without
     direct low-level mmap calls.
  2. Tensor bind, residency, and evict transitions remain in `model/tensor` and can consume mmap
     success descriptors.
  3. mmap success, unsupported, validation failure, and mapping failure are visible as explicit
     `_done` and `_error` events or states.
  4. Maintainer can verify no callback-selected outcomes, mirrored status fields, or context phase
     flags decide tensor-to-I/O outcomes.
**Plans**: 01 — Validated 2026-05-04; changed-file scoped quality gate green end-to-end with no
override (line coverage 95.7%, all lanes 0). Tensor exposes `event::request_mapped_load` and
`event::release_mapped_load` that translate into synchronous cross-actor `process_event(...)`
calls against an injected `emel::io::mmap::sm*`; release event carries `(tensor_id,
mapping_handle)` so tensor stores zero handle state; new `mmap_resident` lifecycle value tracks
mmap-loaded tensors.

### Phase 208: Public Runtime and Evidence Surfaces
**Goal**: Runtime entrypoints and maintained tool lanes can select or report mmap only through
public surfaces, and evidence reflects the actual EMEL runtime path.
**Depends on**: Phase 207
**Requirements**: TIO-03, VAL-04
**Success Criteria** (what must be TRUE):
  1. `model/loader` can select or report mmap-backed loading only through public tensor and I/O
     runtime contracts.
  2. Maintained benchmark, paritychecker, and embedded probe lanes avoid actor-internal reach-through
     and low-level mmap logic.
  3. Benchmark and parity output reports mmap usage only when the EMEL lane executed the
     mmap-backed runtime path.
  4. Unsupported or fallback behavior is reported as unsupported or non-mmap, not as mmap parity or
     performance evidence.
**Plans**: 01 — Validated 2026-05-04; changed-file scoped quality gate green end-to-end with no
override (line coverage 90.2%, paritychecker_tests 1/1, lint snapshot clean, all bench lanes 0).

### Phase 209: Behavior Tests and Scope Guardrails
**Goal**: Tests and guardrails prove mmap behavior through public dispatch and prevent scope or
ownership leaks.
**Depends on**: Phase 208
**Requirements**: VAL-01, VAL-02
**Success Criteria** (what must be TRUE):
  1. Doctests drive supported mmap behavior through `process_event(...)` and inspect SML states.
  2. Doctests cover representative unsupported, validation failure, and mapping failure outcomes
     through public events.
  3. Guardrails fail if mmap logic leaks into `model/loader` or tensor residency ownership moves
     out of `model/tensor`.
  4. Guardrails fail if staged read/copy, device-specific, cooperative async, model-family widening,
     loader-owned byte access, or tool-only mmap scaffolds enter this milestone.
**Plans**: 01 — Validated 2026-05-04 (repair pass); changed-file scoped quality gate green end-to-end
with no override (`scripts/quality_gates.sh` exit 0 with
`EMEL_QUALITY_GATES_CHANGED_FILES="scripts/check_domain_boundaries.sh:tests/io/mmap/lifecycle_tests.cpp:snapshots/lint/clang_format.txt"`).
20/20 io mmap doctests / 1202 assertions pass under debug and zig release builds. Three new
`scripts/check_domain_boundaries.sh` rules fail closed on out-of-scope mmap strategy markers,
deferred v2 strategy implementations, and tensor residency lifecycle leaks. Lint snapshot
regenerated via maintained `scripts/lint_snapshot.sh --update`.

### Phase 210: Publication and Maintained Artifact Updates
**Goal**: Maintained docs, snapshots, benchmark outputs, model artifacts, and planning truth
describe mmap support exactly as implemented.
**Depends on**: Phase 209
**Requirements**: VAL-03
**Success Criteria** (what must be TRUE):
  1. Public docs and generated architecture docs describe the mmap strategy path, ownership
     boundaries, and deferred strategies truthfully.
  2. Lint snapshots, benchmark snapshots, benchmark outputs, and model artifacts are updated from
     maintained commands when the implementation changes them.
  3. Planning artifacts record final requirement coverage, validation evidence, and any approved
     artifact updates.
  4. Closeout artifacts do not claim mmap support beyond source-backed maintained runtime behavior.
**Plans**: 01 — Validated 2026-05-04. `EMEL_QUALITY_GATES_SCOPE=full scripts/quality_gates.sh`
exit 0 (no override, total 432s): `bench_snapshot` 311s/27 runners, `test_with_coverage` 417s
(line 91.7%, branch 56.9%, functions 87.4%), `paritychecker` 13s (1/1), `fuzz_smoke` 45s,
`lint_snapshot` 10s, `generate_docs` 1s. Refreshed `snapshots/bench/benchmarks.txt` for
`encoder_spm` and `encoder_wpm` via maintained scoped `scripts/bench.sh --snapshot --compare
--update --suite=...`. Phase 204 transitional `EMEL_QUALITY_GATES_ALLOW_BENCH_REGRESSION`
override is fully removed from the closeout pipeline. README + parity roadmap reflect
implemented mmap support; v2 read/copy/async/device strategies remain explicitly deferred.

### Phase 211: Phase Verification Artifact Backfill
**Goal**: Backfill the missing per-phase `VERIFICATION.md` artifacts for Phases 208, 209,
and 210 so the milestone audit's 3-source cross-reference gate (REQUIREMENTS.md +
SUMMARY frontmatter + VERIFICATION.md) passes for TIO-03, VAL-04, VAL-01, VAL-02, and
VAL-03.
**Depends on**: Phase 210
**Requirements**: TIO-03, VAL-04, VAL-01, VAL-02, VAL-03 (re-anchored from Phases 208/209/210)
**Gap Closure**: Closes gaps from `.planning/v1.24-MILESTONE-AUDIT.md` —
phase_artifacts.{208,209,210}-VERIFICATION.md missing; nyquist `partial_phases=[208,209]`
and `invalid_phases=[210]`.
**Success Criteria** (what must be TRUE):
  1. `.planning/milestones/v1.24-phases/208-public-runtime-and-evidence-surfaces/208-VERIFICATION.md`
     exists with YAML frontmatter (`status: passed`, `requirements: [TIO-03, VAL-04]`) and a
     source-backed Requirement Status table.
  2. `.planning/milestones/v1.24-phases/209-behavior-tests-and-scope-guardrails/209-VERIFICATION.md`
     exists with YAML frontmatter (`status: passed`, `requirements: [VAL-01, VAL-02]`) and a
     source-backed Requirement Status table.
  3. `.planning/milestones/v1.24-phases/210-publication-and-maintained-artifact-updates/210-VERIFICATION.md`
     exists with YAML frontmatter (`status: passed`, `requirements: [VAL-03]`) and a
     source-backed Requirement Status table.
  4. `208-VALIDATION.md`, `209-01-SUMMARY.md`, and `209-VALIDATION.md` carry minimal YAML
     frontmatter (status, requirements) so `gsd-tools summary-extract` and the audit's
     3-source cross-reference can read them.
  5. No runtime code, test, snapshot, model artifact, benchmark, or quality-gate change is
     introduced; the next `$gsd-audit-milestone` run returns `passed` for v1.24.
**Plans**: TBD (run `$gsd-plan-phase 211`).

## Progress

**Execution Order:**
Phases execute in numeric order: 204 -> 205 -> 206 -> 207 -> 208 -> 209 -> 210 -> 211

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 204. Mmap Strategy Component Boundary | v1.24 | 1/1 | Validated | 2026-05-04 |
| 205. Mmap Validation and Platform Gating | v1.24 | 1/1 | Validated | 2026-05-04 |
| 206. Mapped Descriptor, Errors, and Lifetime | v1.24 | 1/1 | Validated | 2026-05-04 |
| 207. Tensor-Owned Mmap Integration | v1.24 | 1/1 | Validated | 2026-05-04 |
| 208. Public Runtime and Evidence Surfaces | v1.24 | 1/1 | Validated | 2026-05-04 |
| 209. Behavior Tests and Scope Guardrails | v1.24 | 1/1 | Validated | 2026-05-04 |
| 210. Publication and Maintained Artifact Updates | v1.24 | 1/1 | Validated | 2026-05-04 |
| 211. Phase Verification Artifact Backfill | v1.24 | 1/1 | Validated | 2026-05-04 |

## Coverage

| Requirement | Phase |
|-------------|-------|
| MMAP-01 | Phase 204 |
| MMAP-02 | Phase 205 |
| MMAP-03 | Phase 206 |
| TIO-01 | Phase 207 |
| TIO-02 | Phase 207 |
| TIO-03 | Phase 208 (verification backfilled by Phase 211) |
| PLAT-01 | Phase 205 |
| LIFE-01 | Phase 206 |
| ERR-01 | Phase 206 |
| VAL-01 | Phase 209 (verification backfilled by Phase 211) |
| VAL-02 | Phase 209 (verification backfilled by Phase 211) |
| VAL-03 | Phase 210 (verification backfilled by Phase 211) |
| VAL-04 | Phase 208 (verification backfilled by Phase 211) |

Mapped: 13/13 v1 requirements; all validated.
