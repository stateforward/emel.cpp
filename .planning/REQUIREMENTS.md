# Requirements: v1.19 Benchmark Tool Pluggable Runner Refactor

**Defined:** 2026-05-01
**Source:** GitHub issue #55, "Refactor benchmark tool for cleaner boundaries and pluggable
runners"
**Core Value:** Prove real end-to-end behavior with explicit SML orchestration and
parity-oriented verification before widening API surface or model scope.

## Active Requirements

### Orchestrator Boundary

- [ ] **ORCH-01**: `tools/bench` has a shared benchmark orchestrator boundary that owns
  CLI/config parsing, asset resolution, request normalization, and result/report normalization.
- [ ] **ORCH-02**: Existing benchmark invocations preserve their current user-facing arguments,
  maintained fixtures, output schemas, and failure semantics unless a change is explicitly
  documented and approved.

### Runner Interfaces

- [ ] **RUNNER-01**: Maintainers can add a benchmark family by implementing a runner contract and
  localized registration without editing unrelated runner sources.
- [ ] **RUNNER-02**: The orchestrator-to-runner contract supports a process-level runner seam with
  serialized request/result payloads, so future foreign-language runners do not require shared
  in-process runtime state.

### Discovery And Registration

- [ ] **DISC-01**: The benchmark orchestrator discovers or registers available runners from a
  small runner metadata surface instead of broad static case wiring in `bench_main.cpp`.

### Build Isolation

- [ ] **BUILD-01**: Each maintained benchmark runner builds as its own CMake target or isolated
  runner source group so modifying one runner does not force rebuilding untouched runners.
- [ ] **BUILD-02**: Source or build checks prove new runner registration stays localized and does
  not require broad edits through existing runner implementation files.

### Dependency Manifests

- [ ] **MANIFEST-01**: Each benchmark runner emits or maintains a build-time dependency manifest
  that names source, config, fixture, model, and script inputs needed for conservative gate impact
  detection.
- [ ] **MANIFEST-02**: The benchmark dependency-manifest format is deterministic, documented, and
  explicit about missing, stale, or uncertain-data semantics.

### Quality Gates

- [ ] **GATE-01**: `scripts/quality_gates.sh` can use per-runner benchmark dependency manifests
  to choose the relevant benchmark gate when changed files affect runner inputs.
- [ ] **GATE-02**: Missing, stale, or uncertain benchmark manifest data forces the relevant
  benchmark gate or full benchmark gate instead of permitting a skip.

### Lane Isolation

- [ ] **LANE-01**: Existing EMEL and reference benchmark lanes construct and own their model,
  vocab, tokenizer, formatter, runtime, cache, and output state separately.
- [ ] **LANE-02**: Tests or source checks fail if shared benchmark orchestration or runner code
  shares lane-owned runtime objects, shared bootstrap paths, or direct actor
  `actions.hpp`/`guards.hpp`/`detail.hpp` helper calls.

## Future Requirements

- **RUNNER-F01**: Third-party or out-of-repo benchmark plugin distribution can be designed after
  the repo-owned runner contract is stable.
- **RUNNER-F02**: New benchmark families or model-specific benchmarks can be added after the
  runner boundary is source-backed.
- **GATE-F01**: Hash-assisted gate invalidation can supplement the build-graph dependency
  manifests after conservative manifest behavior is proven.
- **GATE-F02**: Fine-grained benchmark skip optimization can be planned after missing, stale, and
  uncertain manifest data already force safe reruns.

## Out of Scope

| Feature | Reason |
|---------|--------|
| New benchmark semantics or new performance claims | This milestone refactors boundaries and must preserve maintained benchmark intent. |
| New model-family or runtime support | Runner plumbing must not be proven by adding unrelated `src/` runtime scope. |
| Public third-party plugin SDK | The immediate need is a repo-owned runner contract, not external distribution. |
| Remote benchmark services | Process-level local runner isolation is enough for this milestone. |
| Hash-only gate skipping | Content hashes may supplement dependency manifests, but cannot replace conservative graph-derived inputs. |
| Sharing EMEL/reference runtime state | Lane isolation is a hard constraint of issue #55 and the existing project contract. |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ORCH-01 | Phase 157 | Complete |
| ORCH-02 | Phase 163 | Pending |
| RUNNER-01 | Phase 158 | Complete |
| RUNNER-02 | Phase 158 | Complete |
| DISC-01 | Phase 159 | Complete |
| BUILD-01 | Phase 160 | Complete |
| BUILD-02 | Phase 160 | Complete |
| MANIFEST-01 | Phase 161 | Complete |
| MANIFEST-02 | Phase 161 | Complete |
| GATE-01 | Phase 162 | Pending |
| GATE-02 | Phase 162 | Pending |
| LANE-01 | Phase 157 | Complete |
| LANE-02 | Phase 163 | Pending |

**Coverage:**
- Active requirements: 13 total
- Mapped to phases: 13
- Unmapped: 0

---
*Requirements defined: 2026-05-01*
*Last updated: 2026-05-01 after roadmap traceability mapping*
