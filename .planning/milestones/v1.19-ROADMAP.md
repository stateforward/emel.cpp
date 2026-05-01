# Roadmap

## Archived Milestones

- [x] [v1.0: EMEL Llama-68M Generation Slice](.planning/milestones/v1.0-ROADMAP.md)
- [x] [v1.1: EMEL Llama-68M Generation Benchmark](.planning/milestones/v1.1-ROADMAP.md)
- [x] [v1.2: Flash Attention](.planning/milestones/v1.2-ROADMAP.md)
- [x] [v1.3: ARM Flash Optimizations](.planning/milestones/v1.3-ROADMAP.md)
- [x] [v1.4: Full Vectorized Quantized Kernels](.planning/milestones/v1.4-ROADMAP.md)
- [x] [v1.5: Full ARM Quantized Path](.planning/milestones/v1.5-ROADMAP.md)
- [x] [v1.6: Qwen3-0.6B Parity And Benchmark](.planning/milestones/v1.6-ROADMAP.md)
- [x] [v1.7: Generator Prefill Submachine Decomposition](.planning/milestones/v1.7-ROADMAP.md)
- [x] [v1.8: Truthful Qwen3 E2E Embedded Size](.planning/milestones/v1.8-ROADMAP.md)
- [x] [v1.9: Liquid LFM2.5-1.2B Thinking ARM Slice](.planning/milestones/v1.9-ROADMAP.md)
- [x] [v1.11: TE-75M GGUF Trimodal Embedding Runtime](.planning/milestones/v1.11-ROADMAP.md)
  - Shipped 2026-04-15 with maintained TE trimodal embedding runtime support, refreshed closeout
    evidence, and a passing milestone audit.
- [x] [v1.12: Pluggable Reference Parity Bench Architecture](.planning/milestones/v1.12-ROADMAP.md)
  - Shipped 2026-04-18, reopened narrowly for archived closeout-proof repair on 2026-04-19, and
    returned to a passing rerun audit on 2026-04-20.
- [x] [v1.13: Pluggable Generative Parity Bench](.planning/milestones/v1.13-ROADMAP.md)
  - Shipped 2026-04-21 with a maintained generative compare contract, workload manifests,
    `llama_cpp_generation` reference lane, truthful comparable/non-comparable publication, and a
    no-blocker audit.
- [x] [v1.14: Benchmark Variant Organization](.planning/milestones/v1.14-ROADMAP.md)
  - Shipped 2026-04-21 with deterministic data-owned benchmark variant discovery for generation
    and embedding variants.
- [x] [v1.15: ARM Sortformer Diarization GGUF Slice](.planning/milestones/v1.15-ROADMAP.md)
  - Shipped 2026-04-25 with one maintained native Sortformer diarization GGUF slice, PyTorch/NeMo
    parity, ONNX CPU single-thread benchmark reference, EMEL-over-ONNX performance closure, and a
    passing source-backed milestone audit.
- [x] [v1.16: ARM Whisper GGUF Parity And Performance](.planning/milestones/v1.16-ROADMAP.md)
  - Shipped 2026-04-28 with one maintained Whisper tiny GGUF ASR slice, speech-owned runtime
    actors, recognizer-backed exact transcript parity, matched single-thread ARM benchmark proof,
    and source-backed closeout evidence.
- [x] [v1.17: Text Generator Domain Alignment](.planning/milestones/v1.17-ROADMAP.md)
  - Shipped 2026-04-30 after Phase 147 removed the final source-backed `TEXTGEN-04` /
    `TEXTGEN-07` blocker: maintained graph validation, bind, and extract callbacks no longer
    route graph outcomes through action-called `detail.hpp` helper failures or `err_out`.

- [x] [v1.18: Parity Tool Boundary Refactor](.planning/milestones/v1.18-ROADMAP.md)
  - Shipped 2026-05-01 after reopened source-backed gap closure through Phases 153-156; final
    audit passed with 12/12 active requirements satisfied.

## Current Milestone: v1.19 Benchmark Tool Pluggable Runner Refactor

v1.19 starts from GitHub issue #55. The milestone refactors `tools/bench` so shared benchmark
orchestration owns CLI/config parsing, asset resolution, request normalization, and result/report
normalization while benchmark-family execution lives behind pluggable runner boundaries that can
be built, discovered, and gated independently.

The milestone continues phase numbering after v1.18. It must preserve existing maintained
benchmark behavior, output intent, and EMEL/reference lane isolation while replacing broad
compile-time runner coupling with localized runner contracts and conservative dependency
manifests.

- [x] **Phase 157: Benchmark Orchestrator Boundary** - Establish the shared benchmark
  orchestrator boundary for common config, asset, request, and report handling.
- [x] **Phase 158: Benchmark Runner Contract And Process Seam** - Define the narrow runner
  request/result contract and process-level extension seam.
- [x] **Phase 159: Benchmark Runner Discovery And Registration** - Replace broad static case
  registration with localized runner metadata/discovery.
- [x] **Phase 160: Benchmark Independent Build Targets** - Split CMake/build wiring so runners can
  build independently and additions stay local.
- [x] **Phase 161: Benchmark Dependency Manifest Emission** - Emit deterministic per-runner
  dependency manifests from build/source-backed inputs.
- [x] **Phase 162: Benchmark Manifest Quality-Gate Consumption** - Wire conservative
  manifest-driven benchmark gate selection into quality gates.
- [x] **Phase 163: Benchmark Behavior And Lane-Isolation Closure** - Prove existing benchmark
  behavior and EMEL/reference lane isolation remain source-backed after the refactor.

## Phase Details

### Phase 157: Benchmark Orchestrator Boundary

**Goal:** Establish a shared benchmark orchestrator boundary that owns common CLI/config parsing,
asset resolution, request normalization, and result/report normalization without owning
benchmark-family execution behavior.

**Requirements:** `ORCH-01`, `LANE-01`

**Success Criteria**:
1. `bench_main.cpp` delegates common CLI/config, asset, and report setup to an orchestrator-owned
   boundary.
2. Existing benchmark families flow through a shared normalized request/result shape without
   changing their maintained intent.
3. EMEL and reference lane assets remain separately constructed and passed as lane-owned inputs.
4. Focused tests cover orchestrator-owned config/asset/report behavior and existing invocation
   compatibility.
5. The changed-file scoped quality gate passes for touched benchmark orchestrator files.

**Status:** Complete. Phase 157 moved shared benchmark CLI/config/report orchestration behind
`emel::bench::run_bench_cli(...)`, kept `bench_main.cpp` as a process shim, preserved existing
EMEL/reference lane execution, and added focused source tests for ownership drift.

### Phase 158: Benchmark Runner Contract And Process Seam

**Goal:** Define a narrow orchestrator-to-runner contract that lets benchmark-family execution
move behind explicit runner boundaries and supports future out-of-process or foreign-language
runners.

**Requirements:** `RUNNER-01`, `RUNNER-02`

**Success Criteria**:
1. Benchmark-family execution is represented by a small runner request/result contract.
2. Adding a runner requires implementing the contract and localized registration, not editing
   unrelated runner implementation files.
3. The contract has a serialized request/result form suitable for process-level runner execution.
4. Initial maintained runners can remain C++ while the process seam is tested with a lightweight
   local runner fixture.
5. Tests prove malformed or unavailable runner responses fail closed with deterministic errors.

**Status:** Complete. Phase 158 added `bench_runner_request/v1` and
`bench_runner_result/v1` serialized payloads, wired `run_bench_cli(...)` through a normalized
`runner_request`, and added fail-closed contract tests.

### Phase 159: Benchmark Runner Discovery And Registration

**Goal:** Replace broad static benchmark case wiring with localized runner metadata or discovery so
the orchestrator no longer needs compile-time knowledge of every runner implementation.

**Requirements:** `DISC-01`

**Success Criteria**:
1. Available benchmark runners are discovered or registered through a small metadata surface.
2. `bench_main.cpp` no longer owns a large static registry of benchmark families or cases.
3. Runner lookup is deterministic and fails closed for unknown runner or case IDs.
4. A test runner proves registration stays local to runner-owned files.
5. Existing operator-facing benchmark selection still resolves maintained benchmark families.

**Status:** Complete. Phase 159 moved suite metadata into
`tools/bench/bench_runner_registry.hpp` / `.cpp`, updated the orchestrator to consume registered
default and kernel runner spans, preserved tokenizer inclusion filtering, and added focused tests
for deterministic lookup and registration ownership.

### Phase 160: Benchmark Independent Build Targets

**Goal:** Reorganize `tools/bench` build wiring so benchmark runners can build independently and
new runners do not require broad rebuilds or source edits across existing families.

**Requirements:** `BUILD-01`, `BUILD-02`

**Success Criteria**:
1. CMake exposes independent runner targets or isolated source groups for maintained benchmark
   families.
2. Touching one runner's implementation does not force rebuilding unrelated runner targets in the
   normal build graph.
3. Adding a placeholder runner proves build and registration changes stay localized.
4. Source/build checks fail if new runner additions modify unrelated runner implementation files.
5. Focused bench builds and tests pass on the modular target layout.

**Status:** Complete. Phase 160 added per-suite `bench_runner_suite_<suite>` object targets,
linked selected suite object files into the existing `bench_runner` binary, preserved filtered
disabled-stub behavior, and added focused source checks for localized suite build wiring.

### Phase 161: Benchmark Dependency Manifest Emission

**Goal:** Emit deterministic per-runner dependency manifests that describe benchmark source,
config, fixture, model, and script inputs for conservative impact detection.

**Requirements:** `MANIFEST-01`, `MANIFEST-02`

**Success Criteria**:
1. Each maintained benchmark runner emits or maintains a deterministic dependency manifest.
2. Manifest records are derived from build/source-backed runner inputs rather than hand-waved
   milestone claims.
3. Manifest records cover source, config, fixture, model, and script inputs relevant to each
   runner.
4. Missing, stale, and uncertain manifest semantics are documented as conservative rerun triggers.
5. Tests cover manifest rendering, deterministic ordering, and missing/stale/uncertain states.

**Status:** Complete. Phase 161 added `bench_dependency_manifest/v1`, runner CLI write/check
operations, a generated `tools/bench/dependency_manifest.txt` baseline, schema docs, and tests for
registered runner coverage plus missing/stale/uncertain freshness semantics.

### Phase 162: Benchmark Manifest Quality-Gate Consumption

**Goal:** Wire per-runner benchmark dependency manifests into `scripts/quality_gates.sh` so
changed-file scoped gates select relevant benchmark lanes conservatively.

**Requirements:** `GATE-01`, `GATE-02`

**Success Criteria**:
1. The changed-file scoped quality gate can map impacted files to relevant benchmark runners via
   dependency manifests.
2. Missing manifest data forces the relevant benchmark gate or full benchmark gate.
3. Stale or uncertain manifest data forces the relevant benchmark gate or full benchmark gate.
4. Manifest-based selection never skips a runner when touched files are outside known manifest
   coverage.
5. Focused tests or scripted checks cover impacted, unrelated, missing, stale, and uncertain
   manifest cases.

**Status:** Complete. Phase 162 wired `tools/bench/dependency_manifest.txt` into changed-file
scoped benchmark gate selection, added benchmark manifest freshness checks before skip decisions,
and added source tests for scoped runner mapping plus full-gate escalation.

### Phase 163: Benchmark Behavior And Lane-Isolation Closure

**Goal:** Close the milestone by proving existing maintained benchmark families still work with the
same comparison intent and that EMEL/reference lane isolation remains explicit and enforceable.

**Requirements:** `ORCH-02`, `LANE-02`

**Success Criteria**:
1. Existing maintained benchmark families continue to pass their focused smoke, compare, or
   publication checks after the runner boundary refactor.
2. Existing output schemas, fixture identities, result summaries, and failure semantics are stable
   unless a behavior change was explicitly approved and documented.
3. Source checks fail if shared benchmark code owns or reuses lane-owned model, vocab, tokenizer,
   formatter, runtime, cache, or output objects across EMEL/reference lanes.
4. Source checks fail if benchmark harnesses reach directly into actor `actions.hpp`,
   `guards.hpp`, or `detail.hpp` helpers instead of public state-machine or kernel-owned surfaces.
5. The relevant scoped quality gate passes and requirement traceability shows 13/13 active
   requirements mapped.

**Status:** Complete. Phase 163 added source-backed checks for shared benchmark lane neutrality,
actor-boundary cleanliness, and maintained behavior-test coverage. Full-suite
`bench_runner_tests` and the generation-scoped quality gate passed.
