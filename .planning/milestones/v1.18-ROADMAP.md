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

## Current Milestone: v1.18 Parity Tool Boundary Refactor

v1.18 starts from GitHub issue #54. The milestone refactors `tools/paritychecker` into explicit
runner, engine, asset-loading, build-registration, and dependency-manifest boundaries while
preserving existing parity behavior and EMEL/reference lane isolation.

Reopened 2026-05-01 after a source-backed milestone audit found maintained-path gaps in runner
ownership, live generation reference truth, actor-helper boundary enforcement, and dependency
manifest gate consumption.

- [x] **Phase 148: Parity Runner Asset Boundary** - Establish the shared runner boundary and
  centralize asset/config/fixture normalization without changing mode behavior.
- [x] **Phase 149: Parity Engine Adapter Split** - Move tokenizer, GBNF, kernel, Jinja, and
  generation mode implementations behind explicit engine adapters.
- [x] **Phase 150: Parity Build Registration Boundary** - Make CMake and source registration
  modular enough for localized future engine additions.
- [x] **Phase 151: Parity Dependency Manifest Emission** - Emit per-runner dependency manifests
  with conservative stale/missing-data semantics.
- [x] **Phase 152: Parity Behavior And Lane-Isolation Closure** - Prove all existing modes still
  behave the same and lane isolation remains enforceable.
- [x] **Phase 153: Parity Runner Config Ownership Closure** - Move CLI/config parsing ownership
  behind the shared runner boundary and preserve existing CLI behavior.
- [x] **Phase 154: Generation Live Reference Truth Closure** - Make maintained generation parity
  compare against live reference-lane output while preserving lane isolation and snapshot
  publication semantics.
- [ ] **Phase 155: Parity Actor Boundary Enforcement Closure** - Replace direct paritychecker
  actor-helper/detail reaches with public state-machine or owned-kernel surfaces and broaden source
  checks.
- [ ] **Phase 156: Parity Dependency Manifest Gate Closure** - Wire manifest emission and
  missing/stale/uncertain full-gate semantics into production paritychecker or quality-gate entry
  points.

## Phase Details

### Phase 148: Parity Runner Asset Boundary

**Goal:** Establish a shared parity runner boundary that owns CLI/config parsing, asset resolution,
fixture normalization, lane invocation, and result normalization while preserving all existing mode
behavior.

**Requirements:** `PARITY-01`, `LANE-01`

**Success Criteria**:
1. Shared runner code owns common request/config/asset setup used by existing parity modes.
2. Mode behavior remains delegated to existing implementation paths until Phase 149 splits the
   engines.
3. EMEL and reference lane assets remain separately constructed and owned.
4. Focused paritychecker tests cover centralized asset resolution and existing fixtures.
5. Existing parity modes still pass their maintained smoke or fixture checks.

**Status:** Complete. Phase 148 added `emel::paritychecker::assets` for shared file, path,
byte-loading, baseline-directory, and maintained generation fixture resolution. The runner now uses
that boundary without changing mode dispatch or lane-owned runtime state. Focused paritychecker
build/tests and the changed-file scoped quality gate passed.

### Phase 149: Parity Engine Adapter Split

**Goal:** Move tokenizer, GBNF, kernel, Jinja, and generation parity implementations behind
explicit runner-facing engine adapters so the runner stops containing bulk per-mode logic.

**Requirements:** `PARITY-02`, `ENGINE-01`

**Success Criteria**:
1. Each existing mode has a narrow engine adapter boundary with explicit EMEL/reference lane
   execution ownership.
2. Runner orchestration invokes engines through a shared interface instead of mode-specific
   implementation branches.
3. Existing output schemas, fixture IDs, and error behavior remain stable.
4. Tests fail if parity harnesses reach directly into actor action/guard/detail helpers.
5. Changed-file scoped quality gates pass for paritychecker code touched by the adapter split.

**Status:** Complete. Phase 149 moved the existing bulk parity mode implementation into
`parity_engines.cpp`, added explicit `engine_adapter` metadata and per-mode adapter entrypoints,
and reduced `parity_runner.cpp` to orchestration through `find_engine(opts.mode)`. Focused
paritychecker builds/tests and the changed-file scoped quality gate passed.

### Phase 150: Parity Build Registration Boundary

**Goal:** Make parity engine build and registration wiring explicit, modular, and localized so
future engines do not require broad runner rewrites or unrelated mode edits.

**Requirements:** `ENGINE-02`, `BUILD-01`, `BUILD-02`

**Success Criteria**:
1. `tools/paritychecker` CMake exposes separate runner and engine build boundaries.
2. Engine registration is auditable from a small source/build surface.
3. Adding a placeholder or test engine proves registration stays localized.
4. No hidden runtime fallback silently changes which engine handles a mode.
5. Build and focused paritychecker tests pass on the modular target layout.

**Status:** Complete. Phase 150 factored paritychecker CMake into explicit runner, engine
registration, engine implementation, tokenizer-engine, reference-support, and shared common source
groups. Source regressions prove both targets consume the shared group and invalid engine lookup
remains fail-closed. Focused paritychecker builds/tests and the changed-file scoped quality gate
passed.

### Phase 151: Parity Dependency Manifest Emission

**Goal:** Add per-runner dependency manifests that describe parity source/config/fixture/model/script
inputs for conservative quality-gate impact selection.

**Requirements:** `MANIFEST-01`, `MANIFEST-02`, `MANIFEST-03`

**Success Criteria**:
1. Each parity runner has a dependency manifest emitted or maintained from source-backed
   registration data.
2. Manifest records cover source, config, fixture, model, and script inputs that can affect the
   runner.
3. Missing, stale, or uncertain manifest data is documented as a full relevant parity gate trigger.
4. Manifest format is deterministic and documented for later quality-gate consumption.
5. Tests cover manifest generation and stale/missing-data semantics.

**Status:** Complete. Phase 151 added `parity_dependency_manifest/v1` with deterministic per-mode
records for tokenizer, GBNF, kernel, Jinja, and generation parity inputs. The manifest covers
source, config, fixture, model, script, and snapshot inputs, documents conservative freshness
semantics, and tests render/write behavior plus missing/stale/uncertain full-gate triggers.

### Phase 152: Parity Behavior And Lane-Isolation Closure

**Goal:** Close the milestone by proving existing parity modes still behave the same after the
refactor and that lane isolation is explicit, enforced, and source-backed.

**Requirements:** `PARITY-03`, `LANE-02`

**Success Criteria**:
1. Existing tokenizer, GBNF, kernel, Jinja, and generation parity modes continue to pass maintained
   fixture checks.
2. Source checks or tests fail on shared model/vocab/tokenizer/runtime/cache state across EMEL and
   reference lanes.
3. Dependency-manifest missing/stale behavior is validated as conservative rather than permissive.
4. Milestone documentation and requirement traceability match the implemented boundaries.
5. The relevant scoped quality gate passes, and no parity behavior change is reported as a
   boundary refactor without explicit approval.

**Status:** Complete. Phase 152 preserved maintained parity behavior under `paritychecker_tests`,
removed the reference-side direct dependency on EMEL detokenizer action detail, and added source
checks proving shared runner files stay free of lane runtime objects while tokenizer/generation
engine code keeps EMEL and reference model/vocab/runtime state separate.

### Phase 153: Parity Runner Config Ownership Closure

**Goal:** Close `PARITY-01` by making the shared parity runner own CLI/config parsing and request
normalization instead of leaving that behavior in `parity_main.cpp`.

**Depends on:** Phase 152

**Requirements:** `PARITY-01`

**Gap Closure:** Closes the audit finding `runner-boundary-cli-config`.

**Success Criteria**:
1. `tools/paritychecker` exposes a runner-owned config/CLI parsing boundary consumed by
   `parity_main.cpp`.
2. Existing usage text, validation failures, exit codes, and mode selection behavior are preserved
   or explicitly documented if changed.
3. Focused tests cover no-args usage, invalid option handling, `--kernel`, `--gbnf`, `--jinja`,
   tokenizer, and generation argument validation through the runner boundary.
4. Source checks fail if CLI/config parsing ownership drifts back into `parity_main.cpp`.

**Status:** Complete. Phase 153 moved usage text, argument parsing, text-file loading, and CLI
validation behind `run_parity_cli(...)`, leaving `parity_main.cpp` as a process shim. Focused
paritychecker tests and the changed-file scoped quality gate passed.

### Phase 154: Generation Live Reference Truth Closure

**Goal:** Close `PARITY-03` and `LANE-01` by making maintained generation parity compare EMEL output
against live reference-lane generation output on the normal success path.

**Depends on:** Phase 153

**Requirements:** `PARITY-03`, `LANE-01`

**Gap Closure:** Closes the audit findings `generation-reference-truth` and
`generation-live-reference-flow`.

**Success Criteria**:
1. Maintained generation parity invokes the live reference generation path for the normal comparison
   result instead of treating `baseline_record.result` as the reference truth.
2. Snapshot baselines remain append-only publication artifacts, not substitutes for live reference
   runtime proof; any intentional snapshot refresh is source-backed and documented.
3. EMEL and reference lanes continue to construct separate model, vocab, tokenizer, formatter,
   runtime, cache, and output state with tests/source checks guarding against object sharing.
4. Focused paritychecker tests cover live-reference success, deterministic missing-model failure,
   baseline mismatch/reporting behavior, and output schema stability.

**Status:** Complete. Maintained generation now compares EMEL output against live reference-lane
generation before baseline load. Snapshot baselines are secondary publication artifacts, and the
legacy non-current Qwen drift is reported instead of being masked by the stored baseline.

### Phase 155: Parity Actor Boundary Enforcement Closure

**Goal:** Close `LANE-02` by removing paritychecker dependence on actor `actions.hpp`,
`guards.hpp`, and `detail.hpp` internals except for approved kernel-owned arithmetic surfaces.

**Depends on:** Phase 154

**Requirements:** `LANE-02`

**Gap Closure:** Closes the audit finding `actor-helper-boundary`.

**Success Criteria**:
1. `tools/paritychecker` no longer includes or calls actor action/guard/detail helpers directly for
   loader, model, generator, tokenizer, formatter, or parser behavior.
2. Any needed diagnostics or lifecycle interactions are exposed through public actor events,
   public contracts, or kernel-owned execution surfaces rather than actor internals.
3. Source checks cover all paritychecker `.cpp`/`.hpp` files for broad actor-helper include and
   namespace patterns, not only selected generator/detokenizer/Jinja strings.
4. Focused paritychecker tests and the scoped quality gate pass after the boundary tightening.

**Status:** Pending.

### Phase 156: Parity Dependency Manifest Gate Closure

**Goal:** Close `MANIFEST-01` and `MANIFEST-02` by wiring dependency-manifest emission and
conservative freshness semantics into production paritychecker or quality-gate entrypoints.

**Depends on:** Phase 155

**Requirements:** `MANIFEST-01`, `MANIFEST-02`

**Gap Closure:** Closes the audit findings `manifest-emission-consumer` and
`manifest-to-gate-flow`.

**Success Criteria**:
1. Operators or gates have a maintained production path that emits
   `parity_dependency_manifest/v1` for every registered parity mode.
2. Missing, stale, or uncertain manifest data forces the relevant full parity gate in the
   maintained quality-gate path instead of remaining a test-only helper.
3. Manifest docs describe the production emission path and freshness decision path with examples.
4. Focused tests cover manifest emission, stale/missing/uncertain full-gate behavior, and unchanged
   parity mode execution behavior.

**Status:** Pending.
