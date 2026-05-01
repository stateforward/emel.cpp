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

## Current Milestone

v1.18 starts from GitHub issue #54. The milestone refactors `tools/paritychecker` into explicit
runner, engine, asset-loading, build-registration, and dependency-manifest boundaries while
preserving existing parity behavior and EMEL/reference lane isolation.

- [x] **Phase 148: Parity Runner Asset Boundary** - Establish the shared runner boundary and
  centralize asset/config/fixture normalization without changing mode behavior.
- [x] **Phase 149: Parity Engine Adapter Split** - Move tokenizer, GBNF, kernel, Jinja, and
  generation mode implementations behind explicit engine adapters.
- [x] **Phase 150: Parity Build Registration Boundary** - Make CMake and source registration
  modular enough for localized future engine additions.
- [x] **Phase 151: Parity Dependency Manifest Emission** - Emit per-runner dependency manifests
  with conservative stale/missing-data semantics.
- [ ] **Phase 152: Parity Behavior And Lane-Isolation Closure** - Prove all existing modes still
  behave the same and lane isolation remains enforceable.

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

**Status:** Pending.
