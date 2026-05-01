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

- [x] [v1.19: Benchmark Tool Pluggable Runner Refactor](.planning/milestones/v1.19-ROADMAP.md)
  - Shipped 2026-05-01 after reopened source-backed gap closure through Phases 164-166; final
    audit passed with 13/13 active requirements satisfied.

## Current Milestone

## v1.20 SML Dependency And Namespace Migration

**Goal:** Upgrade EMEL to the current `stateforward/sml.cpp` dependency and migrate
project-owned code/docs from the legacy SML surface to `stateforward::sml` without weakening
actor-model rules or maintained parity/benchmark evidence.

**Source:** GitHub issue #56

**Requirements:** 12 active requirements, all mapped.

| Phase | Name | Goal | Requirements |
|-------|------|------|--------------|
| 167 | SML Upstream Pin And Surface Audit | Pin the intended upstream commit and prove the new SML include/namespace surface before broad migration. | DEP-01, DEP-02, DEP-03 |
| 168 | Project-Owned Source Namespace Migration | Move active EMEL source, headers, tests, and tools to the preferred `stateforward` SML include and namespace surface. | SRC-01, SRC-02 |
| 169 | SML Orchestration Behavior Preservation | Prove transition tables, dispatch tables, unexpected-event handling, loggers, and state inspection still behave after migration. | SRC-03 |
| 170 | SML Rules And Documentation Migration | Update contributor rules, docs, examples, docsgen, and planning guidance so active instructions no longer point at legacy naming. | DOC-01, DOC-02, DOC-03 |
| 171 | Legacy SML Reference Guardrails | Add source checks and scoped quality gates that prevent unapproved legacy SML drift while preserving maintained behavior. | VAL-01, VAL-02 |
| 172 | v1.20 Source-Backed Closeout | Run source-backed audit and close the milestone only after all active migration requirements are proven. | VAL-03 |

### Phase 167: SML Upstream Pin And Surface Audit

**Goal:** Pin the intended upstream `stateforward/sml.cpp` commit and establish the migration
contract before changing broad code.

**Requirements:** DEP-01, DEP-02, DEP-03

**Success criteria:**
1. `cmake/sml_version.cmake` pins the intended newer upstream commit from issue #56.
2. The build consumes the newer SML dependency without relying on an unreviewed moving branch.
3. The preferred upstream include path and `stateforward::sml` namespace are verified in a focused
   compile/test target.
4. Any temporary reliance on legacy compatibility shims is documented with a bounded follow-up.

### Phase 168: Project-Owned Source Namespace Migration

**Goal:** Convert active project-owned code to the preferred `stateforward` SML surface.

**Requirements:** SRC-01, SRC-02

**Success criteria:**
1. Active `src/`, `include/`, `tests/`, and `tools/` SML include sites use the preferred upstream
   include path where supported.
2. Active project-owned legacy SML namespace references are migrated to `stateforward::sml`
   without changing transition-table semantics.
3. Existing canonical machine aliases remain stable for EMEL callers.
4. Focused build/test coverage passes for representative SML machines after the rename.

### Phase 169: SML Orchestration Behavior Preservation

**Goal:** Prove the namespace migration did not alter actor orchestration behavior.

**Requirements:** SRC-03

**Success criteria:**
1. Transition-table tests still cover destination-first rows, completion/internal transitions, and
   unexpected-event behavior after migration.
2. Dispatch-table and state-inspection call sites compile and run against the new namespace.
3. Logger and test-introspection surfaces still work without falling back to legacy includes.
4. Maintained runtime smoke tests pass through the migrated SML surface.

### Phase 170: SML Rules And Documentation Migration

**Goal:** Align active contributor guidance and generated docs with the new SML naming.

**Requirements:** DOC-01, DOC-02, DOC-03

**Success criteria:**
1. `docs/rules/sml.rules.md`, `AGENTS.md`, and active planning guidance instruct new work to use
   `stateforward::sml`.
2. Docsgen and examples emit or check the new namespace consistently.
3. Historical legacy SML references are either archival context or explicitly exempted.
4. Documentation checks pass after the migration.

### Phase 171: Legacy SML Reference Guardrails

**Goal:** Prevent accidental legacy SML naming drift after the migration.

**Requirements:** VAL-01, VAL-02

**Success criteria:**
1. Source checks fail on new unapproved active legacy SML namespace or include-path
   include references.
2. Changed-file scoped quality gates run against the migration files without weakening parity,
   benchmark, coverage, or docs lanes.
3. Any allowlist is small, documented, and limited to historical or upstream compatibility
   contexts.
4. Maintained domain-boundary checks remain green after the guardrail update.

### Phase 172: v1.20 Source-Backed Closeout

**Goal:** Close v1.20 only after source-backed evidence proves the migration is complete.

**Requirements:** VAL-03

**Success criteria:**
1. A milestone audit traces requirements to actual code, docs, checks, and quality-gate evidence.
2. All 12 active requirements are mapped, verified, and either complete or explicitly reopened
   before closeout.
3. Benchmark/parity claims remain source-backed and do not depend on synthetic or stale artifacts.
4. `v1.20` closeout artifacts archive the milestone with the next action clearly stated.
