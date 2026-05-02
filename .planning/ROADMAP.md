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

**Requirements:** 12 active requirements, all 12 satisfied after Phase 179 repaired the
closeout evidence reproducibility gap and reran the source-backed full milestone gate on
2026-05-02.

| Phase | Name | Goal | Requirements |
|-------|------|------|--------------|
| 167 | SML Upstream Pin And Surface Audit | Pin the intended upstream commit and prove the new SML include/namespace surface before broad migration. | DEP-01, DEP-02, DEP-03 |
| 168 | Project-Owned Source Namespace Migration | Move active EMEL source, headers, tests, and tools to the preferred `stateforward` SML include and namespace surface. | SRC-01, SRC-02 |
| 169 | SML Orchestration Behavior Preservation | Prove transition tables, dispatch tables, unexpected-event handling, loggers, and state inspection still behave after migration. | SRC-03 |
| 170 | SML Rules And Documentation Migration | Update contributor rules, docs, examples, docsgen, and planning guidance so active instructions no longer point at legacy naming. | DOC-01, DOC-02, DOC-03 |
| 171 | Legacy SML Reference Guardrails | Add source checks and scoped quality gates that prevent unapproved legacy SML drift while preserving maintained behavior. | VAL-01, VAL-02 |
| 172 | v1.20 Source-Backed Closeout | Run source-backed audit and close the milestone only after all active migration requirements are proven. | VAL-03 |
| 173 | SML Migration Evidence Reconstruction | Reconstruct source-backed phase evidence for the dependency pin and source namespace migration gaps found by audit. | DEP-01, DEP-02, DEP-03, SRC-01, SRC-02 |
| 174 | SML Orchestration Surface Proof | Add or restore live proof for logger, dispatch-table, unexpected-event, and state-inspection behavior after the namespace migration. | SRC-03 |
| 175 | SML Documentation Rule Path Repair | Repair stale rule-path guidance and prove docs/examples/planning guidance no longer conflict on the migrated SML surface. | DOC-01, DOC-02, DOC-03 |
| 176 | Legacy SML Guardrail And Quality Gate Repair | Wire maintained legacy SML drift checks and restore scoped quality-gate coverage without weakening required lanes. | VAL-01, VAL-02 |
| 177 | v1.20 Final Source-Backed Closeout Rerun | Backfill Nyquist evidence and rerun source-backed audit after all reopened v1.20 gaps are closed. | VAL-03 |
| 178 | v1.20 Closeout Gate And Evidence Repair | Resolve the blocked full closeout gate, repair contradicted closeout evidence, and produce final source-backed VAL-03 artifacts. | VAL-03 |
| 179 | v1.20 Closeout Evidence Reproducibility Repair | Repair stale closeout claims and make the bench tooling validation reproducible from a maintained build command before final audit rerun. | VAL-01, VAL-03 |

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

### Phase 173: SML Migration Evidence Reconstruction

**Goal:** Reconstruct source-backed evidence for the dependency pin and source namespace migration
requirements reopened by `.planning/v1.20-MILESTONE-AUDIT.md`.

**Requirements:** DEP-01, DEP-02, DEP-03, SRC-01, SRC-02

**Gap Closure:** Closes audit gaps for missing Phase 167/168 SUMMARY.md, VERIFICATION.md, and
VALIDATION.md evidence, plus source-backed provenance for the current and target SML pin.

**Success criteria:**
1. Source-backed evidence identifies the previous EMEL SML pin, the target upstream commit, and the
   live `cmake/sml_version.cmake` pin.
2. Active source/test/tool include and namespace migration evidence is captured from live repo
   scans, not planning artifacts alone.
3. Any compatibility exception or absence of exceptions is documented with a bounded allowlist.
4. Phase artifacts record the commands and files proving DEP-01, DEP-02, DEP-03, SRC-01, and
   SRC-02.

### Phase 174: SML Orchestration Surface Proof

**Goal:** Prove all SML behavior surfaces named by SRC-03 compile and behave through the migrated
`stateforward::sml` surface.

**Requirements:** SRC-03

**Gap Closure:** Closes audit findings for docs-only logger/dispatch-table evidence and incomplete
live behavior proof.

**Success criteria:**
1. Active tests or source-backed proof cover transition tables, completion/internal transitions,
   `sml::unexpected_event`, state inspection, logger wiring, and dispatch-table usage.
2. Any intentionally unused upstream surface is explicitly documented and removed from the active
   requirement wording or proved by a focused compile/test surface.
3. Maintained runtime smoke tests pass through the migrated SML surface.
4. Phase artifacts capture the focused commands and source locations used as evidence.

### Phase 175: SML Documentation Rule Path Repair

**Goal:** Remove conflicting SML rule-path guidance and prove active documentation consistently
points contributors at `docs/rules/sml.rules.md` and `stateforward::sml`.

**Requirements:** DOC-01, DOC-02, DOC-03

**Gap Closure:** Closes audit findings for stale `docs/sml.rules.md` references and incomplete
archival/exemption evidence.

**Success criteria:**
1. `AGENTS.md`, active plans, generated-doc tooling, and contributor docs reference the normative
   `docs/rules/sml.rules.md` path where applicable.
2. Active examples and docsgen output use `stateforward::sml` and the preferred include path.
3. Historical legacy SML references are archival, quoted, or explicitly exempted.
4. Documentation and lint snapshot checks pass after the repair.

### Phase 176: Legacy SML Guardrail And Quality Gate Repair

**Goal:** Add maintained drift checks and restore quality-gate coverage for the v1.20 migration.

**Requirements:** VAL-01, VAL-02

**Gap Closure:** Closes audit findings for missing legacy SML source-check wiring, disabled
`lint_snapshot` in `scripts/quality_gates.sh`, and failed scoped benchmark-gate selection.

**Success criteria:**
1. A maintained source check fails on unapproved active legacy SML include or namespace references.
2. `scripts/quality_gates.sh` runs the required lint/docs/parity/benchmark/coverage lanes or
   records an explicitly approved, bounded exception with equivalent maintained enforcement.
3. The changed-file scoped quality gate passes for the migration and guardrail files without
   weakening relevant lanes.
4. Phase artifacts capture the gate output and source-check evidence.

### Phase 177: v1.20 Final Source-Backed Closeout Rerun

**Goal:** Complete v1.20 only after all reopened gaps have source-backed evidence and Nyquist
validation.

**Requirements:** VAL-03

**Gap Closure:** Closes audit findings for missing closeout evidence, missing validation artifacts,
and the failed milestone audit.

**Success criteria:**
1. SUMMARY.md, VERIFICATION.md, and VALIDATION.md artifacts exist for all v1.20 closure phases.
2. The milestone audit rerun traces requirements to live code, docs, checks, and quality-gate
   evidence.
3. All 12 active requirements are complete with no orphaned verification rows.
4. `v1.20` closeout artifacts state the final status and next action.

### Phase 178: v1.20 Closeout Gate And Evidence Repair

**Goal:** Close the remaining VAL-03 audit gaps by resolving the full closeout benchmark timeout,
repairing contradicted closeout evidence, and producing the final source-backed closeout artifacts.

**Requirements:** VAL-03

**Gap Closure:** Closes `.planning/v1.20-MILESTONE-AUDIT.md` findings for the blocked full
quality-gate benchmark comparison, missing Phase 177 closeout artifacts, and contradicted Phase 172
VAL-03 completion claim.

**Success criteria:**
1. The benchmark comparison timeout is reproduced or isolated with the narrowest command that still
   exercises the required full benchmark comparison contract.
2. The maintained closeout path either completes `EMEL_QUALITY_GATES_SCOPE=full
   scripts/quality_gates.sh` successfully or records an explicitly approved, source-backed
   closeout path that does not weaken benchmark, parity, coverage, fuzz, docs, or lint requirements.
3. Phase 172/177 closeout evidence is superseded or repaired so no artifact claims VAL-03 complete
   before the final source-backed validation passes.
4. Final SUMMARY.md, VERIFICATION.md, and VALIDATION.md artifacts exist for the authoritative
   closeout phase, and the milestone audit rerun reports all 12 active requirements satisfied.

### Phase 179: v1.20 Closeout Evidence Reproducibility Repair

**Goal:** Close the remaining VAL-01 and VAL-03 audit gaps by making the closeout validation
reproducible from maintained commands and repairing stale closeout artifacts.

**Requirements:** VAL-01, VAL-03

**Gap Closure:** Closes `.planning/v1.20-MILESTONE-AUDIT.md` findings for the current
`bench_runner_tests` failure under `build/bench_tools_ninja`, the tokenizer-filtered bench-tools
build cache, and the stale Phase 172 VAL-03 completion claim.

**Success criteria:**
1. The bench tooling validation uses a maintained build command whose suite filter cannot hide the
   generation, diarization, and batch suites required by `bench_runner_tests`.
2. `ctest --test-dir build/bench_tools_ninja -R 'quality_gates_tests|bench_runner_tests'
   --output-on-failure` passes from the documented maintained build state.
3. Phase 172 closeout artifacts are repaired or superseded so they no longer claim VAL-03 complete
   before the authoritative final closeout evidence.
4. The final full closeout validation and milestone audit rerun pass with benchmark, parity,
   coverage, fuzz, docs, and lint lanes intact. Benchmark and snapshot updates are explicitly
   approved for this closure phase when required for truthful maintained evidence.
