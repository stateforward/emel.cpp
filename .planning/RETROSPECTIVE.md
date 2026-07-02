# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.27 - Ryzen AVX2/FMA Kernel Support

**Shipped:** 2026-06-25
**Phases:** 6 | **Plans:** 6 | **Sessions:** autonomous execution, source-backed audit, and closeout

### What Was Built

- x86_64 host feature contract for AVX2, FMA, and F16C on the Ryzen 9 5950X.
- EMEL-owned AVX2/FMA/F16C flash-attention path with explicit fallback/no-claim behavior.
- EMEL-owned AVX2/FMA q2_K/q3_K/q6_K x q8_K hot-path kernels.
- Maintained generator and paritychecker attribution proving optimized x86_64 dispatch.
- `kernel_x86_64` benchmark publication with counter-checked optimized flash and q2/q3/q6 rows.

### What Worked

- The source-backed audit caught a real benchmark-publication gap after phase artifacts looked green.
- Counter checks in benchmarks made optimized/shared attribution mechanically enforceable.
- Keeping x86_64 routing in explicit SML guards/transitions made the unary rule-debt repair small.

### What Was Inefficient

- The first benchmark snapshot update only covered common x86_64 rows and had to be repaired before closeout.
- The quality gate coverage shard was slow under coverage instrumentation, so closeout needed long-running observation.

### Patterns Established

- Benchmark parity claims need counter-backed maintained entries for each optimized lane, not only suite-level presence.
- x86_64 support should mirror NEON by proving host contract, native kernels, runtime attribution, parity, and publication as one slice.
- Pre-close audits should separate current milestone blockers from historical backlog artifacts before archiving.

### Key Lessons

1. Artifact agreement is not enough for benchmark truth; source entrypoints and counters must match the claim.
2. Runtime behavior selection debt can survive in helper APIs even when production transitions are explicit.
3. Snapshot updates need a second source-backed pass when they create a new benchmark suite.

### Cost Observations

- Model mix: not measured.
- Sessions: one long autonomous closeout session with an integration-checker agent.
- Notable: `commit_docs=false` left archive and planning changes local instead of committing them.

---

## Milestone: v1.25 - I/O Read Loading Strategy

**Shipped:** 2026-05-06
**Phases:** 15 | **Plans:** 20 | **Sessions:** autonomous audit, cleanup, and closeout

### What Was Built

- Canonical `src/emel/io/read` Stateforward.SML actor for read/copy tensor loading.
- RTC-safe source-span read/copy execution into caller-owned target buffers.
- Tensor-owned public `request_read_load` route through injected `io/read`.
- Maintained model-loader read/copy evidence through public source loading, tensor plan/apply,
  and `io/loader -> io/read`.
- Final closeout artifacts distinguishing the direct tensor result-carrier path from maintained
  model-loader lanes.

### What Worked

- The final integration check caught an overbroad audit claim before archive.
- Phase 224 was useful as a cleanup-only phase because it closed ambiguity without changing source.
- Re-running `emel_tests_io` after the transient dyld launch failure produced fresh passing evidence.

### What Was Inefficient

- Milestone archive automation counted only the reopened active phase after earlier phase archival,
  leaving duplicate/stale planning prose that needed manual cleanup.
- The audit truth around direct `request_read_load` versus maintained model-loader lanes required a
  second integration pass to phrase precisely.

### Patterns Established

- Closeout audits must distinguish direct public actor routes from maintained benchmark/parity lanes
  when they use different same-RTC handoff mechanisms.
- Archived phase directories should be folded under `.planning/milestones/vX.Y-phases/` before
  treating the active roadmap as clean.

### Key Lessons

1. Source-backed wording needs to name the exact lane it applies to.
2. Transient local launch failures should be rerun before becoming archive-time tech debt.
3. Milestone completion should be followed by a consistency check after phase-directory cleanup.

### Cost Observations

- Model mix: not measured.
- Sessions: one autonomous closeout session with verifier and integration-checker agents.
- Notable: `commit_docs=false` skipped all planning commits, so archive changes remain local.

---

## Milestone: v1.22 - Weight Loading Ownership Cutover

**Shipped:** 2026-05-03
**Phases:** 10 | **Plans:** 10 | **Sessions:** autonomous audit, gap closure, and closeout

### What Was Built

- Tensor load and residency ownership moved under `src/emel/model/tensor`.
- The maintained model loader now coordinates tensor-owned bind, plan, and apply behavior through
  public tensor actor events.
- The retired `model/weight_loader` owner path was removed from source, tests, CMake wiring, docs,
  and lint baselines.
- Loader tensor bulk outcomes now route through explicit bind, plan, and apply decision states
  instead of local callback flag capture.
- Public roadmap prose and domain-boundary guardrails now keep concrete I/O strategy work deferred
  under the future `emel/io` seam.
- Maintained generation benchmark, Sortformer benchmark, embedded-size probe, and paritychecker
  lanes now prebind GGUF KV storage before model-loader dispatch.

### What Worked

- The milestone audit caught two real closeout contradictions that earlier artifacts had claimed as
  complete.
- Adding Phases 192, 193, and 194 as gap-closure phases kept the repair scope narrow and
  source-backed.
- The final changed-file quality gates were able to scope benchmark work to the maintained
  generation and Sortformer entrypoints.
- User-approved snapshot/model/benchmark updates avoided artificial blockers around lint and
  benchmark evidence.

### What Was Inefficient

- The first Phase 194 quality-gate run used a space-separated changed-file list, which the script
  treated as one path and forced a rerun with colon-separated files.
- The all-suite benchmark gate timed out before the milestone-relevant suite override was applied.
- Header-only coverage needed additional direct callback and guard coverage to satisfy the required
  threshold.
- The archive tool created an empty accomplishments section, requiring manual MILESTONES cleanup.

### Patterns Established

- For `scripts/quality_gates.sh`, pass changed files as colon, comma, or newline-separated values.
- Loader/tensor ownership audits should include source scans for local capture structs and stale
  public prose, not only exact retired paths.
- Maintained-path audits should trace allocations inside callbacks reached from
  `model_loader.process_event(...)`, not only source ownership boundaries.
- Same-RTC callback result handoff can be acceptable when the transition graph still performs the
  outcome selection through explicit guards and states.
- Public docs need semantic guardrails when a retired owner has human-readable names that do not
  appear as exact source identifiers.

### Key Lessons

1. Closeout claims about retired runtime ownership need a semantic public-doc check, not only source
   path checks.
2. Header-only actor changes should add focused unit coverage for callback, guard, and error-mapping
   contracts before running the full quality gate.
3. Archive automation still needs manual review for accomplishments and current-state wording.
4. Benchmark snapshot updates should use the maintained update command and then rerun the scoped
   gate without `EMEL_QUALITY_GATES_ALLOW_BENCH_REGRESSION`.

### Cost Observations

- Model mix: not measured.
- Sessions: one autonomous audit/gap/closeout session.
- Notable: the corrected Phase 194 scoped quality gate ran generation and diarization Sortformer
  benchmarks plus full paritychecker validation after the generation snapshot update.

---

## Milestone: v1.15 - ARM Sortformer Diarization GGUF Slice

**Shipped:** 2026-04-25
**Phases:** 24 | **Plans:** 24 | **Sessions:** autonomous execution and closeout

### What Was Built

- One maintained `openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf` ARM diarization
  slice with pinned fixture provenance and explicit loader/model acceptance.
- A diarization-owned mono 16 kHz PCM request contract, native feature preparation, native
  Sortformer encoder/executor path, and deterministic probability/segment publication.
- Lane-isolated PyTorch/NeMo parity and ONNX CPU single-thread benchmark reference lanes.
- A strict generated benchmark record where EMEL exact-matches PyTorch/NeMo and ONNX and beats
  ONNX CPU single-thread.
- Source-backed closeout artifacts tracing claims from fixture through loader, runtime, parity,
  benchmark, docs, and milestone audit.

### What Worked

- The milestone did not close until synthetic/tool-only evidence was replaced by maintained runtime
  evidence.
- Treating PyTorch/NeMo as the parity reference and ONNX as the benchmark reference made the proof
  roles explicit and reviewable.
- Recursive profiling focused optimization on measured hotspots instead of guessing.
- Kernel/runtime ownership rules prevented a second ad hoc diarization compute lane from becoming
  permanent.

### What Was Inefficient

- Early closeout evidence overstated benchmark/parity truth and forced multiple inserted repair
  phases.
- Full quality gates and benchmark regeneration consumed too much wall-clock time during the final
  loop.
- The `audit-open` CLI had a wrapper bug (`output` instead of `core.output`) that blocked closeout
  until repaired.
- Summary one-liner extraction produced low-quality `Completed:` accomplishments, requiring manual
  MILESTONES cleanup.

### Patterns Established

- Milestone benchmark claims must be verified by tracing the live maintained codepath, not by
  trusting ROADMAP, SUMMARY, VERIFICATION, or generated docs alone.
- Reference lanes should have named roles: parity reference, benchmark reference, or diagnostic
  reference.
- Performance closeout records must include provider/thread contract, output identity, checksum,
  and generated record paths.
- If optimized kernel work is required for truth, add phases until it is in the kernel/runtime
  domain rather than keeping parallel helpers.

### Key Lessons

1. Source-backed audit rules need to run before milestone closeout, not after the first closeout
   attempt.
2. Benchmark gates need deterministic generated reports, but the slowest docs/gate steps should be
   profiled and optimized as tooling debt.
3. Summary frontmatter should contain useful one-liners so archive automation does not need manual
   repair.

### Cost Observations

- Model mix: not measured.
- Sessions: autonomous execution plus repeated scheduled Phase 93 optimization ticks.
- Notable: final scoped quality gate timing was `246s`, with docs generation dominating at `211s`.

---

## Milestone: v1.13 - Pluggable Generative Parity Bench

**Shipped:** 2026-04-21
**Phases:** 8 | **Plans:** 8 | **Sessions:** current autonomous closeout

### What Was Built

- One canonical `generation_compare/v1` contract for EMEL and reference generation lanes.
- Manifest-pinned maintained generation workloads with prompt, formatter, sampling, stop, seed,
  and comparability metadata.
- A maintained `llama_cpp_generation` reference backend behind operator-facing wrapper tooling.
- Truthful compare publication for exact matches, bounded drift, non-comparable runs, missing
  lanes, and errors.
- Regression coverage for both comparable LFM2 workflow publication and selected single-lane
  non-comparable publication.

### What Worked

- The audit gap workflow was useful: the initial audit exposed real behavior and evidence gaps,
  then Phases 74 through 76 closed them without broadening the milestone.
- Keeping reference backend setup in wrapper/tooling space preserved the EMEL runtime boundary.
- Requirement and Nyquist backfills made the final audit mechanically checkable instead of relying
  on narrative closeout.

### What Was Inefficient

- Early closeout summaries did not expose `requirements-completed` consistently, so audit tooling
  could not infer accomplishment and requirement coverage cleanly.
- The `audit-open` CLI path has a tooling bug in its direct command wrapper, requiring a manual
  call into the audit library during closeout.
- Metadata mismatch tests cover representative fields directly while implementation checks the full
  tuple; this is acceptable but leaves coverage debt.

### Patterns Established

- Comparable and non-comparable generative workloads should both flow through the same
  operator-facing wrapper, with the verdict carrying the truth rather than the entrypoint changing.
- Material comparability metadata must be checked before output comparison.
- Single-lane publication should leave the absent lane's raw record file empty and explain the
  non-comparable reason explicitly.

### Key Lessons

1. Audit evidence should be written at the same time as behavior work, not reconstructed at the end.
2. Wrapper-level tests catch integration claims that JSONL-only tests can miss.
3. Closeout artifacts need stable frontmatter keys so milestone tooling can summarize without
   hand repair.

### Cost Observations

- Model mix: not measured.
- Sessions: current autonomous closeout plus phase execution sessions.
- Notable: final audit did not rerun the full quality gate; it relied on the Phase 75 post-review
  full gate pass.

---

## Milestone: v1.18 - Parity Tool Boundary Refactor

**Shipped:** 2026-05-01
**Phases:** 5 | **Plans:** 5 | **Sessions:** autonomous execution and closeout

### What Was Built

- A shared paritychecker asset boundary for repo paths, byte loading, and maintained generation
  fixture resolution.
- Explicit tokenizer, GBNF, kernel, Jinja, and generation engine adapters behind one runner-facing
  registration surface.
- Modular paritychecker CMake source groups shared by the executable and tests.
- Deterministic `parity_dependency_manifest/v1` records with conservative full-gate semantics for
  missing, stale, or uncertain data.
- Lane-isolation source checks and a reference-side byte-token parser that avoids EMEL actor action
  detail helpers.

### What Worked

- Keeping the runner small made later engine/build/manifest boundaries easy to audit.
- Focused source checks caught the exact regression risks for this milestone without changing
  parity output semantics.
- The archive safety commit before removing `REQUIREMENTS.md` gave a clean recovery point.

### What Was Inefficient

- `gsd-tools init milestone-op` still returned stale `v1.0` metadata, which required manual
  closeout repair for `STATE.md` and completion commands.
- Summary one-liner extraction again produced empty accomplishments, so `MILESTONES.md` needed
  manual cleanup.
- The quality gate timing snapshot was rewritten by scoped runs and had to be restored repeatedly.

### Patterns Established

- Parity tool refactors should add source-backed boundary tests before moving behavior.
- Dependency manifests should be conservative evidence only; unknown freshness means run the full
  relevant gate.
- Shared runner code should stay free of lane-owned runtime objects; mode engines own lane setup.

### Key Lessons

1. Milestone closeout tooling needs to trust ROADMAP/STATE only after validating the current
   milestone identity.
2. Source checks are useful when they target a narrow contract and avoid pretending to prove every
   future misuse.
3. Scoped quality gates should avoid persisting timing snapshot churn unless the user explicitly
   approves a snapshot update.

### Cost Observations

- Model mix: not measured.
- Sessions: one autonomous execution/closeout session after issue-based milestone initialization.
- Notable: focused paritychecker tests gave fast confidence for tool-boundary work without running
  unrelated benchmark suites.

---

## Milestone: v1.19 - Benchmark Tool Pluggable Runner Refactor

**Shipped:** 2026-05-01
**Phases:** 10 | **Plans:** 10 | **Sessions:** autonomous execution, reopened gap closure, and closeout

### What Was Built

- Shared benchmark orchestration behind `emel::bench::run_bench_cli(...)`, with `bench_main.cpp`
  left as a process shim.
- Deterministic `bench_runner_request/v1` and `bench_runner_result/v1` contracts wired through a
  live `bench_runner` process-level runner seam.
- Localized benchmark suite registration and lookup through `bench_runner_registry.hpp` / `.cpp`.
- Per-suite `bench_runner_suite_<suite>` CMake object targets for maintained runner source
  isolation.
- Deterministic `bench_dependency_manifest/v1` records, a checked-in baseline, write/check CLI
  operations, and manifest-driven quality-gate escalation.
- Source-backed generation and diarization behavior coverage plus shared runner lane-isolation and
  actor-boundary checks.
- Reopened closure phases closed the live process-seam, maintained runner actor-boundary, and
  Nyquist validation artifact gaps found by source-backed audit.

### What Worked

- Reusing the v1.18 paritychecker boundary pattern made the benchmark runner refactor predictable.
- The phase order kept behavior-preservation proof last, after orchestration, contract, registry,
  build, and manifest boundaries were already source-backed.
- Changed-file scoped quality gates stayed focused on benchmark work by passing the generation
  benchmark suite explicitly.
- The audit was able to rely on source tests and verification artifacts instead of roadmap claims.

### What Was Inefficient

- The milestone archive command still produced an empty accomplishments section that needed manual
  repair in `MILESTONES.md`.
- `STATE.md` kept a stale v1.18 closeout sentence after archiving v1.19.
- Quality-gate runs rewrote `snapshots/quality_gates/timing.txt`, requiring repeated restoration
  because the run was not a snapshot update.
- Full `bench_runner_tests` took several minutes when the build directory had to be reconfigured
  away from the generation-only suite filter.

### Patterns Established

- Benchmark and parity tool refactors should share the same boundary progression: runner shim,
  contract, registration, build isolation, dependency manifest, quality-gate consumption, then
  maintained behavior closure.
- Shared orchestration source checks should stay narrow to the shared boundary and not claim to
  replace suite-owned implementation tests.
- Dependency manifests should be treated as conservative routing evidence; missing, stale, or
  uncertain data means run the relevant benchmark gate.

### Key Lessons

1. Reconfigure benchmark builds explicitly after suite-filtered quality gates before running tests
   that expect multiple maintained suites.
2. Milestone audit files should be committed before archival so complete-milestone can move them
   with history.
3. Closeout automation needs manual review for generated accomplishments and stale state text.

### Cost Observations

- Model mix: not measured.
- Sessions: one autonomous issue-based milestone execution and closeout session.
- Notable: final `bench_runner_tests` passed after full benchmark-tool reconfiguration; the final
  scoped quality gate passed with benchmark manifest freshness checked.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.19 | autonomous execution/reopened closeout | 10 | Benchmark runner boundaries now mirror paritychecker boundaries with manifests, live process-seam proof, and source-backed lane-isolation enforcement. |
| v1.18 | autonomous execution/closeout | 5 | Paritychecker boundaries became explicit runner, engine, build, manifest, and lane-isolation contracts. |
| v1.15 | autonomous execution/closeout | 24 | Source-backed audit and recursive profiling became mandatory before milestone archive. |
| v1.13 | current closeout | 8 | Audit gaps became explicit repair phases before archive. |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| v1.19 | `bench_runner_tests`, `quality_gates_tests`, changed-file scoped quality gates | Coverage gate skipped because no `src/emel` files changed | No new runtime dependency in `src/`; benchmark manifests are source-controlled. |
| v1.18 | `paritychecker_tests`, focused paritychecker executable builds, changed-file scoped quality gates | Coverage gate skipped because no `src/emel` files changed | No new runtime dependency in `src/`; manifest records are source-controlled. |
| v1.15 | Sortformer runtime/parity/benchmark tests plus final scoped quality gate | Line coverage gate preserved; scoped gate timing `246s` | No external runtime dependency in the EMEL lane. |
| v1.13 | `generation_compare_tests`, focused `bench_runner_tests`, quality gates | 90.4% line coverage at last full gate | No new runtime dependency in `src/`. |

### Top Lessons

1. Maintain lane isolation as a testable contract, not just a design statement.
2. Publish non-comparable outcomes explicitly when workload or metadata contracts diverge.
3. Keep milestone closeout files machine-readable enough for audit tooling.
4. Treat benchmark performance claims as source-backed claims that must trace through the live
   maintained runtime path.
5. Treat dependency manifests as conservative gate evidence, never as permissive skip evidence.
