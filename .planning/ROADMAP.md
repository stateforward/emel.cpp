# Roadmap: EMEL

## Milestones

- ✅ **v1.25 I/O Read Loading Strategy** — shipped 2026-05-06
- ✅ **v1.26 I/O Staged Read Loading Strategy** — shipped 2026-05-08
- ⚠️ **v1.27 co_sm Cooperative Async I/O Strategy** — gap closure active 2026-05-09,
  issue #64

## Phases

### ⚠️ v1.27 co_sm Cooperative Async I/O Strategy (Phases 239-252) — GAP CLOSURE ACTIVE

Source: GitHub issue #64, "Add co_sm-based cooperative async io strategy for resumable tensor
loading". Adds the first approved `co_sm` / coroutine actor usage for a separate cooperative async
I/O strategy under the existing `emel/io` boundary, while tensor residency remains owned by
`model/tensor`.

Execution order: 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
252.

**Milestone progress (v1.27):** **10 / 14** phases complete; milestone audit reopened gap
closure phases 249-252 for runtime scheduler error semantics, public loader resumability,
evidence consistency, and large-model constrained-RAM profiling.

**Current audit status:** `.planning/v1.27-MILESTONE-AUDIT.md` found partial `AIO-06`,
`DOC-01`, and `EVI-01` coverage plus a maintained-loader resumability caveat affecting
`AIO-04`, `TNX-03`, and `PERF-01`. Phase 248 recorded initial maintained
`cooperative_async` execution evidence (`488169958 ns/op` EMEL vs `353585500 ns/op`
reference, `1.381x`), but Phase 250 must renew the evidence after preserving loader-level
partial progress.

- [x] Phase 239: Coroutine Actor Contract (CO-01)
- [x] Phase 240: co_sm Wrapper, Scheduler, and Allocation Proof (CO-02, CO-03, CO-04, CO-05, TST-01)
- [x] Phase 241: Async I/O Strategy Component Boundary (AIO-01, AIO-02)
- [x] Phase 242: Suspension-Safe Request and Progress Ownership (AIO-03, OWN-01, OWN-02, OWN-03, OWN-04)
- [x] Phase 243: Suspend/Resume Progress and Error Semantics (AIO-05, TST-02; AIO-04 and
      AIO-06 reopened for Phases 250 and 249)
- [x] Phase 244: Tensor-Owned Async Integration Graph (TNX-01, TNX-02, TST-03; TNX-03 reopened
      for Phase 250)
- [x] Phase 245: Public Loader and Maintained Entrypoints (TNX-04)
- [x] Phase 246: Scope and Non-Regression Guardrails (GRD-01, GRD-02, GRD-03, GRD-04)
- [x] Phase 247: Publication and Evidence Truthfulness (LNT-01, QG-01; DOC-01 and EVI-01
      reopened for Phase 251)
- [x] Phase 248: Maintained Cooperative Async E2E Execution Path (initial PERF-01 evidence;
      renewed PERF-01 evidence required by Phase 250)
- [ ] Phase 249: Runtime Scheduler Error Contract Repair (AIO-06)
- [ ] Phase 250: Public Loader Resumable Async Progress (AIO-04, TNX-03, PERF-01)
- [ ] Phase 251: Milestone Evidence Consistency Repair (DOC-01, EVI-01)
- [ ] Phase 252: Large-Model Constrained-RAM Profiling And Optimization (PERF-02)

#### Phase 239: Coroutine Actor Contract

**Goal:** Codify how `co_sm` coroutine actors fit EMEL's RTC/no-queue model before any runtime
behavior depends on coroutine dispatch.  
**Depends on:** Phase 238  
**Requirements:** CO-01
**Plans:** 01 — Completed 2026-05-09; added explicit `co_sm` coroutine actor rules to
`docs/rules/sml.rules.md`, `AGENTS.md`, and `CLAUDE.md`.

**Success criteria:**

1. `docs/rules/sml.rules.md` and `AGENTS.md` contain an explicit coroutine actor contract covering
   RTC boundaries, continuation semantics, no hidden mailboxes, allocation, callbacks, and lifetime.
2. The contract states that coroutine continuations are internal actor/scheduler progress, not
   public queued messages or retained external events.
3. Rule text clearly excludes retained stack-backed request data, stored callbacks, public C ABI
   coroutine types, and behavior selection in awaitables/actions/detail helpers.
4. Existing synchronous actor rules remain intact and do not silently permit general async behavior
   outside the dedicated `co_sm` contract.

#### Phase 240: co_sm Wrapper, Scheduler, and Allocation Proof

**Goal:** Add the opt-in project-owned coroutine machine surface and prove scheduler/allocation
basics before async I/O component work.  
**Depends on:** Phase 239  
**Requirements:** CO-02, CO-03, CO-04, CO-05, TST-01
**Plans:** 01 — Completed 2026-05-09; added `emel::co_sm`, policy aliases, fixed allocator,
and public wrapper doctests.

**Success criteria:**

1. `emel::co_sm` exists parallel to `emel::sm`, preserves synchronous `emel::sm` behavior, and is
   opt-in only.
2. Scheduler policy contracts statically require FIFO, single-consumer, and run-to-completion
   guarantees.
3. Coroutine frame/scheduler storage is fixed-capacity or setup-time-owned for the milestone path,
   with no heap allocation during dispatch/progress tests.
4. Public machine tests prove deterministic `co_sm` dispatch, suspend/resume ordering, state
   inspection, and same-tick run-to-completion behavior.
5. Source checks show no `co_sm` task/scheduler/awaitable types leak through public C ABI or generic
   model/generator contracts.

#### Phase 241: Async I/O Strategy Component Boundary

**Goal:** Establish a dedicated cooperative async I/O strategy component without changing shipped
synchronous strategies.  
**Depends on:** Phase 240  
**Requirements:** AIO-01, AIO-02
**Plans:** 01 — Completed 2026-05-09; added `src/emel/io/async`, canonical aliases, fail-closed
unsupported dispatch, boundary tests, and maintained lint snapshot refresh.

**Success criteria:**

1. A canonical `src/emel/io/<async-strategy>` component exists with component-local `context`,
   `events`, `guards`, `actions`, `errors`, `sm`, and public aliases.
2. The component is separate from `io/mmap`, `io/read`, and `io/staged_read` and does not modify
   their runtime semantics.
3. Initial public dispatch fails closed or reports unsupported until validated async progress
   contracts are introduced.
4. Component docs and tests name the strategy as cooperative async I/O only, not a generic scheduler
   or device/accelerator strategy.

#### Phase 242: Suspension-Safe Request and Progress Ownership

**Goal:** Define stable request/progress ownership so resumable loading never retains borrowed
dispatch-local state across suspension.  
**Depends on:** Phase 241  
**Requirements:** AIO-03, OWN-01, OWN-02, OWN-03, OWN-04
**Plans:** 01 — Completed 2026-05-09; added caller-owned request/progress storage, validation
guards, strict scheduler proof, empty-context tests, and fail-closed behavior after validation.

**Success criteria:**

1. Async load request validation rejects invalid source, target, progress, and scheduler contracts
   before any resumable progress attempt is accepted.
2. Suspension-surviving state is owned by stable actor/scheduler/caller storage with documented
   lifetime.
3. Tests or source checks prove stack-backed spans, event payload pointers, mutable references, and
   dispatch-local request pointers are not retained across suspension.
4. Stored synchronous callbacks are not used for later async completion reporting; explicit events
   and states carry progress and terminal outcomes.
5. Async strategy context remains free of forbidden dispatch-local mirror fields.

#### Phase 243: Suspend/Resume Progress and Error Semantics

**Goal:** Implement bounded cooperative async progress with explicit partial, terminal success, and
terminal error behavior.  
**Depends on:** Phase 242  
**Requirements:** AIO-04, AIO-05, AIO-06, TST-02
**Plans:** 01 — Completed 2026-05-09; added bounded chunk progress, partial and terminal success
outcomes, cancellation/error paths, and public dispatch tests.

**Success criteria:**

1. Public dispatch can advance bounded async loading progress and observe explicit partial-progress
   outcomes.
2. Resume ordering is deterministic and monotonic over the accepted logical byte span.
3. Terminal success is published only after the requested logical span is complete.
4. Representative validation, source-contract, scheduler/resource, cancellation/rejection, and
   partial-progress failures publish deterministic terminal errors.
5. Tests cover suspend/resume ordering, partial progress, success, and representative failures
   through public dispatch and SML state inspection.

#### Phase 244: Tensor-Owned Async Integration Graph

**Goal:** Integrate cooperative async loading through tensor-owned residency semantics.  
**Depends on:** Phase 243  
**Requirements:** TNX-01, TNX-02, TNX-03, TST-03
**Plans:** 01 — Completed 2026-05-09; added tensor-owned async request/progress/done/error
events, public dispatch into `emel::io::async::sm`, residency commit on terminal success, and
direct tensor async tests.

**Success criteria:**

1. `model/tensor` initiates async loading only through public `emel/io` request events.
2. Tensor load, bind, evict, and residency lifecycle remain owned by `model/tensor`.
3. Async progress, success, and failure are visible through explicit tensor `_done` / `_error`
   events or states.
4. Public tests prove direct tensor async-load success and failure through public dispatch surfaces
   and SML state inspection.

#### Phase 245: Public Loader and Maintained Entrypoints

**Goal:** Wire maintained runtime/tool entrypoints to select or report async loading only through
public contracts.  
**Depends on:** Phase 244  
**Requirements:** TNX-04
**Plans:** 01 — Completed 2026-05-09; added public `cooperative_async` strategy reporting in
`io/loader`, model-loader error evidence, maintained tool strategy parsing, and public-contract
guard tests.

**Success criteria:**

1. `io/loader` and maintained model-loader paths can select or report cooperative async loading
   through public runtime contracts.
2. Benchmarks, paritychecker, and embedded probes avoid actor-internal async strategy headers and
   coroutine implementation types.
3. Async strategy usage is reported only when the EMEL lane executed the async runtime path.
4. Unsupported or fallback behavior is reported as unsupported/non-async, not async evidence.

#### Phase 246: Scope and Non-Regression Guardrails

**Goal:** Freeze coroutine/I/O scope and protect shipped synchronous strategies.  
**Depends on:** Phase 245  
**Requirements:** GRD-01, GRD-02, GRD-03, GRD-04
**Plans:** 01 — Completed 2026-05-09; added source guardrails for coroutine/public contract leaks,
async actor internal reach-through, behavior choice outside guards, and shipped I/O regression test
coverage.

**Success criteria:**

1. Guardrails fail if coroutine task, scheduler, or awaitable types leak into public C ABI or
   generic public model/generator contracts.
2. Guardrails fail if `tools/bench`, `tools/paritychecker`, probes, or model-loader code include
   async actor internals.
3. Guardrails fail if awaitables/actions/detail helpers choose runtime behavior that belongs in
   guards/transitions.
4. Public regression tests prove shipped mmap, read/copy, and staged-read strategy behavior still
   passes after async work lands.

#### Phase 247: Publication and Evidence Truthfulness

**Goal:** Align docs, snapshots, reporting, and quality-gate evidence with the implemented async
I/O path.  
**Depends on:** Phase 246  
**Requirements:** DOC-01, EVI-01, LNT-01, QG-01
**Plans:** 01 — Completed 2026-05-09; updated docs, loading-strategy reporting evidence,
unsupported async reporting notes, lint snapshots, and consolidated quality-gate evidence.
Milestone audit reopened `PERF-01` because unsupported `cooperative_async` reporting is not
maintained async performance evidence.

**Success criteria:**

1. Maintained docs describe the `co_sm` actor contract, async I/O scope, and deferred broader
   scheduler/device work accurately.
2. Benchmark, parity, and probe artifacts do not label a run as async/cooperative unless the async
   runtime path executed.
3. Maintained benchmark evidence compares loading-strategy performance differences across mmap,
   read/copy, staged-read, and cooperative async loading when the async runtime path exists;
   unsupported async boundary runs are reported as unsupported rather than async performance.
4. Lint snapshots or successor baselines are refreshed only through maintained workflows when
   implementation changes require it.
5. Changed-file scoped quality gates pass for coroutine wrapper, async I/O, tensor integration,
   docs, guardrails, and tests without benchmark-regression override.
6. ROADMAP, REQUIREMENTS, STATE, and milestone audit inputs are source-backed and consistent.

#### Phase 248: Maintained Cooperative Async E2E Execution Path

**Goal:** Wire the maintained model-loader/generation path through cooperative async loading so
`EMEL_MODEL_LOAD_IO_STRATEGY=cooperative_async` executes the async runtime path end to end and
publishes measured strategy evidence.  
**Requirements:** PERF-01
**Depends on:** Phase 247
**Plans:** 01 — Completed 2026-05-09; wired maintained cooperative async execution through public
`io/loader` contracts and published source-backed constrained-RAM performance evidence.

**Success criteria:**

1. Maintained model-loader/generation execution accepts `cooperative_async` without
   `io_strategy_unavailable` when the async path is available.
2. The maintained EMEL lane reaches `src/emel/io/async` through public `io/loader` and
   `model/tensor` contracts; tools and benchmarks still avoid async actor internals.
3. The generation benchmark command
   `EMEL_MODEL_LOAD_IO_STRATEGY=cooperative_async scripts/bench.sh --snapshot --compare --suite=generation`
   completes successfully and records measured `cooperative_async` timing.
4. Benchmark/docs evidence distinguishes the executed async path from unsupported/fallback paths and
   does not claim async performance unless the async runtime path executed.
5. Changed-file scoped quality gates pass for the maintained async execution path, benchmark
   evidence, docs, and guardrails without benchmark-regression override.

Plans:
- [x] 01 — Maintained cooperative async execution path

#### Phase 249: Runtime Scheduler Error Contract Repair

**Goal:** Make scheduler/resource terminal error semantics reachable through explicit
Stateforward.SML guards and transitions instead of constant true/false scheduler predicates.  
**Depends on:** Phase 248  
**Requirements:** AIO-06  
**Gap Closure:** Closes audit gaps `AIO-06` and `INT-02`.

**Success criteria:**

1. Async scheduler/resource failure is modeled by explicit guards and transitions in
   `src/emel/io/async/sm.hpp`, with predicate logic in `guards.hpp`.
2. Runtime scheduler/resource errors publish deterministic terminal `_error` outcomes through public
   async dispatch.
3. Actions and detail helpers remain free of runtime behavior selection and hidden fallback routing.
4. Public doctests cover scheduler/resource failure, cancellation/rejection, and existing success
   paths through `process_event(...)` and SML state inspection.
5. Changed-file scoped quality gates pass without benchmark-regression override.

#### Phase 250: Public Loader Resumable Async Progress

**Goal:** Preserve bounded cooperative async partial progress across the maintained
`io/loader` and model-loader boundary instead of draining all chunks inside one loader action.  
**Depends on:** Phase 249  
**Requirements:** AIO-04, TNX-03, PERF-01  
**Gap Closure:** Closes audit gap `INT-01` and renews maintained performance evidence after
loader-level resumability is preserved.

**Success criteria:**

1. The maintained `cooperative_async` loader path exposes bounded partial progress through public
   contracts instead of hiding all async chunks in a local action loop.
2. `model/tensor` remains the sole owner of tensor residency and observes async progress, success,
   and failure through explicit events/states.
3. Maintained model-loader/generation entrypoints continue to reach `src/emel/io/async` only through
   public `io/loader` and tensor contracts.
4. Benchmark and probe output distinguish partial progress, terminal success, unsupported paths, and
   fallback paths honestly.
5. Changed-file scoped quality gates and the maintained generation benchmark pass without
   benchmark-regression override.

#### Phase 251: Milestone Evidence Consistency Repair

**Goal:** Make roadmap, requirements, state, and audit evidence internally consistent after gap
closure phases are added.  
**Depends on:** Phase 250  
**Requirements:** DOC-01, EVI-01  
**Gap Closure:** Closes audit gaps `DOC-01`, `EVI-01`, and `INT-03`.

**Success criteria:**

1. `ROADMAP.md` progress, phase lists, coverage tables, and milestone status match the source-backed
   phase state.
2. `REQUIREMENTS.md` traceability maps reopened requirements to the gap-closure phases and reports
   accurate pending/satisfied counts.
3. Milestone audit inputs no longer conflict with phase summaries, verifications, validations, or
   maintained benchmark evidence.
4. Documentation describes the implemented async I/O scope without overstating unsupported device,
   scheduler, or large-model behavior.
5. Changed-file scoped docs/planning gates pass.

#### Phase 252: Large-Model Constrained-RAM Profiling And Optimization

**Goal:** Recursively profile and optimize the maintained cooperative async loading path on a model
larger than available RAM, or a maintained constrained-RAM emulation when a real model is
unavailable, without substituting tool-only fallbacks for runtime behavior.  
**Depends on:** Phase 251  
**Requirements:** PERF-02  
**Gap Closure:** Adds the requested large-model/high-performance proof for the cooperative async
strategy.

**Success criteria:**

1. A maintained profiling path exercises `cooperative_async` through public model-loader,
   `io/loader`, and tensor contracts on a large-model or constrained-RAM workload.
2. The profiling loop records recursive bottleneck evidence, applies scoped optimizations, and
   reruns the same maintained path until the remaining bottleneck is documented.
3. No whole-model in-RAM shortcut, tool-only compute fallback, or unsupported strategy result is
   reported as cooperative async performance evidence.
4. Performance output clearly reports model size, effective RAM constraint, chunk/window behavior,
   peak memory, throughput/latency, and whether the run used a real large model or maintained
   constrained-RAM emulation.
5. Changed-file scoped quality gates and the relevant benchmark/profiling command pass without
   benchmark-regression override.

#### Coverage

| Requirement | Phase |
|-------------|-------|
| CO-01 | Phase 239 |
| CO-02 | Phase 240 |
| CO-03 | Phase 240 |
| CO-04 | Phase 240 |
| CO-05 | Phase 240 |
| AIO-01 | Phase 241 |
| AIO-02 | Phase 241 |
| AIO-03 | Phase 242 |
| AIO-04 | Phase 250 |
| AIO-05 | Phase 243 |
| AIO-06 | Phase 249 |
| OWN-01 | Phase 242 |
| OWN-02 | Phase 242 |
| OWN-03 | Phase 242 |
| OWN-04 | Phase 242 |
| TNX-01 | Phase 244 |
| TNX-02 | Phase 244 |
| TNX-03 | Phase 250 |
| TNX-04 | Phase 245 |
| TST-01 | Phase 240 |
| TST-02 | Phase 243 |
| TST-03 | Phase 244 |
| GRD-01 | Phase 246 |
| GRD-02 | Phase 246 |
| GRD-03 | Phase 246 |
| GRD-04 | Phase 246 |
| DOC-01 | Phase 251 |
| EVI-01 | Phase 251 |
| PERF-01 | Phase 250 |
| PERF-02 | Phase 252 |
| LNT-01 | Phase 247 |
| QG-01 | Phase 247 |

Mapped: 32/32 v1.27 requirements; satisfied 25, pending 7.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 239. Coroutine Actor Contract | v1.27 | 1/1 | Complete | 2026-05-09 |
| 240. co_sm Wrapper, Scheduler, and Allocation Proof | v1.27 | 1/1 | Complete | 2026-05-09 |
| 241. Async I/O Strategy Component Boundary | v1.27 | 1/1 | Complete | 2026-05-09 |
| 242. Suspension-Safe Request and Progress Ownership | v1.27 | 1/1 | Complete | 2026-05-09 |
| 243. Suspend/Resume Progress and Error Semantics | v1.27 | 1/1 | Complete | 2026-05-09 |
| 244. Tensor-Owned Async Integration Graph | v1.27 | 1/1 | Complete | 2026-05-09 |
| 245. Public Loader and Maintained Entrypoints | v1.27 | 1/1 | Complete | 2026-05-09 |
| 246. Scope and Non-Regression Guardrails | v1.27 | 1/1 | Complete | 2026-05-09 |
| 247. Publication and Evidence Truthfulness | v1.27 | 1/1 | Reopened | 2026-05-09 |
| 248. Maintained Cooperative Async E2E Execution Path | v1.27 | 1/1 | Reopened | 2026-05-09 |
| 249. Runtime Scheduler Error Contract Repair | v1.27 | 0/1 | Planned | — |
| 250. Public Loader Resumable Async Progress | v1.27 | 0/1 | Planned | — |
| 251. Milestone Evidence Consistency Repair | v1.27 | 0/1 | Planned | — |
| 252. Large-Model Constrained-RAM Profiling And Optimization | v1.27 | 0/1 | Planned | — |
