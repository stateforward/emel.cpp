# Coroutine Plan

status: decode wavefront reserved-compute path measured
owner: emel
last updated: 2026-06-26

## Decision

Use `co_sm` first at the graph execution boundary.

- Infrastructure surface: `src/emel/sm.hpp`
- First inference actor: `src/emel/graph/processor/sm.hpp`
- First current consumer: `src/emel/graph/sm.hpp` through `action::request_execute`
- Experimental payoff path: standalone `src/emel/text/generator/decode_wavefront/**`
  using graph-owned reserved compute

This is not a plan to make one SML dispatch faster. The useful speedup path is to make graph
execution schedulable at a bounded phase boundary so a later decode wavefront can keep compatible
sequences ready for the same kernel route and weight stream. Single-request latency must remain
neutral before the wavefront path is allowed to ship.

Review update: production `text/generator` decode remains on direct graph dispatch. The current
decode wavefront actor is a standalone component and benchmark target until a maintained
multi-lane integration proves a benefit without regressing batch-1 latency.

`async` is not deferred by definition. An async/coroutine dispatch is compatible with the EMEL
RTC actor model when completion is driven and observed before the enclosing top-level dispatch
returns. The forbidden case is hidden work escaping the RTC boundary: an incomplete task,
continuation, scheduler work item, callback handle, mailbox entry, or post-for-later queue.

## Source-Backed Current State

- `stateforward::sml` is a namespace alias to the underlying SML implementation in the configured
  Stateforward dependency, so EMEL code must use `stateforward::sml::utility::co_sm` rather than
  naming `boost::sml` directly.
- `src/emel/sm.hpp` now includes `stateforward/sml/utility/co_sm.hpp`, exposes `emel::co_sm`,
  `emel::bool_task`, scheduler policy aliases, and a fixed no-heap coroutine allocator.
- `emel::co_sm` defaults to `emel::policy::coroutine_scheduler<emel::policy::inline_scheduler>`
  and `emel::policy::coroutine_allocator<emel::policy::fixed_coroutine_allocator<>>`.
- `src/emel/graph/processor/sm.hpp` now inherits from `emel::co_sm<model, action::context,
  inline_co_policy>` and keeps its public `process_event(const event::execute &)` wrapper
  synchronous by driving `process_event_async(...).result()` to completion before returning.
- `src/emel/graph/actions.hpp` dispatches graph execution through
  `ctx.processor_actor.process_event_async(request).result()` within the same graph RTC chain.
- `src/emel/graph/sm.hpp` now exposes `event::compute_reserved`, an internal reserved-compute
  path that reuses the graph reservation output and enters processor execution without running
  the graph assembler path again.
- `src/emel/text/generator/decode_wavefront/sm.hpp` uses bounded static-scheduler `co_sm` and
  dispatches compatible lanes through graph-owned `compute_reserved` events.
- `src/emel/text/generator/actions.hpp` no longer routes single-lane production decode through
  the wavefront actor; it dispatches the already-selected graph compute request directly.
- `tests/sm/sm_policy_tests.cpp` covers the wrapper surface: default inline scheduler, sync
  dispatch, inline `process_event_async(...).result()`, error normalization, context injection,
  scheduler access, and fixed allocator exhaustion.
- `docs/rules/sml.rules.md` and `AGENTS.md` now explicitly allow RTC async/coroutine dispatch
  while forbidding hidden deferred work.
- `docs/third_party/sml.md` now documents the inline `emel::co_sm` default.

## Invariants

- No `sml::process_queue`, `sml::defer_queue`, user mailbox, background worker, or hidden
  post-for-later mechanism.
- No dynamic allocation during dispatch. Coroutine frames must use fixed storage or reject the
  dispatch; no heap fallback is acceptable in hot paths.
- No runtime behavior selection in actions, detail helpers, coroutine bodies, or awaitables.
  Runtime decisions stay in guards and transition rows.
- No coroutine use inside kernels, logits scans, sampler loops, tokenizer inner loops, renderer
  loops, tensor-element loops, packing, quant/dequant, or matmul/attention numeric loops.
- No public API exposure of coroutine tasks or scheduler internals.
- No performance claim from `co_sm` adoption alone. Claims require benchmark evidence.
- Snapshot updates require explicit user consent.

## Completed Work

### Completed 0: Coroutine Surface

Evidence:

- `src/emel/sm.hpp` exposes the EMEL wrapper around Stateforward's utility `co_sm`.
- `emel::policy::fixed_coroutine_allocator` returns `nullptr` on pool exhaustion instead of
  falling back to heap allocation.
- Existing `emel::sm` users are untouched.
- `emel::co_sm` mirrors the existing contextless/contextful wrapper shape and normalizes
  `error_out` results like `emel::sm`.
- `process_event_async` in the EMEL wrapper observes completion before returning and always
  returns a normalized immediate `emel::bool_task`; incomplete scheduler work is not allowed to
  escape the RTC boundary.

Validation already run:

```bash
cmake --build build/zig --target emel_tests_bin -j2
ctest --test-dir build/zig -R '^emel_tests_sm$' --output-on-failure
```

Result: passed.

### Completed 1: No-Op Graph Processor Conversion

Evidence:

- `src/emel/graph/processor/sm.hpp` uses `emel::co_sm` with an inline scheduler.
- The public graph processor `event::execute` entrypoint still creates the same
  `event::execute_ctx` and `event::execute_step`, drives the inline async base dispatch to
  completion, and returns only after the RTC chain finishes.
- `src/emel/graph/actions.hpp` uses the processor async execute wrapper and observes completion
  before the graph compute action returns.
- No transition rows were moved into coroutine bodies.

Validation already run:

```bash
ctest --test-dir build/zig -R '^emel_tests_kernel_and_graph$' --output-on-failure
```

Result: passed.

## Quality Gate Status

Current status: focused unit tests and the corrected decode-wavefront benchmark pass; the repo
quality gate does not yet pass for this expanded wavefront change set. The strict LFM2 generation
comparison now correctly fails on x86_64 because the runtime uses the shared Q4 fallback instead of
an optimized Q4 kernel.

A current changed-file scoped gate was run with
`EMEL_QUALITY_GATES_BENCH_SUITE=decode_wavefront` and the temporary `clang-format` shim in
`/tmp/emel-clang-format-venv/bin`. It selected the decode wavefront benchmark runner, passed the
legacy SML scan, Zig build, dependency manifest freshness checks, parity skip, and fuzz skip, then
failed on the snapshot blockers below.

Gate blockers:

- `bench_snapshot`: failed because the new focused benchmark rows
  `decode_wavefront/batch1`, `decode_wavefront/batch4`, and `decode_wavefront/batch8` have no
  approved baseline in `snapshots/bench/benchmarks_compare.txt`.
- `lint_snapshot`: failed because new source/test files under
  `src/emel/text/generator/decode_wavefront/**` and
  `tests/text/generator/decode_wavefront/lifecycle_tests.cpp` are not yet listed in
  `snapshots/lint/clang_format.txt`.
- Snapshot updates require explicit user approval, so `snapshots/bench/benchmarks_compare.txt`,
  `snapshots/lint/clang_format.txt`, and `snapshots/quality_gates/timing.txt` were not updated.

Code-owned review fixes completed after the first failed gate:

- `tools/bench/bench_runner.cpp` now treats x86_64 and AArch64 as optimized flash hosts, matching
  existing generator tests and runtime counters.
- LFM2 generation quantized evidence validation again requires optimized Q4 and optimized Q6
  evidence on all hosts, with no shared fallback and no unrelated Q2/Q3/Q8 routes.
- `src/emel/kernel/x86_64/**` now reports shared Q4 fallback dispatches, so a missing optimized Q4
  path is visible in generation evidence instead of looking like no Q4 path ran.

Validation for those fixes:

```bash
EMEL_BENCH_ITERS=100 \
EMEL_BENCH_RUNS=3 \
EMEL_BENCH_WARMUP_ITERS=10 \
EMEL_BENCH_WARMUP_RUNS=1 \
scripts/bench.sh --compare --suite=generation
```

Result: failed as expected on this x86_64 host because optimized Q4 is still missing. The run
reported `optimized_q4_dispatch_calls=0`, `shared_q4_dispatch_calls=2378`,
`optimized_q6_dispatch_calls=291`, and `shared_q6_dispatch_calls=0`.

Current decode wavefront benchmark evidence after review fixes:

```text
decode_wavefront/batch1 emel.cpp 417.890 ns/op, reserved-scalar-baseline 340.800 ns/op, ratio=1.226x
decode_wavefront/batch4 emel.cpp 1533.400 ns/op, reserved-scalar-baseline 1355.200 ns/op, ratio=1.131x
decode_wavefront/batch8 emel.cpp 2930.290 ns/op, reserved-scalar-baseline 2713.990 ns/op, ratio=1.080x
```

Interpretation: the previous apparent speedup was from comparing reserved compute against full
graph assemble+compute. Against direct per-lane reserved compute, the current wavefront component
is slower in this fixture and is not ready for production generator integration.

```text
src/emel/text/generator/decode_wavefront/actions.hpp
src/emel/text/generator/decode_wavefront/context.hpp
src/emel/text/generator/decode_wavefront/errors.hpp
src/emel/text/generator/decode_wavefront/events.hpp
src/emel/text/generator/decode_wavefront/guards.hpp
src/emel/text/generator/decode_wavefront/sm.hpp
tests/text/generator/decode_wavefront/lifecycle_tests.cpp
```

Gate command:

```bash
PATH="/tmp/emel-clang-format-venv/bin:$PATH" \
EMEL_QUALITY_GATES_CHANGED_FILES="AGENTS.md \
docs/rules/sml.rules.md \
docs/third_party/sml.md \
src/emel/sm.hpp \
src/emel/graph/actions.hpp \
src/emel/graph/context.hpp \
src/emel/graph/events.hpp \
src/emel/graph/guards.hpp \
src/emel/graph/processor/sm.hpp \
src/emel/graph/sm.hpp \
src/emel/text/generator/actions.hpp \
src/emel/text/generator/context.hpp \
src/emel/text/generator/decode_wavefront/actions.hpp \
src/emel/text/generator/decode_wavefront/context.hpp \
src/emel/text/generator/decode_wavefront/errors.hpp \
src/emel/text/generator/decode_wavefront/events.hpp \
src/emel/text/generator/decode_wavefront/guards.hpp \
src/emel/text/generator/decode_wavefront/sm.hpp \
tests/sm/sm_policy_tests.cpp \
tests/graph/graph_tests.cpp \
tests/graph/processor/processor_action_branch_tests.cpp \
tests/text/generator/decode_wavefront/lifecycle_tests.cpp \
tools/bench/CMakeLists.txt \
tools/bench/bench_cases.hpp \
tools/bench/bench_dependency_manifest.cpp \
tools/bench/bench_disabled_cases.cpp \
tools/bench/bench_runner.cpp \
tools/bench/bench_runner_registry.cpp \
tools/bench/dependency_manifest.txt \
tools/bench/graph/processor_bench.cpp \
tools/bench/text/generator/decode_wavefront_bench.cpp \
coroutine-plan.md" \
EMEL_QUALITY_GATES_BENCH_SUITE=decode_wavefront \
  scripts/quality_gates.sh
```

Result: failed only for the snapshot-baseline blockers above.

### Completed 2: Graph Processor Neutrality Benchmark

Goal: prove the no-op `co_sm` graph processor is neutral before adding any async call surface or
decode wavefront behavior.

Evidence:

- `tools/bench/graph/processor_bench.cpp` adds a focused `graph_processor` suite.
- The EMEL lane uses the current `emel::graph::processor::sm` inline `co_sm` wrapper.
- The reference lane uses a benchmark-local `emel::sm<processor::model, action::context>`
  wrapper over the same transition table.
- Cases cover invalid request rejection, reused-buffer success, allocation-required success,
  lifecycle gate/publish/release, and done/error callback publication.
- `tools/bench/CMakeLists.txt`, `tools/bench/bench_cases.hpp`,
  `tools/bench/bench_disabled_cases.cpp`, and `tools/bench/bench_runner_registry.cpp` register
  the suite without requiring llama.cpp.
- `tools/bench/bench_runner.cpp` prints graph-processor compare rows as `reference-baseline`
  instead of `llama.cpp`.

Validation run:

```bash
EMEL_BENCH_ITERS=200000 \
EMEL_BENCH_RUNS=9 \
EMEL_BENCH_WARMUP_ITERS=10000 \
EMEL_BENCH_WARMUP_RUNS=1 \
scripts/bench.sh --compare --suite=graph_processor
```

Result:

| case | inline `co_sm` | `emel::sm` baseline | ratio |
| --- | ---: | ---: | ---: |
| `graph/processor_alloc` | 350.723 ns/op | 357.034 ns/op | 0.982x |
| `graph/processor_invalid` | 24.942 ns/op | 24.958 ns/op | 0.999x |
| `graph/processor_reused` | 305.952 ns/op | 315.271 ns/op | 0.970x |

Interpretation:

- The no-op graph processor `co_sm` conversion passes the neutrality checkpoint.
- The benchmark has low-single-digit run-to-run movement, including earlier samples where
  successful dispatch was slightly slower and later samples where it was slightly faster.
- Treat the result as neutral infrastructure, not an inference speedup claim.
- Do not expand this no-op conversion into generator/decode paths as a performance feature.
- Use the benchmark as the regression guard for future coroutine candidates.

Generation comparison still required before any inference-throughput claim:

```bash
EMEL_BENCH_ITERS=1 \
EMEL_BENCH_RUNS=1 \
EMEL_BENCH_WARMUP_ITERS=0 \
EMEL_BENCH_WARMUP_RUNS=0 \
EMEL_GENERATION_WORKLOAD_ID=qwen3_single_user_hello_max_tokens_1_v1 \
scripts/bench.sh --compare --suite=generation
```

## Phase 3: RTC `process_event_async` Surface

Goal: make graph execution callable through `process_event_async` without changing generator or
graph RTC semantics.

Status: completed and measured. This is not an inference speedup by itself.

Evidence:

- `processor::sm::process_event(const event::execute &)` delegates to
  `process_event_async(ev).result()`.
- `processor::sm::process_event_async(const event::execute &)` preserves the dispatch-local
  `execute_ctx` / `execute_step` handoff and returns an immediate `emel::bool_task`.
- `graph::action::request_execute` calls
  `ctx.processor_actor.process_event_async(request).result()`.
- `graph_processor_process_event_async_execute_completes_in_rtc` proves the public async execute
  wrapper publishes callbacks and output before `.result()` is observed.
- The existing `graph_machine_compute_lifecycle_dispatch_is_alloc_free` test now covers graph
  compute through the async processor execution path.

Acceptance:

- `graph::sm` and `text::generator::sm` remain synchronous RTC actors.
- No generated token can observe partially completed graph work.
- No hidden scheduler work survives return from the top-level dispatch.
- Allocator counters show zero graph compute hot-path heap allocation.

Validation run:

```bash
cmake --build build/zig --target emel_tests_bin -j2
ctest --test-dir build/zig -R '^emel_tests_sm$' --output-on-failure
ctest --test-dir build/zig -R '^emel_tests_kernel_and_graph$' --output-on-failure
EMEL_BENCH_ITERS=200000 \
EMEL_BENCH_RUNS=9 \
EMEL_BENCH_WARMUP_ITERS=10000 \
EMEL_BENCH_WARMUP_RUNS=1 \
scripts/bench.sh --compare --suite=graph_processor
```

Result from the latest repeated sample:

| case | async `co_sm` graph processor | `emel::sm` baseline | ratio |
| --- | ---: | ---: | ---: |
| `graph/processor_alloc` | 349.161 ns/op | 344.381 ns/op | 1.014x |
| `graph/processor_invalid` | 32.150 ns/op | 26.940 ns/op | 1.193x |
| `graph/processor_reused` | 316.821 ns/op | 310.523 ns/op | 1.020x |

Interpretation:

- The coroutine-driven graph processor is implemented and remains RTC.
- Successful dispatch has a low-single-digit overhead against the direct `emel::sm` baseline.
- Invalid rejection is materially slower because the async wrapper overhead dominates a tiny
  failure path.
- This phase proves semantics and establishes a benchmark guard; it does not speed inference.

## Completed 4: Decode Wavefront Driver

Goal: turn `co_sm` into an inference speedup by batching compatible decode work across sequences.

Status: completed for the bounded graph-compute wavefront target.

Evidence:

- `src/emel/text/generator/decode_wavefront/**` defines a bounded generator-owned wavefront
  actor with explicit lane stages for up to 8 lanes.
- Guards require compatible model identity, backend identity, kernel kind, attention mode, kernel
  route, output contract, dtype/layout contract, quantized contract, step size, and token count.
- The wavefront path does not share mutable lane context. Each lane carries its own graph actor,
  graph compute request, compatibility key, and acceptance flag.
- The graph wrapper owns the optimization boundary through `event::compute_reserved`; wavefront
  does not reach into graph actions or assembler internals.
- Reserved compute requires a successful graph reservation first, seeds the compute output from the
  reservation, and skips normal graph assemble hints.
- `process_event_async` on the wavefront observes the bounded static-scheduler dispatch before
  returning and exposes a normalized immediate `emel::bool_task`, preserving RTC semantics.
- No background worker, mailbox, `defer_queue`, `process_queue`, or hidden post-return work was
  introduced.

Acceptance:

- Deterministic dispatch and first-lane failure behavior are covered by decode wavefront tests.
- Scheduler depth is bounded by `fifo_scheduler<16u, 64u>` and lane count is bounded by
  `event::k_max_lanes == 8`.
- Batch-1/4/8 performance is not accepted for production integration yet. The corrected
  reserved-scalar baseline shows current wavefront overhead rather than a grouping win.

Validation run:

```bash
cmake --build build/zig --target emel_tests_bin -j2
./build/zig/emel_tests_bin --no-breaks --test-case="*graph*"
./build/zig/emel_tests_bin --no-breaks --test-case="*x86_64*q4*"
./build/zig/emel_tests_bin --no-breaks --test-case="co_sm*"
./build/zig/emel_tests_bin --no-breaks --test-case="decode wavefront*"
ctest --test-dir build/zig -R '^emel_tests_kernel_and_graph$' --output-on-failure
ctest --test-dir build/zig -R '^emel_tests_generator_and_runtime$' --output-on-failure
ctest --test-dir build/zig -R '^emel_tests_sm$' --output-on-failure
EMEL_BENCH_ITERS=2000 \
EMEL_BENCH_RUNS=5 \
EMEL_BENCH_WARMUP_ITERS=100 \
EMEL_BENCH_WARMUP_RUNS=1 \
scripts/bench.sh --suite=decode_wavefront --compare
```

Result:

| case | wavefront path | reserved-scalar baseline | ratio |
| --- | ---: | ---: | ---: |
| `decode_wavefront/batch1` | 417.890 ns/op | 340.800 ns/op | 1.226x |
| `decode_wavefront/batch4` | 1533.400 ns/op | 1355.200 ns/op | 1.131x |
| `decode_wavefront/batch8` | 2930.290 ns/op | 2713.990 ns/op | 1.080x |

Interpretation:

- The first useful `co_sm` target remains the graph execution boundary, but the current
  decode-wavefront component is not a production win.
- The earlier speedup claim was an apples-to-oranges comparison against full graph compute. The
  corrected benchmark measures wavefront orchestration against direct reserved graph dispatch.
- The current implementation still drains inside the top-level RTC call; that is intentional for
  actor correctness. Future external completion backends must suspend only at explicit phase
  boundaries without changing public graph/generator semantics.

## Completed 5: Thread-Pool Scheduler Win Over Single-SM and llama.cpp

Goal: get `co_sm` + `thread_pool_scheduler` to beat the single-`sm` baseline and
llama.cpp on a realistic parallel decode workload (independent per-lane GEMVs).

Status: achieved at small per-lane dims (the inter-op-parallelism regime), with a
prerequisite scheduler deadlock fixed along the way.

### Scheduler correctness fix (prerequisite)

`emel::policy::thread_pool_scheduler` deadlocked under rapid repeated fork/join,
most readily when lane count == worker count (the decode wavefront's 8-lane,
8-worker config). A single dispatch rarely hit it, so existing tests passed while
the bug was latent. Two independent races in the join latch:

1. `join_group`'s close/complete handshake was a Dekker pattern with
   release/acquire ordering. StoreLoad reordering let `wait()` miss the final
   completion while the last completer missed the close, stranding the wakeup.
2. The per-group `std::binary_semaphore` was destroy-during-release: groups are
   stack-reused each round, so the waiter could return and destroy the group
   while the last completer was still inside libc++ `release()`/notify (UAF).
   `run_or_schedule_and_wait`'s local `done` semaphore had the same flaw.

Fix: replaced the semaphore/closed handshake with a lifetime-safe spin-join on
`pending_` (a completer's last touch is its decrement, so `wait()` returns only
after every completer is done); `run_or_schedule_and_wait` spins on a local
`atomic<bool>` the worker sets last. Added `emel::policy::cpu_relax()`. New
regression test `thread_pool_scheduler_ref_fork_join_survives_rapid_repeated_rounds`
(20000 rounds x 8 lanes). Validated: 80M+ batch-8 fork/joins with 0 stalls.

### Warm worker loop

Workers now spin-claim a wake permit (bounded `k_idle_spin_budget`) before
falling back to a blocking acquire, keeping the pool warm across a burst of
fork/joins and removing ~3-4us of resleep/wakeup latency per round (the same
warm-polling strategy ggml's threadpool uses). The permit-per-task invariant is
preserved, so there is no drift or lost-wakeup regression.

### Evidence (Ryzen 9 5950X, 16C/32T; `-O3 -march=native`)

In-repo, via the actual production decode wavefront actor (thread-pool co_sm)
through `scripts/bench.sh --compare --suite=decode_wavefront`. New realistic
cases run a decode-representative f32 GEMV (`y = W@x`, dim x dim, independent
weights per lane); the `gemv_*` cases compare against the single graph sm
(reserved scalar), the `ggml_*` cases against a ggml reference (independent
mul_mat lanes, warm threadpool, same core budget). Default `dim=256`:

| case (batch 8) | emel.cpp | baseline | result |
| --- | ---: | ---: | --- |
| `gemv_batch8` (vs single-sm) | ~8.4-10.5 us | ~33 us | 3.1-3.9x faster |
| `ggml_batch8` (vs llama.cpp) | ~9.1-9.3 us | ~12.7 us | 1.37-1.41x faster (3 runs) |

The single-sm win holds at every dim (3-6x). The llama.cpp win holds at small
dims (dim<=256, ~1.4x) where inter-op parallelism dominates ggml's intra-op
threading; the crossover is ~dim 384-512, where ggml's hand-optimized blocked
kernel catches up. That crossover is a kernel-quality matter, not a scheduling
one: the scheduler delivers near-linear parallelism (up to 7.85x on 8 lanes for
bare GEMV lanes), and its overhead floor is ~5-10us (idle-pool worker wakeup),
dwarfed by real decode work.

Contrast: the original trivial-work cases (`batch8`, kernel = `*calls += 1`)
still show the wavefront 2-4x slower, confirming the earlier "wavefront is
slower" result was a fixture artifact, not a scheduler limitation.

Caveat: EMEL's inter-op parallelism beats ggml only for independent-weight lanes
(concurrent requests / multi-model / MoE). For shared-weight batched decode, ggml
batches sequences into one GEMM and reuses the weight stream (bandwidth-optimal);
the comparison here is framed as independent-weight concurrent lanes accordingly.

## Completed 6: Parallel Matmul Cutover (view-sliced lanes, ith/nth removed)

Goal: turn the proven thread-pool fork/join into a maintained single-request hot-path win for
prefill GEMM and per-token decode GEMV, and remove ggml's `ith`/`nth` thread-partition fields so
view slicing is the only parallelism model in the kernel event contract.

Decision record:

- `ith`/`nth` were declared on every kernel op event but never used for partitioning; the only
  read site was a validator that rejected anything but `ith==0 && nth==1`. The architecture-native
  slice descriptor is the `tensor_view` itself: a row slice is a smaller view, the kernel computes
  whatever its views describe, and partition policy lives only at the orchestration fork site.
  The fields and their validator clause are deleted; events remain complete work descriptions
  with no thread identity.

Implementation:

- `src/emel/text/generator/detail.hpp` owns the lanes: `k_matmul_lanes == 8` per-lane
  `emel::kernel::sm` actors plus an `std::optional` `thread_pool_scheduler<8, 16, 128>` engaged
  once in `prepare()`. A parallel dispatch forks one logical mul_mat into pack-group-aligned
  contiguous row-slice events (`compute_matmul_row_slices`, groups of 8 for `*_x8`, 4 for
  `*_x4`, 1 otherwise), the caller computes slice 0 while pool workers compute the rest, and the
  join completes before the action returns. Rejected submits run the same slice inline, so a
  pool-worker caller (wavefront lane) degrades to serial automatically — lanes-first composition.
- Lane mode is a compile-time template parameter threaded through the matmul helpers and
  runners; route guards choose it: prefill parallel rows require the pool engaged and
  `prompt_token_count >= k_parallel_min_prefill_tokens` (8), decode parallel rows require
  `n_embd >= k_parallel_min_gemv_dim` (1024) so tiny models keep the scalar route. Parallel
  contract rows sit above their serial siblings for flash materialized + preselected prefill
  chunk routes and flash materialized + preselected decode scalar routes.
- Evidence counters live per kernel actor, so audit reads sum the primary kernel and every lane
  (`compute_kernel_counter_total`); `kernel_dispatch_calls` and route counters stay once per
  logical matmul at the fork site.

Evidence (Ryzen 9 5950X, zig c++ -O3 -mavx2 -mfma):

```text
parallel_matmul/gemv_f32   emel.cpp 285062 ns/op, reference-baseline 2419432 ns/op, ratio=0.118x
parallel_matmul/gemv_q8_0  emel.cpp  54841 ns/op, reference-baseline  371154 ns/op, ratio=0.148x
parallel_matmul/gemv_q4_k  emel.cpp  68648 ns/op, reference-baseline  470182 ns/op, ratio=0.146x
parallel_matmul/gemv_q6_k  emel.cpp  43786 ns/op, reference-baseline  282983 ns/op, ratio=0.155x
parallel_matmul/gemm8_f32  emel.cpp 156122 ns/op, reference-baseline 1265074 ns/op, ratio=0.123x
```

Interpretation: 6.5x-8.5x over serial single-kernel dispatch at dim 2048 across dtypes — the
inter-op fork/join regime delivering near-linear scaling on 8 lanes, now on the maintained
matmul dispatch path rather than a standalone component.

Correctness:

- Row slices write disjoint dst rows and reorder no reductions, so parallel output is
  bit-identical to serial; `tests/kernel/matmul_tests.cpp` proves slice
  arithmetic, group alignment, and f32/q8_0 serial-vs-parallel byte equality, and the full
  generator fixture suites pass with prompts >= 8 tokens taking the parallel prefill route.
- The strict LFM2 x86_64 generation evidence failure is unchanged and pre-existing
  (`optimized_q4_dispatch_calls=0 shared_q4_dispatch_calls=2378 optimized_q6_dispatch_calls=291`
  — identical counts to the pre-cutover run, which also proves lane-counter aggregation
  preserves evidence exactly).

## Completed 7: Matched-Thread llama.cpp Comparison Lanes

Goal: measure the view-sliced parallel matmul against llama.cpp/ggml at the same core budget
instead of only against EMEL's own serial dispatch, per the reference-comparison rules.

Kernel-class lane (`parallel_matmul/ggml_*` cases): the EMEL side runs the identical 8-lane
fork/join as the plain-named cases; the reference side runs the same logical matmul as one ggml
`mul_mat` node on a warm 8-thread ggml threadpool (the decode_wavefront reference pattern).
Operand class is plain GGUF-native blocks on BOTH sides — this exercises EMEL's shared
(non-repacked) kernels, not the repacked x4/x8 kernels the production decode routes dispatch
after `prepare()`.

Evidence (Ryzen 9 5950X, dim 2048, 8 threads both sides, iter=2000 runs=5):

```text
parallel_matmul/ggml_gemm8_f32 emel.cpp 170008 ns/op, llama.cpp 201689 ns/op, ratio=0.843x
parallel_matmul/ggml_gemv_f32  emel.cpp 284070 ns/op, llama.cpp  29684 ns/op, ratio=9.570x
parallel_matmul/ggml_gemv_q8_0 emel.cpp  55579 ns/op, llama.cpp  20757 ns/op, ratio=2.678x
parallel_matmul/ggml_gemv_q4_k emel.cpp  68174 ns/op, llama.cpp  16335 ns/op, ratio=4.173x
parallel_matmul/ggml_gemv_q6_k emel.cpp  44451 ns/op, llama.cpp  17805 ns/op, ratio=2.496x
```

Interpretation:

- Prefill-shape GEMM: EMEL's inter-actor row slicing beats ggml's intra-op chunking (0.843x).
- Decode-shape GEMV: the gap is per-kernel arithmetic on plain blocks, not orchestration — the
  EMEL parallel numbers match serial/8 scaling exactly, and the serial kernels trail ggml's
  vec_dot by the same factors (plain-q4_k ~4x, plain-q8_0 ~2.7x, f32 GEMV ~10x, consistent with
  the known missing optimized plain-Q4 kernel and a near-scalar f32 GEMV path).
- These rows are a kernel-class comparison and must not be quoted as production decode numbers:
  production decode dispatches the repacked x4/x8 kernels. The production-class cross-engine
  number is the end-to-end generation compare at matched reference threads (in progress; blocked
  for LFM2-architecture fixtures on x86 by the same strict Q4 evidence audit until either the
  optimized plain-Q4 kernel lands or the audit gains a quant-class-aware LFM2+Q8_0 branch).

Follow-up work surfaced by this comparison: optimized plain-Q4 GEMV kernel (already the known
blocker), SIMD f32 GEMV, and a repacked-operand cross-engine lane once llama.cpp's x86 repack
path is wired into the reference fixture.

End-to-end lane (LFM2.5-230M-Q8_0, prompt "hello", `EMEL_BENCH_REFERENCE_THREADS=8` so the
reference runs the same 8-core budget as EMEL's lane pool; 1-thread reference rows kept for
context):

```text
max_tokens_1   emel.cpp 443.0 ms/op, llama.cpp  27.1 ms/op (8t), ratio=16.337x   [1t: 130.7 ms, 3.407x]
max_tokens_100 emel.cpp 1002.6 ms/op, llama.cpp 514.0 ms/op (8t), ratio=1.950x
```

Decomposition (t100 minus t1, over 99 decode tokens): EMEL ~5.65 ms/token vs llama.cpp
~4.92 ms/token — steady-state decode is within ~1.15x at matched threads, consistent with both
sides being memory-bandwidth-bound on q8_0 GEMV at this model size. The end-to-end gap is
concentrated in the first-token path: EMEL spends ~443 ms before the first sampled token where
llama.cpp spends ~27 ms at 8 threads. The "hello" prompt is only a handful of tokens, so this is
not GEMM volume; profiling the EMEL first-token/session path (prompt format, tokenize, prefill
graph build/plan, first dispatch) is the top follow-up from this lane.

## Phase 5: External Completion Backends

Goal: use coroutine suspension only where an actual external completion source exists.

Valid future boundaries:

- accelerator submission/completion
- OS-backed cold-load or staged-read completion
- platform DMA/NPU completion

Invalid boundaries:

- CPU AArch64 or x86_64 kernel inner loops
- software prefetch
- per-token sampler loops
- synthetic sleep/poll loops

Acceptance:

- Async backend routes are explicit guarded transitions.
- Synchronous CPU routes remain direct and are not slower.
- The EMEL lane remains EMEL-owned; no llama.cpp/ggml object is shared into runtime execution.

Status: completed for the OS-backed staged-read boundary (streaming weights
milestone, 2026-07-02). The external-completion scheduler policy landed
upstream-first (stateforward/sml.cpp PR #16: `completion_source`, park/drain
in `co_sm`, ascending commit sweep, dispatch-thread-only resumption; emel pin
follows the PR branch until merge). The first consumer is
`emel::model::tensor::window` — a tensor-owned streaming weight window whose
acquire dispatches suspend on in-flight slot loads fired by a 2-lane I/O pool
running `io/staged_read` copies from a whole-file mmap source, with willneed/
dontneed advise ahead of and behind the ring. The generator streams decode
through it behind one engagement guard (`guard_decode_stream_window_ready`)
with streamed rows above their resident/parallel siblings; streamed-vs-
resident token parity is proven through the public generator path, and the
opt-in `weight_streaming` bench compares emel window / emel mmap / llama.cpp
mmap lanes under an optional systemd-run memory cap
(`scripts/bench.sh --memory-max=...`). Follow-ons recorded in the milestone
memory: wavefront x streaming multiplexing, aarch64 repack-on-acquire,
session context caps for honest capped benches, MoE expert-granular
streaming, co_sm-throughout sweep.

## Validation Checklist

Before claiming the graph processor coroutine phase is complete:

- `emel_tests_sm` passes.
- `emel_tests_kernel_and_graph` passes.
- Changed-file scoped quality gate passes.
- A graph processor benchmark exists and compares `emel::sm` vs async inline `emel::co_sm`.
- No snapshots were updated without explicit approval.
- Code review confirms no hidden runtime behavior selection was moved into coroutine bodies,
  awaitables, actions, or detail helpers.

Before making any broader maintained-generation throughput claim:

- Generation benchmark evidence exists for single-token and multi-token maintained decode
  workloads, not only the focused decode wavefront microbenchmark.
- Snapshot/parity checks cover maintained generation fixtures.
- A wavefront or external-completion implementation exists at the maintained generation entrypoint,
  not only as a callable component.

## Risk Register

- `co_sm` adds overhead but no speedup.
  Detection: graph processor benchmark.
  Response: keep graph processor on `emel::sm` and use `co_sm` only where it preserves RTC or
  where a later completion is modeled as an explicit external event.
- Scheduler becomes a hidden queue.
  Detection: SML rule review, callback-order tests, and escaping-task tests.
  Response: restrict inference paths to inline/immediate-drain scheduling until a bounded driver
  is explicitly approved.
- Coroutine frame allocates on hot path.
  Detection: allocator accounting and allocation instrumentation.
  Response: increase fixed pool or reject dispatch; never fall back to heap in hot paths.
- Awaitable chooses behavior.
  Detection: action/guard branch tests and code review.
  Response: move choice into guards and transition rows.
- Decode batching changes outputs.
  Detection: generation parity tests and compare summaries.
  Response: fix lane ordering and publication; do not accept drift as a performance tradeoff.
- Kernel route mismatch weakens parity claims.
  Detection: diagnostics and runtime contract counters.
  Response: benchmark only equivalent operand paths.

## Decision Checkpoints

- After graph processor measurement: continue only if the no-op `co_sm` conversion is neutral.
- After RTC `process_event_async` wiring: continue only if RTC semantics and allocation
  guarantees are proven.
- After decode wavefront implementation: focused correctness passed, corrected performance did not.
  Next checkpoint is either a real multi-lane scheduler integration that amortizes the overhead or
  a different coroutine boundary with external completion work.
