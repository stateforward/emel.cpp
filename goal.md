# Current Goal

Keep `process_event_async` aligned with the upstream `sml.cpp` awaitable
completion-handle contract after `stateforward/sml.cpp#17`, and keep hot
generation matmul multithreading modeled as an explicit dependency-injected
`parallel` actor route. Iterate until the relevant deterministic CPU inference
path beats `llama.cpp` on this CPU.

## Hard Corrections

- Do not merge draft PRs. Only the non-draft open `emel.cpp` PRs are in scope
  for the local integration branch.
- Remove wrapper-level `.result()` flattening from `src/emel/sm.hpp`
  `process_event_async`; async dispatch should remain awaitable instead of
  being synchronously joined in the local wrapper.
- Do not normalize `error_out` or other domain error payloads in the local SML
  wrapper. `process_event` / `process_event_async` report SML acceptance; if an
  event needs error data, that data belongs in the event or output payload.
- Preserve RTC actor semantics: async work may be awaited later only when the
  RTC boundary belongs to the dispatched machine and no hidden deferred work
  escapes that boundary.
- No hidden behavior selection: runtime behavior must be modeled explicitly with
  guards, states, and transitions, not buried in actions, wrapper member
  functions, detail helpers, scheduler fallbacks, or callback/error channels.
- Generation matmul wording and APIs should say `parallel`, not `async`.
  Coroutines preserve awaitability; they are not the multithreading contract for
  the hot matmul route.
- After implementation changes, run a recursive dual-subagent review for rule
  violations and fix findings before declaring completion.

## Current Integration State

- Working branch: `codex/sml17-async-threadpool-cutover`.
- Merged local non-draft PR heads: `emel.cpp#95`, `#96`, `#97`, `#98`, `#99`.
- Explicitly not merged: draft PRs `emel.cpp#48` and `#49`.
- Upstream scheduler fix: `stateforward/sml.cpp#17` merged as
  `49207123cd3f39767764bae774932cb48623f92f`.
- Local SML pin should point at that merge commit.
- Local wrapper state: `src/emel/sm.hpp` no longer flattens
  `process_event_async(...)` through `.result()`, and wrapper-level event
  error normalization has been removed.
- Scheduler finding: upstream `thread_pool_scheduler` is correct for the SML
  coroutine scheduler cutover but still has too much shared-queue wake/claim
  overhead for the hot row-sliced matmul actor fanout. The matmul route now
  uses an EMEL-owned fixed fork/join lane pool with one direct slot per worker
  behind an explicit `emel::text::generator::matmul::sm` actor. The caller
  still joins every submitted lane actor before dispatch returns, so no
  deferred work escapes the RTC boundary.
- Benchmark finding: the aarch64 f32 GEMV path has a 4-row RHS-reuse kernel,
  q4_k has a 4-row q4_k x q8_k NEON helper, q8_0 avoids stack lane/scale arrays
  in its 4-row helper, and raw q4_k/q6_k/q8_0 GEMV now has explicit prepared-RHS
  guard/action/transition routes. The parallel benchmark now uses the production
  EMEL prepared weight layout for q4_k/q8_0 (one-time setup, outside the timed
  dispatch), quantizes the RHS vector inside the measured EMEL function for
  quantized GEMV cases, then dispatches the explicit prepared-input route. It
  uses the same fixed lane pool as production.
- Dual-review blocker fixes now applied: `fork_join_lane_pool::wait()` cannot
  observe completion before the worker slot is reusable, production and
  benchmark fork/join callers no longer ignore failed lane submission or join
  failure, and the SML rules text records the corrected upstream
  `process_event_async` awaitable completion-handle semantics.
- Dual-review follow-up fixes now applied: the production-layout
  `parallel_matmul` benchmark slices packed q4_k/q8_0/q6 weight storage by
  packed row groups instead of raw logical rows, and production parallel
  matmul treats lane-pool absence, failed lane submission, or join rejection as
  scheduler contract violations (`std::terminate`) instead of returning a
  normal data-plane `false` from `detail.hpp`.
- Latest measured compare
  (`EMEL_BENCH_ITERS=50`, `EMEL_BENCH_WARMUP_ITERS=1000`, after the q4/q8
  kernel changes and 7-worker pool): EMEL wins `ggml_gemm8_f32` at
  `181051.680 ns/op` vs `269058.320`, but still loses `ggml_gemv_f32` at
  `121395.000` vs `37575.000`, `ggml_gemv_q4_k` at `19468.340` vs
  `15583.320`, `ggml_gemv_q6_k` at `27315.000` vs `24594.160`, and
  `ggml_gemv_q8_0` at `22376.660` vs `15911.660`. The current benchmark still
  does not meet the goal.
- Stable current-tree compare with the upstream scheduler spin budget
  (`EMEL_BENCH_ITERS=100`, `EMEL_BENCH_RUNS=3`,
  `EMEL_BENCH_WARMUP_ITERS=1000`): EMEL wins `ggml_gemm8_f32` at
  `177905.410 ns/op` vs `268556.250`, but still loses `ggml_gemv_f32` at
  `103457.500` vs `37129.170`, `ggml_gemv_q4_k` at `23112.910` vs
  `14135.000`, `ggml_gemv_q6_k` at `29935.000` vs `24400.830`, and
  `ggml_gemv_q8_0` at `22199.580` vs `15842.920`.
- Scheduler probe evidence, reverted after measurement: raising the fetched
  upstream `thread_pool_scheduler` idle spin budget from `2048` to `1048576`
  improved the current gated route to `ggml_gemv_f32=65744.170`,
  `ggml_gemv_q4_k=20943.750`, `ggml_gemv_q6_k=26532.500`, and
  `ggml_gemv_q8_0=20569.160`; combining that high-spin probe with no start
  gate got `ggml_gemv_q6_k=24473.750` vs llama `24602.080` but still lost
  f32, q4_k, and q8_0. This supports an upstream scheduler fix around
  warm-poll/fair batch wake latency, but it is not enough alone.
- Gate-shaped fair fixed-lane-pool compare after switching the benchmark to the
  production EMEL packed q4_k/q8_0 weight layout
  (`EMEL_BENCH_SUITE=parallel_matmul`, `EMEL_BENCH_ITERS=2000`,
  `EMEL_BENCH_RUNS=5`, `EMEL_BENCH_WARMUP_ITERS=200`,
  `EMEL_BENCH_WARMUP_RUNS=1`, quantized RHS prep included in EMEL timing):
  EMEL wins every `ggml_*` row:
  `ggml_gemm8_f32=162777.104 ns/op` vs llama `334417.021`,
  `ggml_gemv_f32=30908.812` vs `40519.584`,
  `ggml_gemv_q4_k=11415.250` vs `15300.375`,
  `ggml_gemv_q6_k=22154.458` vs `24581.938`, and
  `ggml_gemv_q8_0=9725.750` vs `16132.312`.
- `snapshots/bench/benchmarks.txt` was updated for the scoped
  `parallel_matmul` suite with the verified production-layout EMEL rows.
- Validation so far: `EMEL_BUILD_JOBS=4 scripts/build_with_zig.sh`,
  `./build/zig/emel_tests_bin --test-case='fork_join_lane_pool_wait_returns_after_worker_slot_reusable' --no-skip`,
  `./build/zig/emel_tests_bin --test-case='co_sm_thread_pool_scheduler*' --no-skip`,
  `./build/zig/emel_tests_bin --test-case='kernel_aarch64_raw_quantized_prepared_rhs_routes_are_explicit_and_numeric_match' --no-skip`,
  `./build/zig/emel_tests_bin --test-case='kernel_aarch64_q4_k_4rows_neon_matches_scalar' --no-skip`,
  `./build/zig/emel_tests_bin --test-case='parallel matmul*' --no-skip`,
  `EMEL_BUILD_JOBS=4 cmake --build build/bench_tools_ninja_parallel_matmul --parallel 4 --target bench_runner`,
  `EMEL_BUILD_JOBS=4 cmake --build build/bench_tools_ninja --parallel 4 --target bench_runner`,
  `scripts/bench.sh --snapshot --compare --update --suite=parallel_matmul`,
  and `scripts/bench.sh --snapshot --compare --suite=parallel_matmul`
  passed after the implementation changes. After the follow-up blocker fixes,
  `./build/zig/emel_tests_bin --test-case='parallel matmul*' --no-skip`,
  `./build/zig/emel_tests_bin --test-case='fork_join_lane_pool_wait_returns_after_worker_slot_reusable' --no-skip`,
  `./build/zig/emel_tests_bin --test-case='co_sm_thread_pool_scheduler*' --no-skip`,
  `./build/zig/emel_tests_bin --test-case='kernel_aarch64_raw_quantized_prepared_rhs_routes_are_explicit_and_numeric_match' --no-skip`,
  `./build/zig/emel_tests_bin --test-case='kernel_aarch64_q4_k_4rows_neon_matches_scalar' --no-skip`,
  `scripts/lint_snapshot.sh`, `./build/zig/emel_tests_bin --test-case='graph_processor*' --no-skip`,
  and `scripts/bench.sh --snapshot --compare --suite=graph_processor` also
  passed.
- Recursive dual-subagent re-review status: first pass found the packed
  benchmark slicing bug and hidden scheduler-failure `false` return. Fixes are
  applied. Reviewers `019f370f-b36e-7072-8be5-ab2d0ea06928` and
  `019f370f-cc52-7e10-98b5-3bb92a482932` both returned `SIGNOFF`, with no
  remaining actionable findings.
- Final changed-file scoped quality gate first failed only in the
  `bench_snapshot` lane because `tools/bench/graph/processor_bench.cpp` still
  expected invalid graph-processor requests to make `process_event(...)`
  return `false`. That benchmark has been updated to the corrected wrapper
  contract: `process_event(...)` reports SML acceptance, and error detail flows
  through the event callback.
- A follow-up unscoped benchmark snapshot rerun originally exposed missing Qwen
  generation fixtures. Those fixtures are now present locally for benchmarking:
  `tests/models/Qwen3-0.6B-Q8_0.gguf` and
  `tests/models/Qwen3-4B-Q4_K_M.gguf`.
- The scoped quality-gate rerun then failed only because the new internal
  `sm_scheduler/idle_async` and `sm_scheduler/busy_worker_async` benchmark rows
  had no baseline. `scripts/bench.sh --snapshot --compare --update
  --suite=sm_scheduler` passed and merged those scheduler baselines.
- Final scheduler/matmul verification passed with:
  `EMEL_BUILD_JOBS=4 EMEL_QUALITY_GATES_CHANGED_FILES="$changed_files"
  EMEL_QUALITY_GATES_BENCH_SUITE=parallel_matmul,graph_processor,kernel_aarch64,sm_scheduler,decode_wavefront
  scripts/quality_gates.sh`. The scoped benchmark snapshot, changed-line
  coverage, paritychecker, fuzz smoke, determinism check, lint snapshot, and
  docs generation all completed successfully.
- `codex/generation-threaded-benchmark` has been pulled into this worktree as a
  separable staged branch delta. It adds generation compare thread metadata and
  wrapper support for `--reference-threads`, records EMEL as the fixed
  8-lane parallel matmul implementation, and lets the llama.cpp reference run
  with a configured thread count.
- Follow-up correction applied: generation benchmarks expose exactly two
  benchmark lanes, `single` and `multithreaded`. Both lanes now run by default
  from `bench_runner`, `scripts/bench.sh`, and the generation compare wrappers.
  `EMEL_BENCH_GENERATION_LANE=single|multithreaded`,
  `EMEL_BENCH_GENERATION_LANES=single|multithreaded`, or wrapper
  `--benchmark-lane single|multithreaded` are the explicit opt-out paths.
  `single` disables EMEL's parallel matmul route through a public generator
  benchmark-lane event and forces the llama.cpp reference to one thread;
  `multithreaded` keeps EMEL's 8-lane parallel matmul contract and defaults the
  llama.cpp reference to eight threads unless overridden. The public generator
  event keeps the benchmark toggle in the actor transition graph instead of
  reaching into internals.
- Local DI correction applied: `emel::text::generator::sm` now has a
  single public constructor that accepts an explicit `dependencies` aggregate.
  The generator no longer default-constructs or owns a hidden lane pool. Owners
  must provide the model, conditioner, `matmul::execution_policy`, and generator
  `runtime_policy`. `make_auto_dependencies(...)` still builds the matmul auto
  policy from the injected lane pool, but route thresholds now come from the
  caller-supplied runtime policy instead of a generator-owned default.
- Local matmul correction applied: `matmul::sm` no longer has a
  default constructor or `lane_pool&` constructor that auto-detects host kernel
  kind. The actor is constructed from `matmul::execution_policy`, which carries
  the injected lane pool, kernel kind, and active lane count. Lane topology is
  now caller-owned through the `lane_pool<worker_lanes, ...>` alias and
  type-erased `lane_pool_ref`; the production actor allocates per-lane child
  actor scratch once at construction.
- Local route-policy correction applied: generator prefill/decode
  route thresholds now live in injected `runtime_policy.routes`; the numeric
  chunk sizes remain compile-time kernel shape constants. Generic generation no
  longer asks whether the model family is Qwen/Gemma/LFM; it derives q/k norm,
  value norm, shortconv, and sliding-attention behavior from the loaded model
  contract and bound block facts. Tool binaries explicitly use
  `tools/generation_route_policy.hpp`; generator tests explicitly use
  `tests/text/generator/generator_test_policies.hpp`.
- Local fixtures added for the all-generation comparison:
  `tests/models/Qwen3-0.6B-Q8_0.gguf` and
  `tests/models/Qwen3-4B-Q4_K_M.gguf`.
- All-generation benchmark comparison was run with `EMEL_GENERATION_WORKLOAD_ID=all`,
  one measured run, no warmup, and `EMEL_BENCH_GENERATION_REFERENCE_THREADS=8`.
  Qwen3 4B still loses to llama.cpp in every measured route:
  max_tokens_1 `290706958 ns/op` vs `231552208`, max_tokens_10 `582315125`
  vs `490153791`, max_tokens_100 `3417727750` vs `3319405167`, and
  max_tokens_1000 `42790547250` vs `34659454584`.
- Current full quality-gate blocker is not generation: the broad benchmark gate
  expands into `speech_codec_mimi`, and `scripts/bench_mimi_compare.sh
  --reference=moshi_cpp` fails token-exact parity with `code_match_fraction =
  0.953125`. The `speech_codec_mimi_mlx` lane passes. Do not claim a full green
  gate until the Moshi/Mimi token-exact lane is fixed or explicitly scoped out.
- Latest scoped quality-gate verification for the pulled threaded-generation
  branch passed with:
  `EMEL_BUILD_JOBS=4 EMEL_BENCH_GENERATION_REFERENCE_THREADS=8
  EMEL_QUALITY_GATES_TIMEOUT=7200s
  EMEL_QUALITY_GATES_BENCH_SUITE=generation,parallel_matmul,graph_processor,kernel_aarch64,sm_scheduler,decode_wavefront
  scripts/quality_gates.sh`. Selected lane statuses were all zero:
  benchmark snapshot, changed-line coverage, paritychecker, fuzz smoke, and
  determinism check. Non-fatal decode-wavefront benchmark regression warnings
  were emitted.
- Latest two-lane correction verification passed with:
  `BENCH_COMPARE_BUILD_DIR=build/bench_tools_ninja_generation_lanes
  EMEL_BUILD_JOBS=4 EMEL_QUALITY_GATES_TIMEOUT=7200s
  EMEL_QUALITY_GATES_CHANGED_FILES="$changed_files"
  EMEL_QUALITY_GATES_BENCH_SUITE=generation scripts/quality_gates.sh`.
  The scoped benchmark snapshot ran both generation lanes by default and
  reported `multithreaded` at `91204208 ns/op` vs llama.cpp `300555958`
  (`0.303x`) and `single` at `410694792 ns/op` vs llama.cpp `302352708`
  (`1.358x`) for
  `lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1`. Changed-line
  coverage was `34/34 (100.0%)`, changed-branch coverage was `17/30 (56.7%)`,
  and paritychecker, fuzz smoke, determinism, lint snapshot, and docs generation
  completed successfully.
- Follow-up benchmark reporting correction applied: generation benchmark records
  now include generated-token throughput as `tokens_per_second` in the shared
  benchmark result schema, generation JSONL contract, compare summaries,
  `bench_runner` text output, `scripts/bench.sh` snapshot extraction, and
  generated benchmark docs. Historical compare snapshot rows that predate this
  field still have blank token/s docs cells until that snapshot is regenerated
  from a token/s-emitting compare run.
- Latest scoped token/s verification passed with:
  `EMEL_BUILD_JOBS=4 EMEL_QUALITY_GATES_TIMEOUT=7200s
  EMEL_QUALITY_GATES_CHANGED_FILES="$changed_files"
  EMEL_QUALITY_GATES_BENCH_SUITE=generation,parallel_matmul
  scripts/quality_gates.sh`. The generation lane now reports both latency and
  token throughput; the final gated LFM2 1.2B one-token run reported
  `multithreaded` EMEL `95061875 ns/op` / `10.519 tokens/s` vs llama.cpp
  `303115750 ns/op` / `3.299 tokens/s`, and `single` EMEL `412951042 ns/op` /
  `2.422 tokens/s` vs llama.cpp `298027292 ns/op` / `3.355 tokens/s`.
- Fresh Qwen3 4B one-token compare was run with both lanes:
  `EMEL_BENCH_GENERATION_ITERS=1 EMEL_BENCH_GENERATION_RUNS=1
  EMEL_BENCH_GENERATION_WARMUP_ITERS=0
  EMEL_BENCH_GENERATION_WARMUP_RUNS=0
  scripts/bench_generation_compare.sh --reference-backend llama_cpp_generation
  --reference-threads 8 --workload-id
  qwen3_4b_single_user_hello_max_tokens_1_v1 --output-dir
  build/generation_compare_qwen3_4b_tokens`. Outputs matched exactly in both
  lanes. Single-lane EMEL was faster at `1100606292 ns/op` /
  `0.909 tokens/s` vs llama.cpp `1211642500 ns/op` / `0.825 tokens/s`;
  multithreaded EMEL was slower at `317726833 ns/op` / `3.147 tokens/s` vs
  llama.cpp `197450917 ns/op` / `5.065 tokens/s`.
- Current DI-correction verification: `EMEL_BUILD_JOBS=4 scripts/build_with_zig.sh`
  passed. Focused tests passed for `parallel matmul*`,
  `generator_requires_construction_time_dependencies`, streamed generation,
  `generator_detail_prepare_derives_kv_geometry_from_memory_contract`, Qwen3
  and Gemma4 q/k norm detail coverage, and the full
  `tests/text/generator/action_guard_tests.cpp` source-file slice. Bench runner
  and bench-runner test builds passed, and
  `./build/bench_tools_ninja_tests/bench_runner_tests --test-case='generation_stage_probe*' --no-skip`
  passed. The first scoped quality-gate rerun exposed one direct-fixture test
  that still relied on implicit route defaults; that fixture now injects
  `emel::text::generator::test::k_generation_route_policy`, and the failure is
  fixed under both zig and coverage binaries.
- Current DI-correction gate verification passed with:
  `EMEL_BUILD_JOBS=4 EMEL_QUALITY_GATES_PARALLEL=never
  EMEL_QUALITY_GATES_BENCH_SUITE=generation,parallel_matmul
  EMEL_QUALITY_GATES_CHANGED_FILES="$changed_files" scripts/quality_gates.sh`.
  The scoped benchmark snapshot, changed-line coverage, paritychecker, fuzz
  smoke, determinism check, lint snapshot, and docs generation all completed
  successfully. Changed-line coverage was `200/204 (98.0%)`; changed-branch
  coverage was `114/202 (56.4%)`. The scoped generation lane reported LFM2
  one-token `multithreaded` EMEL `93091500 ns/op` / `10.742 tokens/s` vs
  llama.cpp `306978375 ns/op` / `3.258 tokens/s`, and `single` EMEL
  `383651292 ns/op` / `2.607 tokens/s` vs llama.cpp `305508333 ns/op` /
  `3.273 tokens/s`. The scoped `parallel_matmul` suite emitted non-fatal
  regression warnings for `ggml_gemm8_f32` and `ggml_gemv_q6_k`, but the gate
  completed successfully.
- Current broad full-gate blocker is still not generation: the broad benchmark
  gate expands into `speech_codec_mimi`, and `scripts/bench_mimi_compare.sh
  --reference=moshi_cpp` fails token-exact parity with `code_match_fraction =
  0.953125`. The `speech_codec_mimi_mlx` lane passes. Do not claim a full
  unscoped green gate until the Moshi/Mimi token-exact lane is fixed or
  explicitly scoped out.
- Fresh DI-branch 1000-token Qwen compare was rerun with both lanes, one measured
  run, no warmup, and an 8-thread llama.cpp multithreaded reference. Qwen3 0.6B
  single lane: EMEL `15527198209 ns/op` / `64.403 tokens/s` vs llama.cpp
  `19422845209 ns/op` / `51.486 tokens/s` (EMEL faster). Qwen3 0.6B
  multithreaded lane: EMEL `11987232125 ns/op` / `83.422 tokens/s` vs llama.cpp
  `11011163916 ns/op` / `90.817 tokens/s` (EMEL slower). Qwen3 4B single lane:
  EMEL `83996730375 ns/op` / `11.905 tokens/s` vs llama.cpp `74950374459 ns/op`
  / `9.900 tokens/s`; llama.cpp stopped at 742 output tokens, so tokens/s is the
  fairer throughput comparison and EMEL is ahead there despite higher total
  wall time. A current rerun of the Qwen3 4B multithreaded pluggable compare
  after the stage-probe selector change shows EMEL losing: `53387228333 ns/op` /
  `18.731 tokens/s` vs llama.cpp `34270789667 ns/op` / `29.179 tokens/s`. All
  Qwen runs are `bounded_drift`, not exact output matches.
- Fresh Qwen compare records include total generation latency and generated-token
  throughput. Qwen prefill reporting was made explicit through
  `EMEL_GENERATION_STAGE_PROBE=selected` instead of being hidden behind
  `current_publication`. The selected stage probe can now print Qwen rows, but
  EMEL still reports public `generate` time as `actor_public_generate` with
  `emel_prefill_ns=0` and the work in `emel_unattributed_ns`; separate EMEL
  prefill/decode breakdown needs a new explicit public probe contract. The
  latest Qwen3 0.6B single-lane selected probe reported EMEL
  `17607694959 ns/op` / `56.793 tokens/s` vs llama.cpp `19231584542 ns/op` /
  `51.998 tokens/s`, with reference `prefill_ns=126055084`. The latest Qwen3 4B
  multithreaded selected probe reported EMEL `44856931042 ns/op` /
  `22.293 tokens/s` vs llama.cpp `35004383333 ns/op` / `28.568 tokens/s`, with
  reference `prefill_ns=161160458`.
- Stage-probe selector verification passed:
  `./build/bench_tools_ninja_tests/bench_runner_tests --test-case='generation_stage_probe*' --no-skip`.
- Current generation DI audit:
  - Now dependency-injected: `emel::text::generator::sm` requires a
    `dependencies` aggregate; the generator no longer owns or constructs a
    hidden matmul lane pool. `matmul::sm` requires an explicit
    `matmul::execution_policy` containing the injected lane pool, kernel kind,
    and active lane count. Generator route thresholds and host kernel kind flow
    through `runtime_policy`, with tool and test defaults held in
    `tools/generation_route_policy.hpp` and
    `tests/text/generator/generator_test_policies.hpp` instead of private
    generator detail.
  - Acceptable constants, not DI knobs: `k_prefill_q8_chunk_rows`,
    `k_prefill_q8_chunk8_rows`, and `matmul::k_max_matmul_lanes` are fixed
    kernel/actor shape limits. They should stay source constants unless the
    kernel shape itself becomes configurable.
  - Remaining hardcoded/default convenience: `make_auto_runtime_policy(...)`
    still chooses default route thresholds and detected host kernel kind for
    callers that explicitly ask for the auto policy. That is acceptable only as
    a caller-visible convenience; maintained benchmarks and tools must continue
    to pass explicit policies.
  - Remaining rule blocker: `detail::prepare(...)` no longer checks model-family
    strings for Qwen/Gemma/LFM, but it still derives
    `requires_attention_qk_norm`, `requires_attention_v_norm`, shortconv, and
    sliding-attention behavior from loaded block/model facts inside
    `detail.hpp`, then later compute helpers branch on those flags. That is
    better than family hardcoding, but it is still hidden behavior selection
    under the no-hidden-control-flow rule. The follow-up should move those
    choices into explicit guarded states/routes or a public variant contract,
    not leave them as detail-local booleans.
- Latest scoped DI/generation gate passed with:
  `EMEL_BUILD_JOBS=4 EMEL_QUALITY_GATES_PARALLEL=never
  EMEL_QUALITY_GATES_BENCH_SUITE=generation,parallel_matmul
  EMEL_QUALITY_GATES_CHANGED_FILES="$CHANGED_FILES" scripts/quality_gates.sh`.
  The gate completed domain boundaries, legacy SML surface scan, zig build,
  benchmark snapshot, changed-file coverage, paritychecker, fuzz smoke skip,
  determinism, lint snapshot, and docs generation. Coverage was
  `204/210 (97.1%)` changed lines and `130/260 (50.0%)` changed branches.
  The final LFM2 one-token generation compare reported multithreaded EMEL
  `87424875 ns/op` / `11.438 tokens/s` vs llama.cpp `310638833 ns/op` /
  `3.219 tokens/s`; single-lane EMEL remained slower at `367750459 ns/op` /
  `2.719 tokens/s` vs llama.cpp `310440209 ns/op` / `3.221 tokens/s`.
  `parallel_matmul` still emitted non-fatal f32 GEMM regression warnings, but
  every listed EMEL matmul compare row beat its reference row in that run.

## Completion Bar

- The targeted CPU lane must beat the corresponding `llama.cpp` reference lane.
- Snapshot updates are allowed for this goal when they reflect the verified new
  benchmark truth.
- Generation benchmark publication must report generated-token throughput, not
  only ops latency.
- Completion requires tests/benchmarks plus the recursive dual-subagent rule
  review signoff.
