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
  - Remaining architectural cleanup: `detail::prepare(...)` no longer checks
    model-family strings for Qwen/Gemma/LFM, and the first follow-up moved
    residual, q/k-norm, value, v-norm, and attention-window behavior into
    explicit `emel::model::llama::detail::generation_*_route` fields on the
    model generation execution descriptor. The generator now copies those route
    fields into `block_weights` instead of carrying generator-local booleans.
    The compute layer loop no longer branches helper-locally on
    `block_uses_attention(...)`, `requires_attention_qk_norm(...)`, or
    `requires_attention_v_norm(...)`; scalar, chunk4, and chunk8 residual
    selection is now made by explicit SML route tables in the canonical
    `text/generator/layer` component.
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
- Qwen3 4B discrepancy investigation update: the output mismatch traced to the
  q4 packed chunk-prefill route selecting the experimental q4 x q8 i8mm
  multi-RHS tile for `matrix_x4` / `matrix_x8`. That route was not scalar
  parity-safe over multi-block q4 rows. The fix keeps q6 prepared-q8/i8mm
  enabled, but prevents q4 BL8 prepared chunk routes from selecting the i8mm
  tile; q4 now uses the parity-safe dotprod tile for chunk prefill.
- Control-flow cleanup from the same investigation: graph KV validation no
  longer infers flash vs nonflash by comparing the request's function pointer.
  The selected attention mode is carried explicitly in `compute_io` by the
  action templates, so validation reads event/request state instead of doing
  hidden behavior selection from `request.run_kernel`.
- Regression coverage added for the 4B issue: nonzero q4 hybrid fixture tests
  compare nonflash chunk4 and chunk8 prefill against the scalar q8_k path, q4
  BL8 multi-block matrix-x4 compares against the scalar row reference, and q6
  prepared-q8/i8mm multi-block vector and matrix-x4 compare against scalar to
  prove the kept i8mm path.
- Focused verification after the minimal-diff rebuild passed:
  `build/zig-generator-detail/emel_tests_bin --no-breaks
  --test-case="*nonzero_hybrid_fixture*"`,
  `build/zig-generator-detail/emel_tests_bin --no-breaks
  --test-case="generator_detail_graph_callbacks_reject_incoherent_kv_snapshots"`,
  `build/zig-generator-detail/emel_tests_bin --no-breaks
  --test-case="*prefill*"`, `build/zig/emel_tests_bin --no-breaks
  --test-case="*q4_k_packed_bl8*"`, and `build/zig/emel_tests_bin
  --no-breaks --test-case="*q6_k_prepared_q8_rhs_i8mm*"`.
- Coverage follow-up verification also passed for the explicit action and
  kernel guard coverage added to satisfy the changed-branch gate:
  `build/zig-generator-detail/emel_tests_bin --no-breaks
  --test-case="*generator core actions cover reset*"`,
  `build/zig-generator-detail/emel_tests_bin --no-breaks
  --test-case="*generator prefill actions publish*"`,
  `build/zig/emel_tests_bin --no-breaks
  --test-case="kernel_aarch64_detail_branch_paths"`, and
  `build/zig/emel_tests_bin --no-breaks
  --test-case="kernel_aarch64_q4_k_vector_q8_rhs_tail_rows_match_scalar"`.
- Final scoped 4B-discrepancy gate passed with:
  `EMEL_BUILD_JOBS=4 EMEL_QUALITY_GATES_PARALLEL=never
  EMEL_QUALITY_GATES_SCOPE=changed
  EMEL_QUALITY_GATES_CHANGED_FILES="$changed_files"
  EMEL_QUALITY_GATES_BENCH_SUITE=generation,kernel_aarch64,parallel_matmul
  EMEL_QUALITY_GATES_TIMEOUT=7200s scripts/quality_gates.sh`. The selected
  benchmark lanes, changed-file coverage, paritychecker, fuzz smoke skip,
  determinism check, and docs skip completed successfully. Changed coverage was
  `777/783 (99.2%)` lines and `328/554 (59.2%)` branches. The final gate's
  selected benchmark rows showed LFM2 multithreaded generation EMEL
  `89887291 ns/op` / `11.125 tokens/s` vs llama.cpp `293679583 ns/op` /
  `3.405 tokens/s`; LFM2 single-lane EMEL still lost at `379691416 ns/op` /
  `2.634 tokens/s` vs llama.cpp `294916209 ns/op` / `3.391 tokens/s`.
  `kernel_aarch64` and `parallel_matmul` EMEL rows beat their listed reference
  rows in that final gate run.
- Current Qwen3 4B 100-token compare after the q4 chunk fix exact-matches both
  lanes. Single lane: EMEL `8631446417 ns/op` / `11.586 tokens/s` vs llama.cpp
  `10038588000 ns/op` / `9.962 tokens/s`. Multithreaded lane: EMEL
  `3640301375 ns/op` / `27.470 tokens/s` vs llama.cpp `3204083792 ns/op` /
  `31.210 tokens/s`.
- Current Qwen3 4B 1000-token evidence after the q4 chunk fix: multithreaded
  exact-matches and EMEL remains slower (`45085521917 ns/op` /
  `22.180 tokens/s` vs llama.cpp `34547428625 ns/op` / `28.946 tokens/s`).
  Single lane is `bounded_drift`, not a current EMEL-vs-llama exact mismatch:
  EMEL generated 1000 tokens at `11.821 tokens/s`, while llama.cpp one-thread
  stopped at 742 tokens at `10.067 tokens/s`, sharing 61.66% of bytes.
- Route-contract cleanup update: model-layer generation descriptors now publish
  explicit `generation_residual_route`, `generation_attention_qk_norm_route`,
  `generation_attention_value_route`, `generation_attention_v_norm_route`, and
  `generation_attention_window_route` values. `generator::detail::block_weights`
  consumes those route fields directly; the old generator-local
  `uses_attention`, `uses_shortconv`, `requires_attention_qk_norm`,
  `uses_shared_kv_value`, `requires_attention_v_norm`, and
  `uses_sliding_attention` booleans were removed. Generator guard support scans
  and tests now use the explicit residual/value/norm routes. Verification
  passed with a new focused shard build:
  `cmake -S . -B build/zig-model-route -G Ninja ... -DEMEL_TEST_SHARDS=model_and_batch,generator_and_runtime`,
  `cmake --build build/zig-model-route --target emel_tests_bin`, and focused
  doctest filters `model_llama_detail_describes_*`,
  `generator_detail_routes_static_*`,
  `generator_detail_gemma4_shared_kv_layer_rms_norms_value_branch_before_cache`,
  `*generator core actions cover reset*`, and
  `*generator prefill actions publish*`.
- Layer route actor update: scalar/chunk4/chunk8 layer residual selection now
  runs through explicit SML route tables over residual route plus q/k and value
  norm route combinations. The numeric work is split into route-specialized
  residual bodies, so q/k norm and v-norm decisions are compile-time inside the
  selected action instead of runtime booleans inside the compute loop. The
  temporary header-local `layer_route_actor` has now been extracted to the
  canonical `src/emel/text/generator/layer/{events,guards,actions,sm}.hpp`
  component. Its event surface consumes model-generic
  `emel::model::generation_*_route` types; `model::llama::detail` aliases those
  types for its descriptor, so generic generator events no longer depend on
  llama-specific route enum names. Focused verification passed with
  `cmake --build build/zig-model-route --target emel_tests_bin`,
  `model_llama_detail_describes_*`, `generator_detail_*qk_norm*`,
  `generator_detail_gemma4_shared_kv_layer_rms_norms_value_branch_before_cache`,
  `generator_detail_routes_static_*`,
  `generator_detail_route_templates_reject_unprepared_inputs`, `*prefill*`, and
  `scripts/lint_snapshot.sh`. Bounded Qwen3 4B compare was rerun before the
  canonical extraction after rebuilding `bench_runner` with the repo job limit:
  100-token output exact-matches both lanes (single EMEL `11.884 tokens/s` vs
  llama.cpp `10.074`; multithreaded EMEL `28.005 tokens/s` vs llama.cpp
  `31.860`), and 1000-token output remains single-lane `bounded_drift` (EMEL
  `12.284 tokens/s` vs llama.cpp `10.208`, 61.66% prefix) while multithreaded
  exact-matches (EMEL `22.814 tokens/s` vs llama.cpp `27.904`).
- Fresh current-tree Qwen3 4B discrepancy rerun after canonical layer extraction
  used `scripts/bench_generation_compare.sh --reference-backend
  llama_cpp_generation --reference-threads 8` with one measured run, no warmup,
  and output directories `build/generation_compare_current_qwen3_4b_tokens_100_fresh`
  and `build/generation_compare_current_qwen3_4b_tokens_1000_fresh`. The
  100-token run exact-matches both lanes: single EMEL `8342912167 ns/op` /
  `11.986 tokens/s` vs llama.cpp `10471132792 ns/op` / `9.550 tokens/s`;
  multithreaded EMEL `4033642333 ns/op` / `24.791 tokens/s` vs llama.cpp
  `3184509041 ns/op` / `31.402 tokens/s`. The 1000-token run keeps the same
  shape as the saved evidence: multithreaded exact-matches but loses throughput
  (EMEL `43777061458 ns/op` / `22.843 tokens/s` vs llama.cpp `34769558416 ns/op`
  / `28.761 tokens/s`), while single lane is bounded drift rather than the
  original q4 chunk mismatch (EMEL 1000 tokens at `87819843166 ns/op` /
  `11.387 tokens/s`, llama.cpp 742 tokens at `77420592666 ns/op` /
  `9.584 tokens/s`, shared prefix 61.66%).
- Follow-up no-hidden-control-flow cleanup: residual-route guard predicates no
  longer call `emel::text::generator::detail::block_uses_attention(...)`.
  The predicate now lives in `src/emel/text/generator/guards.hpp`, and the
  stale chunk prefill backend-readiness helpers were removed from
  `src/emel/text/generator/detail.hpp` so route support decisions cannot
  regress back into detail helpers. `detail.hpp` now uses direct route-field
  checks only in construction/setup code that binds model-owned block data.
  Structural coverage was tightened in
  `tests/text/generator/lifecycle_tests.cpp`: it now fails if guard files call
  the detail residual helper, if `detail.hpp` reintroduces the removed
  backend-readiness helpers, or if the canonical layer route actor hides
  residual choice in actions or llama-specific event contracts. Verification
  passed with `cmake --build build/zig-model-route --target emel_tests_bin`,
  focused doctest cases
  `generator_route_guards_do_not_delegate_behavior_selection_to_detail_helpers`,
  `generator_scalar_kernel_route_choice_stays_in_state_machines`,
  `generator_layer_route_actor_keeps_residual_choice_explicit`,
  `generator_detail_route_templates_reject_unprepared_inputs`,
  `generator_detail_scalar_routes_run_prepared_qwen3_paths`, and
  `scripts/lint_snapshot.sh`.
- 4B q4 BL8 discrepancy guardrail: the dead
  `dot_q4_k_x8_q8_k_group_bl8_i8mm_x4` helper was removed from
  `src/emel/kernel/aarch64/actions.hpp` so the Qwen3 4B q4 BL8 generation path
  cannot silently re-enter the known parity-unsafe i8mm fold. The q4 BL8
  `matrix_x8` kernel regression now uses `col_count = QK_K * 10u`, matching the
  multi-block shape that the previous single-block test missed, and compares
  the packed route against scalar q4/q8 arithmetic. Verification used a fresh
  kernel-scoped zig build configured with `EMEL_TEST_SHARDS=kernel_and_graph`;
  CMake reported `-mcpu=native+dotprod+i8mm`. Focused doctest cases passed:
  `*multi_block_matrix_x8*` (`1` case, `71` assertions) and `*bad_i8mm*` (`1`
  case, `2` assertions).
- Layer actor completion cleanup: scalar/chunk4/chunk8 layer route actors no
  longer compose residual plus feed-forward work with action-local `&&`, and
  the local wrapper no longer passes a by-reference `ok` latch through
  `layer::event::*_run`. Each route now transitions to `state_residual_done`,
  uses `sml::completion<event::*_run>` plus explicit residual outcome guards to
  either run feed-forward or mark failure, then uses feed-forward outcome guards
  to mark success or failure before returning to idle. Generator detail calls
  the layer-owned `process_scalar`, `process_chunk4`, and `process_chunk8`
  wrappers instead of constructing raw `*_sm` instances and returning
  `actor.process_event(ev) && ok`. Verification passed with
  `cmake --build build/zig-model-route --target emel_tests_bin`, the focused TU
  compile for `tests/text/generator/lifecycle_tests.cpp.o`, and doctest cases
  `generator_layer_route_actor_keeps_residual_choice_explicit`,
  `generator_detail_route_templates_reject_unprepared_inputs`, and
  `generator_detail_scalar_routes_run_prepared_qwen3_paths`.
- Fresh post-layer-fix Qwen3 4B 1000-token compare used
  `qwen3_4b_single_user_hello_max_tokens_1000_v1`
  (`Qwen3-4B-Q4_K_M.gguf`) with one measured run and no warmup, output in
  `build/generation_compare_current_qwen3_4b_tokens_1000_after_layer_fix`.
  Single lane remains `bounded_drift` with the same prefix shape:
  EMEL generated `1000` tokens at `12.128 tokens/s`, llama.cpp generated `742`
  tokens at `10.358 tokens/s`, shared prefix `1486` bytes / `61.66%`. The
  multithreaded lane exact-matches: EMEL `1000` tokens at `22.598 tokens/s`
  (`44251277959 ns/op`) vs llama.cpp `1000` tokens at `19.224 tokens/s`
  (`52018009209 ns/op`). Treat this apparent multithreaded win as unstable
  until repeated: the previous current-tree run had EMEL in the same band
  (`22.843 tokens/s`) but llama.cpp much faster (`28.761 tokens/s`), so the
  difference is likely reference-run variance or host state, not a proven EMEL
  improvement. The benchmark still reports `prepare_ns_per_op`,
  `encode_ns_per_op`, and `publish_ns_per_op` as `0` for both lanes because this
  workload is a `generation/preloaded_request` case; prefill/phase throughput
  needs separate instrumentation or a non-preloaded workload.
- Fresh scoped gate after canonical layer extraction and generic route type
  ownership cleanup passed with:
  `EMEL_BUILD_JOBS="$EMEL_BUILD_JOBS" EMEL_QUALITY_GATES_PARALLEL=never
  EMEL_QUALITY_GATES_SCOPE=changed
  EMEL_QUALITY_GATES_CHANGED_FILES="$changed_files"
  EMEL_QUALITY_GATES_BENCH_SUITE=generation,kernel_aarch64,parallel_matmul
  EMEL_BENCH_GENERATION_LANES=both EMEL_QUALITY_GATES_TIMEOUT=7200s
  scripts/quality_gates.sh`. Selected lanes completed: legacy SML scan, scoped
  zig build, generation benchmark, kernel_aarch64 benchmark, parallel_matmul
  benchmark, changed-source coverage, paritychecker, determinism, lint
  snapshot, and docs generation. Fuzz smoke was skipped as irrelevant to the
  changed files. Changed coverage was `2227/2349 (94.8%)` lines and
  `741/1434 (51.7%)` branches. The gate's generation rows were LFM2
  multithreaded EMEL `99750458 ns/op` / `10.025 tokens/s` vs llama.cpp
  `303990333 ns/op` / `3.290 tokens/s`, and LFM2 single-lane EMEL
  `398612625 ns/op` / `2.509 tokens/s` vs llama.cpp `306376083 ns/op` /
  `3.264 tokens/s`. `kernel_aarch64` and `parallel_matmul` EMEL rows beat
  their listed reference rows.
- Fresh scoped gate after the 4B q4 BL8 guardrail and layer completion cleanup
  also passed with the same serialized command shape:
  `EMEL_BUILD_JOBS="$EMEL_BUILD_JOBS" EMEL_QUALITY_GATES_PARALLEL=never
  EMEL_QUALITY_GATES_SCOPE=changed
  EMEL_QUALITY_GATES_CHANGED_FILES="$changed_files"
  EMEL_QUALITY_GATES_BENCH_SUITE=generation,kernel_aarch64,parallel_matmul
  EMEL_BENCH_GENERATION_LANES=both EMEL_QUALITY_GATES_TIMEOUT=7200s
  scripts/quality_gates.sh`. Selected lanes completed: legacy SML scan, scoped
  zig build, generation benchmark with both single and multithreaded lanes,
  kernel_aarch64 benchmark, parallel_matmul benchmark, changed-source coverage,
  parity dependency/test checks, determinism, lint snapshot, and docs
  generation. Fuzz smoke was skipped because no fuzz-affecting files changed.
  Changed coverage was `2381/2574 (92.5%)` lines and `791/1540 (51.4%)`
  branches. The gate's generation rows were LFM2 multithreaded EMEL
  `96148000 ns/op` / `10.401 tokens/s` vs llama.cpp `309455667 ns/op` /
  `3.231 tokens/s`, and LFM2 single-lane EMEL `402073834 ns/op` /
  `2.487 tokens/s` vs llama.cpp `296805042 ns/op` / `3.369 tokens/s`.
  `kernel_aarch64`, `parallel_matmul`, `paritychecker_tests`, and the
  cross-process determinism check all passed.
- Single-thread Qwen3 4B parity investigation update: the temporary
  flash-attention route toggle was removed because forcing EMEL single through
  the shared flash implementation did not change output and only slowed the
  lane. The cleaned current runner again reports
  `thread_contract="emel_serial_matmul_lanes=1"`, with `36792` optimized flash
  dispatches and `0` shared flash dispatches for the 1000-token Qwen3 4B run.
  Focused verification passed with
  `./build/bench_tools_ninja_tests/bench_runner_tests --test-case='bench_runner generation jsonl emits manifest-driven workload metadata and explicit comparability' --no-skip`
  and focused aarch64 flash doctests after rebuilding `build/zig/emel_tests_bin`.
- Current cleaned single-lane compare for
  `qwen3_4b_single_user_hello_max_tokens_1000_v1` is in
  `build/generation_compare_qwen3_4b_single_after_no_route`. It remains
  `bounded_drift`: EMEL generated `1000` tokens / `2410` bytes at
  `12.262 tokens/s` with checksum `11222026555239503021`; llama.cpp
  `n_threads=1,n_threads_batch=1` generated `742` tokens / `2159` bytes at
  `10.274 tokens/s` with checksum `7218963136212605388`. Shared prefix is
  `1486` bytes (`61.6598%`), `output_tokens_delta=258`, and
  `output_bytes_delta=251`.
- Current cleaned multithreaded-lane compare for the same workload is in
  `build/generation_compare_qwen3_4b_multithreaded_after_no_route` and remains
  exact-match: both lanes generated `1000` tokens / `2410` bytes with matching
  checksum, EMEL at `22.643 tokens/s` and llama.cpp at `30.383 tokens/s`.
- Reference-only decode-thread sweep for the same Qwen3 4B 1000-token workload
  shows the reference output itself is thread-sensitive: llama.cpp
  `n_threads=1,n_threads_batch=1` produced `742` tokens / checksum
  `7218963136212605388`; `n_threads=2` and `n_threads=4` both produced
  `1000` tokens / checksum `9696415458321741055`; `n_threads=8` produced
  `1000` tokens / checksum `11222026555239503021`, matching EMEL's current
  single-lane checksum. Batch-thread changes alone did not explain the drift;
  decode `n_threads` controls it.
- Token-trace diagnostics were added to generation benchmark JSONL artifacts as
  paired `.tokens.txt` sidecars. The traced single-lane compare in
  `build/generation_compare_qwen3_4b_single_token_trace` reproduced the same
  `bounded_drift` shape: EMEL `1000` tokens at `11.873 tokens/s`, llama.cpp
  `n_threads=1,n_threads_batch=1` `742` tokens at `9.514 tokens/s`, shared
  prefix `1486` bytes. The selected-token streams are identical through token
  index `617`, then diverge at index `618`: EMEL selects token `4906`, while
  llama.cpp one-thread selects token `91`. Text-wise this is where EMEL
  continues the repeated `<|im_start|>` loop and llama.cpp emits `<|im|>`
  followed by normal prose. This proves the single-lane mismatch starts at
  selected-token/logit argmax, not detokenization.
- A traced reference-only run with `n_threads=8,n_threads_batch=1` in
  `build/reference_jsonl_qwen3_4b_decode_8_trace` exactly matches EMEL's
  selected-token stream end to end: both traces have `1000` tokens and checksum
  `11222026555239503021`. The older `decode1_batch8` output has the same
  SHA-256 as default one-thread reference output, so prefill/batch threading is
  not the driver. Current next parity target is the decode flash-attention /
  argmax numeric path around token index `618`: either EMEL single must emulate
  llama.cpp's one-decode-thread chunking/reduction numerics, or the benchmark
  contract must intentionally split thread-sensitive reference outputs instead
  of treating one-thread and multithreaded llama.cpp as the same oracle.
- Tested and rejected one simple arithmetic hypothesis: changing EMEL flash
  online-softmax updates from `std::exp` to `::expf` produced an identical EMEL
  token stream, checksum, and divergence point (`618`). That experiment was
  reverted; do not treat `expf` alone as the remaining single-thread parity fix.
- PR `#100` follow-up correction: generator layer dispatch outcomes have been
  moved out of `text/generator/detail.hpp` and into the layer actor boundary.
  Scalar layer execution now carries streamed-layer errors on the
  `event::scalar_run` payload, prepares/acquires and normalizes through
  explicit `layer::sm` states, and routes prepare/normalize/residual/feed-forward
  success or failure through guards and completion transitions. Chunk4/chunk8
  prefill layer execution likewise normalizes through explicit layer actor
  phases before residual route selection. `layer/actions.hpp` stays free of the
  source-scanned hidden-control-flow tokens (`if (`, `switch`, ternary, and
  `&&`), while `detail.hpp` no longer owns `layer::process_scalar`,
  `layer::process_chunk4`, or `layer::process_chunk8` calls.
- Reviewer A follow-up correction: the layer dispatch facade no longer lives in
  `layer/actions.hpp` or the `layer::action` namespace. Production generator
  detail code, tests, and the paritychecker now call the public layer
  state-machine interface in `layer/sm.hpp` (`layer::run_layer*`), and
  `layer/actions.hpp` is limited to SML action bodies.
- Latest PR `#100` verification after that correction passed:
  `cmake --build build/zig --parallel "$EMEL_BUILD_JOBS" --target emel_tests_bin`,
  `ctest --test-dir build/zig -R 'emel_tests_(text|generator_and_runtime|sm)$' --output-on-failure`,
  `scripts/lint_snapshot.sh`, and
  `EMEL_QUALITY_GATES_CHANGED_FILES="src/emel/text/generator/detail.hpp,src/emel/text/generator/layer/actions.hpp,src/emel/text/generator/layer/sm.hpp,tests/text/generator/detail_tests.cpp,tests/text/generator/lifecycle_tests.cpp,tools/paritychecker/parity_engines.cpp" EMEL_QUALITY_GATES_COVERAGE_CHANGED_FILES="src/emel/text/generator/layer/sm.hpp" EMEL_QUALITY_GATES_BENCH_SUITE=generation scripts/quality_gates.sh`.
  The scoped gate selected generation benchmarks and all parity; benchmark
  snapshot, paritychecker, fuzz smoke, determinism, lint snapshot, and docs
  lanes completed successfully. Coverage ran with the current-fix facade file
  override and reported no changed source lines under that override. The
  generation benchmark still reports LFM2 single-lane behind llama.cpp and
  multithreaded ahead, with benchmark-regression warnings non-fatal under the
  scoped gate.

## Completion Bar

- The targeted CPU lane must beat the corresponding `llama.cpp` reference lane.
- Snapshot updates are allowed for this goal when they reflect the verified new
  benchmark truth.
- Generation benchmark publication must report generated-token throughput, not
  only ops latency.
- Completion requires tests/benchmarks plus the recursive dual-subagent rule
  review signoff.
