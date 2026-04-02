# benchmark snapshot policy

this project requires a benchmark snapshot for new state machines once they are marked complete.
the benchmark gate exists to prevent performance regressions and to force explicit, repeatable
performance characterization for each machine.

## summary

- every **new** `src/emel/**/sm.hpp` must include a benchmark readiness marker.
- markers live in the `sm.hpp` file header comment:
  - `// benchmark: designed` means no benchmark required yet.
  - `// benchmark: ready` means a benchmark **is required** and will be enforced by gates.
- benchmarks are only required for machines marked `ready`.
- benchmarks are enforced via a snapshot diff with a **30% regression tolerance in quality gates during rearchitecture**.
- reference comparison tracks the latest upstream `llama.cpp` ref by default.
- the default tracked upstream ref lives in `tools/bench/reference_ref.txt` and currently points at
  the moving `master` branch.

## benchmark harness

we use a single benchmark harness that lives in `tools/bench/bench_main.cpp` and
links EMEL with the reference allocator implementation for apples-to-apples
comparisons.
- output format is deterministic and diff-friendly:
  - `machine_path ns_per_op=123.4 iter=1000 runs=3`
  - `machine_path` matches the `src/emel/<path>/sm.hpp` path without the prefix and suffix.
- optional warmup controls (shared by EMEL + reference benches):
- `EMEL_BENCH_WARMUP_ITERS` (default: min(`EMEL_BENCH_ITERS`, 100))
  - `EMEL_BENCH_WARMUP_RUNS` (default: 1)
- toolchain defaults to zig (`zig cc` / `zig c++`) for determinism. use
  `scripts/bench.sh --system` if you explicitly want system compilers.
- compare builds record the resolved upstream SHA in their output so each run still tells you which
  `llama.cpp` revision you actually measured.
- `BENCH_REF_OVERRIDE` can still pin a branch, tag, or SHA when you explicitly need it.

## external reference compare (optional)

we also maintain an optional comparison mode that runs EMEL and reference benchmarks
in the same tool build.

to add a new comparison case:
1. add the EMEL case in `tools/bench/bench_main.cpp`.
2. add the matching reference case in `tools/bench/bench_main.cpp`.
3. keep the case names identical for proper matching.

the compare mode expects exact case-name matches and prints `ratio` values for quick inspection.
run it via `scripts/bench.sh --compare`.
when updating the comparison snapshot used by documentation, run:
`scripts/bench.sh --compare-update`.
the stored compare snapshot also records `# benchmark_config: ...` and the resolved upstream SHA,
so any publication-time env overrides and fetched `llama.cpp` revision are explicit in the artifact
instead of staying implicit in local shell state.

## canonical generation compare workflow

the canonical generation benchmark runs through the same compare surface. do not use a separate
generation-only script path.

run the normal compare workflow:

```bash
EMEL_BENCH_ITERS=1 \
EMEL_BENCH_RUNS=1 \
EMEL_BENCH_WARMUP_ITERS=0 \
EMEL_BENCH_WARMUP_RUNS=0 \
scripts/bench.sh --compare
```

the current maintained publication case name is:

`generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1`

to isolate that row from the normal compare output:

```bash
EMEL_BENCH_ITERS=1 \
EMEL_BENCH_RUNS=1 \
EMEL_BENCH_WARMUP_ITERS=0 \
EMEL_BENCH_WARMUP_RUNS=0 \
scripts/bench.sh --compare | \
rg '^generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 .* ratio='
```

the current maintained evidence workload is fixed to the checked-in Liquid
`LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture, prompt `hello`, and `max_tokens=1`. fixture loading
and one-time setup stay outside the timed loop; this case measures preloaded request latency.

the generation compare suite is additive across maintained supported fixtures. the current set
includes both:

- `generation/preloaded_request/qwen3_0_6b_q8_0_prompt_hello_max_tokens_1`
- `generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1`

## phase 13 flash-evidence publication workflow

phase 13 keeps flash-evidence publication on the existing benchmark surfaces only:
`scripts/bench.sh --compare`, `scripts/bench.sh --compare-update`, `snapshots/bench`,
`tools/docsgen`, and `docs/benchmarks.md`.

the BENCH-03 approval gate is the canonical short case:

`generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`

this phase-13 artifact is historical and stays tied to the archived llama canonical slice. it is
not the live maintained generation identity for the current Liquid benchmark publication.

before any checked-in snapshot refresh, compare the preserved non-flash artifact
`snapshots/bench/generation_pre_flash_baseline.txt` against the current compare snapshot with:

```bash
python3 tools/bench/compare_flash_baseline.py \
  --baseline snapshots/bench/generation_pre_flash_baseline.txt \
  --current snapshots/bench/benchmarks_compare.txt \
  --case generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1
```

the preserved baseline artifact is a key-value file with these required fields:

- `source_commit`
- `baseline_ref`
- `case`
- `baseline_emel_ns`
- `baseline_reference_ns`
- `baseline_ratio`

the comparator exits non-zero unless the current EMEL short-case latency is lower than the
preserved pre-flash baseline. phase 13 remains anchored to the short case until a trustworthy
maintained long-case baseline artifact exists.

stop and obtain explicit user approval before running `scripts/bench.sh --compare-update` or
checking in any new snapshot artifact under `snapshots/bench/`. do not treat local benchmark runs
as permission to refresh checked-in snapshot or generated benchmark evidence files.

## reading the generation compare row

compare mode prints one row per matched case:

```text
generation/preloaded_request/lfm2_5_1_2b_thinking_q4_k_m_prompt_hello_max_tokens_1 emel.cpp 550019417.000 ns/op, llama.cpp 481248875.000 ns/op, ratio=1.143x
```

- `emel.cpp` is the measured EMEL-side time for the canonical generation case.
- `llama.cpp` is the direct reference time for the same named case.
- `ratio` is `emel.cpp / llama.cpp`.
- `ratio > 1.0x` means EMEL is slower for that workload; `ratio < 1.0x` means EMEL is faster.

## generation-specific local overrides

the canonical generation case has its own bounded local validation knobs:

- `EMEL_BENCH_GENERATION_ITERS` overrides per-run iterations for the generation case only.
- `EMEL_BENCH_GENERATION_RUNS` overrides run count for the generation case only.
- `EMEL_BENCH_GENERATION_WARMUP_ITERS` overrides warmup iterations for the generation case only.
- `EMEL_BENCH_GENERATION_WARMUP_RUNS` overrides warmup run count for the generation case only.

for narrow local debugging, you can also isolate a single case with `EMEL_BENCH_CASE_INDEX`. if
you need seam-audit output for the generation case, set `EMEL_BENCH_AUDIT_GENERATION_SEAMS=1`;
that audit output stays on stderr and does not change the normal compare row on stdout.

## reproducible generation profiling

for a reproducible macOS time-profiler capture of the maintained generation benchmark group, use:

```bash
scripts/profile_generation.sh
```

this script:

- rebuilds `bench_runner` through the maintained bench workflow
- profiles the current generation benchmark group with `xctrace`
- writes trace, stdout, exported XML, and a compact hot-frame summary under `tmp/profiles/`
- optionally writes flamegraphs when `stackcollapse-instruments.pl` and `flamegraph.pl` are
  available on `PATH` or through `FLAMEGRAPH_DIR`

use `--out-basename=...` to control the artifact prefix and `--case-index=...` if the generation
group moves in `tools/bench/bench_main.cpp`.

## gate behavior

the benchmark gate script enforces the following:
- if a new `sm.hpp` file is added:
  - it must declare `// benchmark: designed` or `// benchmark: ready`.
  - if `ready`, a benchmark case **must** exist and be registered.
- snapshot regression failure if `ns_per_op` exceeds baseline by > tolerance.
  - default (`scripts/bench.sh`): 10% (`BENCH_TOLERANCE=0.10`)
  - current quality gate override (`scripts/quality_gates.sh` during rearchitecture): 30%
- snapshot baseline can be updated intentionally with `--update`.

## developer workflow

1. add a new machine `sm.hpp` with a benchmark marker.
2. if the machine is complete, set `// benchmark: ready` and add a benchmark case.
3. run `scripts/bench.sh --compare` when you need the live EMEL-vs-reference compare surface,
   including the canonical generation row.
4. run `scripts/bench.sh --snapshot` locally to verify, or
   `scripts/bench.sh --snapshot --update` to update the baseline.
5. run `scripts/bench.sh --compare-update` only when intentionally refreshing the compare
   snapshot used by generated benchmark docs. benchmark snapshot updates require explicit user
   approval.

## rationale

designed-but-unbenchmarked machines often produce trivial workloads that distort baselines. the
readiness marker allows incremental rollout without blocking CI, while still forcing benchmarks
once behavior is complete.
