---
phase: 165
plan: 01
status: complete
requirements-completed:
  - LANE-02
key_files:
  modified:
    - tools/bench/CMakeLists.txt
    - tools/bench/batch/planner_bench.cpp
    - tools/bench/bench_runner_tests.cpp
    - tools/bench/diarization/sortformer_bench.cpp
    - tools/bench/diarization/sortformer_fixture.hpp
    - tools/bench/text/jinja/formatter_bench.cpp
    - tools/bench/text/jinja/parser_bench.cpp
completed: 2026-05-01
---

# Summary

Phase 165 closed the benchmark actor-boundary enforcement gap for `LANE-02`.

## Changes

- Removed direct benchmark use of actor `action::context` from the Jinja parser and formatter
  benchmark harnesses by using the public state-machine wrappers.
- Removed direct batch planner action-constant reach-through by deriving local benchmark capacity
  from the public event scratch surface.
- Removed diarization request/pipeline/executor actor-detail reach-through from maintained
  Sortformer benchmark sources, keeping only non-actor output/feature diagnostic constants and
  public error casting.
- Added a recursive source-backed `bench_runner_tests` check over maintained `tools/bench` source
  and header files for prohibited actor include and namespace patterns.
- Fixed scoped benchmark CMake filtering so `batch_planner`, `jinja_parser`, and
  `jinja_formatter` compile with the reference-side inputs they require during snapshot gates.

## Verification

Commands passed:

```sh
git diff --check -- tools/bench/CMakeLists.txt tools/bench/batch/planner_bench.cpp tools/bench/text/jinja/parser_bench.cpp tools/bench/text/jinja/formatter_bench.cpp tools/bench/diarization/sortformer_bench.cpp tools/bench/diarization/sortformer_fixture.hpp tools/bench/bench_runner_tests.cpp .planning/phases/165-benchmark-actor-boundary-enforcement-closure .planning/ROADMAP.md .planning/STATE.md .planning/milestones/v1.19-REQUIREMENTS.md .planning/milestones/v1.19-ROADMAP.md
rg -n '/actions\.hpp|/guards\.hpp|::action::|::guard::|emel/(batch/planner|diarization/request|diarization/sortformer/(executor|pipeline)|text/(generator|jinja/(formatter|parser)))/detail\.hpp|emel::(batch::planner|diarization::request|diarization::sortformer::(executor|pipeline)|text::(generator|jinja::(formatter|parser)))::detail::' tools/bench --glob '*.cpp' --glob '*.hpp' -g '!bench_runner_tests.cpp'
scripts/check_domain_boundaries.sh
cmake --build build/bench_tools_phase93_kernel12 --target bench_runner_tests -j2
ctest --test-dir build/bench_tools_phase93_kernel12 --output-on-failure -R bench_runner_tests
EMEL_QUALITY_GATES_CHANGED_FILES="tools/bench/CMakeLists.txt:tools/bench/batch/planner_bench.cpp:tools/bench/text/jinja/parser_bench.cpp:tools/bench/text/jinja/formatter_bench.cpp:tools/bench/diarization/sortformer_bench.cpp:tools/bench/diarization/sortformer_fixture.hpp:tools/bench/bench_runner_tests.cpp" EMEL_QUALITY_GATES_BENCH_SUITE="jinja_formatter,batch_planner,jinja_parser,diarization_sortformer" EMEL_QUALITY_GATES_BENCH_ITERS=5000 EMEL_QUALITY_GATES_BENCH_RUNS=7 EMEL_QUALITY_GATES_BENCH_WARMUP_ITERS=1000 EMEL_QUALITY_GATES_BENCH_WARMUP_RUNS=3 scripts/quality_gates.sh
```

Notes:

- The broad source scan intentionally returned no matches.
- Full unfiltered `bench_runner_tests` passed in 321.96 seconds.
- The quality gate rewrote `snapshots/quality_gates/timing.txt`; it was restored because snapshot
  updates require explicit approval.

Code review status: clean.
