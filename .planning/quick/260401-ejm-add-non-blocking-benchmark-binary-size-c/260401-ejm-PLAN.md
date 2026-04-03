---
phase: quick-260401-ejm-add-non-blocking-benchmark-binary-size-c
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - tools/bench/CMakeLists.txt
  - tools/bench/bench_main.cpp
  - tools/bench/bench_runner_tests.cpp
  - tools/docsgen/docsgen.cpp
autonomous: true
requirements:
  - VERIFY-02
  - BENCH-07
must_haves:
  truths:
    - "`bench_runner --mode=compare` emits one deterministic EMEL-vs-llama binary size metadata line next to the maintained compare output."
    - "`scripts/bench.sh --snapshot --compare` carries the size metadata through the benchmark surface for operator reference without introducing a new blocking threshold."
    - "`scripts/generate_docs.sh --check` and `scripts/quality_gates.sh` stay green without forcing an immediate snapshot refresh just to tolerate the new metadata."
  artifacts:
    - path: "tools/bench/CMakeLists.txt"
      provides: "bench_runner access to the built EMEL and llama artifact paths"
    - path: "tools/bench/bench_main.cpp"
      provides: "compare-mode binary size metadata emission"
    - path: "tools/bench/bench_runner_tests.cpp"
      provides: "regression coverage for the new compare metadata contract"
    - path: "tools/docsgen/docsgen.cpp"
      provides: "optional parse/publish or safe-ignore support for the size metadata line"
  key_links:
    - from: "tools/bench/CMakeLists.txt"
      to: "tools/bench/bench_main.cpp"
      via: "compile definitions carrying target file paths"
      pattern: "TARGET_FILE:emel|TARGET_FILE:llama"
    - from: "tools/bench/bench_main.cpp"
      to: "tools/bench/bench_runner_tests.cpp"
      via: "compare-mode stdout contract"
      pattern: "binary_size_compare"
    - from: "snapshots/bench/benchmarks_compare.txt"
      to: "tools/docsgen/docsgen.cpp"
      via: "optional comment-line metadata parsing"
      pattern: "# binary_size_compare:"
---

<objective>
Add a non-blocking EMEL-vs-llama binary size comparison to the maintained benchmark compare path.

Purpose: Give operators a stable size reference inside the existing benchmark and quality-gate flow
without changing the warning-only benchmark policy or turning size drift into a new gate.
Output: One compare-metadata contract in `bench_runner`, regression coverage for it, and docs/gate
compatibility on the maintained path.
</objective>

<execution_context>
@/Users/gabrielwillen/.codex/get-shit-done/workflows/execute-plan.md
@/Users/gabrielwillen/.codex/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/STATE.md
@.planning/REQUIREMENTS.md
@scripts/quality_gates.sh
@scripts/bench.sh
@tools/bench/CMakeLists.txt
@tools/bench/bench_main.cpp
@tools/bench/bench_runner_tests.cpp
@tools/docsgen/docsgen.cpp

<interfaces>
From `tools/bench/bench_main.cpp` compare mode today:

```cpp
std::printf("# reference_impl: source=%.*s ref=%.*s\n", ...);
print_benchmark_config(cfg);
std::printf("%s emel.cpp %.3f ns/op, llama.cpp %.3f ns/op, ratio=%.3fx\n", ...);
```

From `scripts/quality_gates.sh` benchmark policy today:

```bash
if run_step_allow_fail bench_snapshot env ... "$ROOT_DIR/scripts/bench.sh" --snapshot --compare; then
  bench_status=0
else
  bench_status=$?
fi
...
if [[ $bench_status -ne 0 ]]; then
  echo "warning: benchmark snapshot regression ignored by quality gates" >&2
fi
```

Implication for this quick plan:
- keep the new size signal reference-only and non-blocking
- do not add a new tolerance check, failure path, or snapshot requirement
- preserve current compare row formatting so existing snapshot/docs consumers keep working
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add binary-size metadata to compare-mode benchmark output</name>
  <files>tools/bench/CMakeLists.txt, tools/bench/bench_main.cpp, tools/bench/bench_runner_tests.cpp</files>
  <behavior>
    - Test 1: `bench_runner --mode=compare` prints one `# binary_size_compare:` metadata line.
    - Test 2: the metadata line includes EMEL and llama byte counts when both target artifacts are available.
    - Test 3: the compare command still succeeds and preserves existing benchmark rows and metadata.
  </behavior>
  <action>In `tools/bench/CMakeLists.txt`, inject the built `emel` and `llama` target artifact paths into `bench_runner` via compile definitions so the compare binary can inspect the actual maintained build products. In `tools/bench/bench_main.cpp`, add a small helper that reads those artifact sizes and emits a deterministic `# binary_size_compare:` comment before the latency rows during compare mode only. If either artifact path is missing or unreadable, emit an explicit `status=unavailable` metadata line instead of failing. Keep this reference-only per BENCH-07: do not add threshold logic, do not affect compare exit codes, and do not alter existing row formatting. In `tools/bench/bench_runner_tests.cpp`, extend the compare-mode regression test to assert the metadata line exists and that its stable fields parse without depending on exact byte counts.</action>
  <verify>
    <automated>cmake -S tools/bench -B build/bench_tools_ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF && cmake --build build/bench_tools_ninja --parallel --target bench_runner_tests && ctest --test-dir build/bench_tools_ninja --output-on-failure -R bench_runner_tests</automated>
  </verify>
  <done>Compare-mode benchmark output now carries one deterministic binary-size metadata record and the bench-runner regression suite proves it.</done>
</task>

<task type="auto">
  <name>Task 2: Keep docs and gate surfaces compatible with the new metadata</name>
  <files>tools/docsgen/docsgen.cpp</files>
  <action>Update `tools/docsgen/docsgen.cpp` so the compare snapshot parser safely accepts the new `# binary_size_compare:` metadata line and can optionally surface it in the generated benchmark evidence section when present. Keep the parser backward compatible with older snapshots that do not contain the new line, and do not make binary-size metadata mandatory for docs generation. Do not change benchmark policy in `scripts/quality_gates.sh`; the maintained behavior must remain non-blocking and warning-only. Do not refresh checked-in snapshots or generated docs in this quick task unless the user later approves a baseline/documentation update.</action>
  <verify>
    <automated>scripts/generate_docs.sh --check</automated>
  </verify>
  <done>Docs generation tolerates or publishes the new size metadata without breaking the current maintained compare/docs flow or requiring an immediate snapshot refresh.</done>
</task>

</tasks>

<verification>
- `cmake -S tools/bench -B build/bench_tools_ninja -G Ninja -DCMAKE_BUILD_TYPE=Release -DEMEL_ENABLE_TESTS=OFF && cmake --build build/bench_tools_ninja --parallel --target bench_runner_tests && ctest --test-dir build/bench_tools_ninja --output-on-failure -R bench_runner_tests`
- `scripts/generate_docs.sh --check`
- `scripts/quality_gates.sh`
</verification>

<success_criteria>
- Compare-mode benchmark output includes one non-blocking EMEL-vs-llama binary size metadata line.
- The maintained benchmark snapshot/docs pipeline accepts the new metadata without breaking old snapshots.
- Full quality gates still behave the same way on benchmark drift: informative, not newly blocking.
</success_criteria>

<output>
After completion, create `.planning/quick/260401-ejm-add-non-blocking-benchmark-binary-size-c/260401-ejm-SUMMARY.md`
</output>
