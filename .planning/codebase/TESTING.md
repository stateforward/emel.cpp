# EMEL Testing

This repository uses layered verification: `doctest` unit and integration tests in `tests/`,
`ctest` registration in `CMakeLists.txt`, coverage and sanitizer scripts in `scripts/`, fuzzing
under `tests/fuzz`, parity checking in `tools/paritychecker`, benchmark baselines in `snapshots/`,
and documentation/lint checks as gate steps.

## Test Frameworks And Verification Types

- `doctest` is the primary C++ test framework. The runner is `tests/doctest_main.cpp`, and the
  header is expected at `third_party/doctest/doctest/doctest.h` by `CMakeLists.txt`.
- `CTest` is the top-level test launcher. `CMakeLists.txt` registers `emel_tests`,
  `lint_snapshot`, and `generate_docs` as named tests.
- Boost.SML state introspection is part of the test style. Tests use `is(...)`,
  `visit_current_states`, and `set_current_states(...)` rather than indirect probing, as seen in
  `tests/text/jinja/parser_tests.cpp`, `tests/text/jinja/formatter_tests.cpp`,
  `tests/batch/planner/planner_sm_flow_tests.cpp`, and `tests/text/detokenizer/detokenizer_tests.cpp`.
- Branch-path and guard/action tests are common, for example
  `tests/graph/allocator/allocator_action_branch_tests.cpp`,
  `tests/graph/assembler/assembler_action_branch_tests.cpp`,
  `tests/batch/planner/planner_action_branch_tests.cpp`, and
  `tests/text/tokenizer/tokenizer_action_guard_tests.cpp`.
- Wrapper/API visibility tests exist and are important for architecture enforcement, for example
  `tests/graph/wrapper_visibility_tests.cpp` and `tests/kernel/lifecycle_tests.cpp`.
- Fuzz tests use libFuzzer entrypoints with `extern "C" int LLVMFuzzerTestOneInput(...)` in
  `tests/fuzz/gguf_parser_fuzz.cpp`, `tests/fuzz/gbnf_parser_fuzz.cpp`,
  `tests/fuzz/jinja_parser_fuzz.cpp`, and `tests/fuzz/jinja_formatter_fuzz.cpp`.
- Parity tests compare EMEL behavior against reference behavior in `tools/paritychecker`, driven by
  `scripts/paritychecker.sh`.
- Benchmark regression checks live under `tools/bench` and compare against baselines in
  `snapshots/bench/`.

## Test Layout

- Test sources are organized by domain under `tests/`, mirroring `src/emel/`:
  `tests/gbnf/`, `tests/graph/`, `tests/kernel/`, `tests/logits/`, `tests/memory/`,
  `tests/model/`, `tests/text/`, `tests/token/`, and `tests/batch/`.
- Domain-specific support headers stay near the tests that use them, for example
  `tests/kernel/test_helpers.hpp` and `tests/text/encoders/test_support.hpp`.
- Model artifacts used by parser/tokenizer/loader tests live under `tests/models/`, with
  provenance and hashes documented in `tests/models/README.md`.
- Fuzz corpus directories live under `tests/fuzz/corpus/`.
- Benchmark sources are one-file-per-machine or subsystem under `tools/bench/`, matching the rule
  in `AGENTS.md`, for example `tools/bench/memory/kv_bench.cpp` and
  `tools/bench/logits/validator_bench.cpp`.

## What CMake Registers Today

- `CMakeLists.txt` builds a single `doctest` executable named `emel_tests_bin`.
- The registered `ctest` targets are:
  `emel_tests` running `emel_tests_bin`,
  `lint_snapshot` running `scripts/lint_snapshot.sh`,
  and `generate_docs` running `scripts/generate_docs.sh --check`.
- Fuzz executables are only built when `EMEL_ENABLE_FUZZ=ON` in `CMakeLists.txt`.
- `EMEL_ENABLE_TESTS` defaults to `ON`; `EMEL_ENABLE_FUZZ` defaults to `OFF` in `CMakeLists.txt`.

## How To Run Tests

### Default build

- `scripts/build_with_zig.sh`
  builds the main tree with `zig cc` and `zig c++` into `build/zig`.

### Unit and integration tests

- Configure and build manually with tests enabled:
  `cmake -S . -B build/dev -G Ninja -DEMEL_ENABLE_TESTS=ON`
- Build:
  `cmake --build build/dev --parallel`
- Run the main suite:
  `ctest --test-dir build/dev --output-on-failure -R emel_tests`
- Run the lint baseline check:
  `ctest --test-dir build/dev --output-on-failure -R lint_snapshot`
- Run documentation drift checks:
  `ctest --test-dir build/dev --output-on-failure -R generate_docs`

### Coverage

- `scripts/test_with_coverage.sh`
  creates `build/coverage` with `gcc` and `g++`, runs `ctest -R emel_tests`, then enforces
  coverage thresholds with `gcovr`.
- Required tools are checked explicitly in `scripts/test_with_coverage.sh`:
  `cmake`, `ctest`, `gcovr`, `clang-format`, `llvm-cov`, `llvm-profdata`, `gcc`, and `g++`.
- Defaults are line coverage `>= 90%` and branch coverage `>= 50%`, controlled by
  `LINE_COVERAGE_MIN` and `BRANCH_COVERAGE_MIN` in `scripts/test_with_coverage.sh`.

### Sanitizers

- `scripts/test_with_sanitizers.sh`
  builds separate `asan_ubsan` and `tsan` trees under `build/sanitizers/` using `clang` and
  `clang++`.
- It runs `ctest -R emel_tests` for each sanitizer configuration.

### Fuzzing

- `scripts/fuzz_smoke.sh`
  enables `EMEL_ENABLE_FUZZ=ON`, builds into `build/fuzz`, and runs short corpus-based fuzz smoke
  passes for the parser/formatter fuzzers.
- Fuzz smoke requires a Clang toolchain with libFuzzer support; the script has macOS Homebrew LLVM
  detection and explicit failure messages when the runtime is unavailable.

### Parity checking

- `scripts/paritychecker.sh`
  configures `tools/paritychecker` with Zig, builds `build/paritychecker_zig`, and runs
  `ctest -R paritychecker_tests`.
- This is the reference-comparison layer required by `scripts/quality_gates.sh`.

### Benchmarks and snapshots

- `scripts/bench.sh --snapshot --compare`
  is the main benchmark regression command used by the quality gates.
- Benchmark baselines live in `snapshots/bench/benchmarks.txt` and
  `snapshots/bench/benchmarks_compare.txt`.
- `scripts/bench.sh --snapshot --update` updates the snapshot baseline and should be treated as a
  deliberate baseline change, not a routine step.

### Full gate

- `scripts/quality_gates.sh`
  is the repository’s expected aggregate validation command per `AGENTS.md`.
- It currently runs:
  `scripts/build_with_zig.sh`,
  `scripts/test_with_coverage.sh`,
  `scripts/paritychecker.sh`,
  `scripts/fuzz_smoke.sh`,
  `scripts/bench.sh --snapshot --compare`,
  and `scripts/generate_docs.sh`.
- Timing output is persisted to `snapshots/quality_gates/timing.txt`.

## Coverage And Quality Gate Details

- Coverage excludes tests and also excludes transition-table headers matching
  `src/emel/.*/sm.hpp` in `scripts/test_with_coverage.sh`.
- That means the line threshold focuses more on guards, actions, helpers, kernels, and wrapper
  logic than on the declarative transition-table rows themselves.
- `scripts/lint_snapshot.sh` is a baseline-style formatting regression check against
  `snapshots/lint/clang_format.txt`, not a straight “all files must match current clang-format”
  check.
- `scripts/generate_docs.sh --check` treats generated documentation as part of the gate.
- `scripts/quality_gates.sh` wraps the whole run in a timeout using `timeout` or `gtimeout` and
  fails fast if that tool is missing.
- Benchmark regressions are intentionally soft-failed in `scripts/quality_gates.sh` right now:
  a benchmark mismatch emits a warning and does not fail the gate.

## Repository-Specific Testing Norms

- `AGENTS.md` requires a failing unit test before fixing a reported bug.
- `AGENTS.md` also expects SML introspection for machine assertions and names `ctest` targets
  `emel_tests` and `lint_snapshot` explicitly.
- Test naming is expected to be machine or behavior scoped, which matches files like
  `tests/tensor/lifecycle_tests.cpp`, `tests/logits/validator/validator_tests.cpp`, and
  `tests/text/tokenizer/bpe_regex_tests.cpp`.
- Tests are expected to stay non-monolithic; the tree largely follows that by splitting behavior
  into lifecycle, action/guard, branch, and flow tests.

## Inferred Gaps And Risks

- `scripts/quality_gates.sh` does not currently run `scripts/test_with_sanitizers.sh`. The script
  exists, but sanitizer coverage is not part of the default aggregate gate.
- `scripts/quality_gates.sh` also comments out `lint_snapshot` with a note that the baseline is
  intentionally not enforced during hard-cutover migration. `CMakeLists.txt` still registers
  `lint_snapshot`, so policy and current aggregate enforcement differ.
- `scripts/quality_gates.sh` notes that sanitizer coverage is temporarily disabled because of an
  SML/UBSan issue under ASan. That is a real gap in default defect detection, not just a missing
  convenience command.
- Two test files in `tests/graph/processor/` are present but not registered in `EMEL_TEST_SOURCES`
  in `CMakeLists.txt`: `tests/graph/processor/processor_sm_transition_tests.cpp` and
  `tests/graph/processor/processor_tests.cpp`. Current gate builds therefore omit part of the
  graph processor test surface.
- `CMakeLists.txt` also contains a comment about legacy processor tests being disabled during the
  hard cutover, which reinforces that graph-processor coverage is intentionally incomplete today.
- Coverage excludes `src/emel/.*/sm.hpp`, so transition-table declarations can drift without
  affecting the line threshold as long as surrounding actions/guards remain tested.
- `AGENTS.md` asks for tests that validate determinism, bounded internal work, and “no allocations
  during dispatch,” but there is no obvious dedicated allocation-instrumentation test harness in
  `tests/` and no visible gate step that enforces the no-allocation rule directly.
- `AGENTS.md` expects line coverage `>= 90%`; branch coverage in
  `scripts/test_with_coverage.sh` is only `>= 50%`, so branch-path rigor depends more on targeted
  test authoring than on the configured numeric threshold.
- Bench regressions are warning-only in `scripts/quality_gates.sh`, which means performance drift
  is visible but not gate-blocking at the moment.
- Fuzzing is smoke-level by default: `scripts/fuzz_smoke.sh` runs roughly 10 seconds per target and
  is not integrated into `ctest`, so deeper fuzz campaigns must be run intentionally.
