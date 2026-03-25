# Tech Stack

## Scope

This map covers the concrete implementation stack in `CMakeLists.txt`, `src/emel/`,
`include/emel/`, `scripts/`, `tools/`, `docs/`, and `.github/`.

## Languages

- `CMakeLists.txt`, `tools/bench/CMakeLists.txt`, `tools/paritychecker/CMakeLists.txt`, and
  `tools/docsgen/CMakeLists.txt` define a C/C++ build centered on `C` and `CXX`.
- `CMakeLists.txt` sets `CMAKE_C_STANDARD 11` and `CMAKE_CXX_STANDARD 20`, so the repo is a
  mixed `C11` and `C++20` codebase.
- `src/emel/**` and `include/emel/**` contain the production implementation, with the public ABI
  surface starting in `include/emel/emel.h` and internal/public C++ helpers in
  `include/emel/callback.hpp` and `include/emel/error/error.hpp`.
- `scripts/*.sh` are Bash entry points for build, coverage, parity, fuzzing, docs, lint
  snapshots, and aggregate gates.
- `.github/workflows/*.yml` and `.github/commands/*.toml` add CI/automation configuration.
- `README.md`, `docs/**/*.md`, and `docs/templates/*.j2` carry project and generated-doc
  content.

## Runtime And Build Outputs

- `CMakeLists.txt` defines `emel_core` as an `INTERFACE` target exposing headers from `include/`,
  `src/`, and fetched Boost.SML headers.
- `CMakeLists.txt` defines `emel` as a `STATIC` library. The current compiled translation unit is
  `src/emel/model/data.cpp`, while most behavior lives header-first under `src/emel/**`.
- `src/emel/machines.hpp` aggregates the main machine aliases and shows the intended top-level
  runtime surface for model loading, tokenization, memory, generation, graph execution, and text
  processing.
- `include/emel/emel.h` is the current C ABI anchor; the ABI is intentionally thin and fixed-width.

## Core Frameworks And Libraries

- Boost.SML is the orchestration framework. It is fetched in `CMakeLists.txt` and
  `tools/docsgen/CMakeLists.txt`, pinned by `cmake/sml_version.cmake`, and used throughout
  `src/emel/**/sm.hpp`.
- `third_party/doctest/doctest/doctest.h` is the unit-test framework wired into `CMakeLists.txt`
  and `tools/paritychecker/CMakeLists.txt`.
- The standard library is the dominant utility layer. Notable examples are `std::regex` usage in
  `src/emel/text/unicode.hpp` and `tools/docsgen/docsgen.cpp`.
- There is no repo-level package manager manifest such as `vcpkg.json`, `conanfile.py`,
  `package.json`, or `pyproject.toml`; dependency wiring is centralized in `CMakeLists.txt`,
  `tools/bench/CMakeLists.txt`, `tools/paritychecker/CMakeLists.txt`, and
  `tools/docsgen/CMakeLists.txt`.

## Toolchain

- `CMakeLists.txt` requires CMake `3.20+`, exports `compile_commands.json`, and enforces warnings
  as errors with `/W4 /WX` on MSVC or `-Wall -Wextra -Wpedantic -Werror` elsewhere.
- `scripts/build_with_zig.sh` is the default dev build path and drives `cmake -G Ninja` with
  `zig cc` and `zig c++` into `build/zig`.
- `scripts/test_with_coverage.sh` switches to native `gcc` and `g++` and requires `ctest`,
  `gcovr`, `clang-format`, `llvm-cov`, and `llvm-profdata` for coverage enforcement.
- `scripts/fuzz_smoke.sh` builds with `clang` and `clang++`, enables `EMEL_ENABLE_FUZZ`, and uses
  libFuzzer/ASan/UBSan.
- `scripts/paritychecker.sh` builds `tools/paritychecker` with Zig plus Ninja and runs
  `ctest -R paritychecker_tests`.
- `scripts/generate_docs.sh` builds `tools/docsgen` with Zig plus Ninja and regenerates
  `docs/architecture/*` and `docs/benchmarks.md`.

## Test And Verification Stack

- `CMakeLists.txt` registers `emel_tests`, `lint_snapshot`, and `generate_docs` as `ctest`
  targets.
- `tests/doctest_main.cpp` is the main unit-test entry point, with behavior split across
  `tests/kernel/`, `tests/text/`, `tests/graph/`, `tests/memory/`, `tests/model/`, and related
  directories.
- `tests/fuzz/*.cpp` provides fuzz targets for GGUF, GBNF, and Jinja parsing/formatting.
- `scripts/quality_gates.sh` is the top-level verifier and composes `scripts/build_with_zig.sh`,
  `scripts/test_with_coverage.sh`, `scripts/paritychecker.sh`, `scripts/fuzz_smoke.sh`,
  `scripts/bench.sh`, and `scripts/generate_docs.sh`.
- `docs/compliance-report.md` shows an additional static-audit layer over `src/emel/**/sm.hpp`.

## Dependency Layers

- Production code under `src/emel/**` depends directly on Boost.SML and the standard library.
- Test code depends on `third_party/doctest/doctest/doctest.h`.
- `tools/bench/CMakeLists.txt` and `tools/paritychecker/CMakeLists.txt` add tool-only reference
  dependencies on `llama.cpp` and `ggml`; those dependencies are intentionally not part of the
  main `emel` target.
- `tools/bench/CMakeLists.txt` optionally discovers `nlohmann/json.hpp` through `find_path`, but
  that dependency is bench-tool scoped, not core-runtime scoped.

## Configuration Surfaces

- Root CMake options live in `CMakeLists.txt`: `EMEL_ENABLE_TESTS` and `EMEL_ENABLE_FUZZ`.
- The Boost.SML source pin is controlled in `cmake/sml_version.cmake` through
  `EMEL_BOOST_SML_GIT_REPOSITORY` and `EMEL_BOOST_SML_GIT_TAG`.
- Benchmark configuration lives in `scripts/bench.sh`, `tools/bench/bench_main.cpp`, and
  `tools/bench/gbnf/bench_main.cpp` via `EMEL_BENCH_ITERS`, `EMEL_BENCH_RUNS`,
  `EMEL_BENCH_WARMUP_ITERS`, `EMEL_BENCH_WARMUP_RUNS`, `EMEL_BENCH_CASE_INDEX`,
  `BENCH_TOLERANCE`, `BENCH_REF_OVERRIDE`, `BENCH_BASE_REF`, `BENCH_CC`, `BENCH_CXX`,
  `BENCH_BUILD_DIR`, and `BENCH_COMPARE_BUILD_DIR`.
- Memory benchmark model selection is configurable in `tools/bench/memory/kv_bench.cpp`,
  `tools/bench/memory/recurrent_bench.cpp`, and `tools/bench/memory/hybrid_bench.cpp` through
  `EMEL_BENCH_KV_MODEL`, `EMEL_BENCH_RECURRENT_MODEL`, and `EMEL_BENCH_HYBRID_MODEL`.
- Coverage thresholds are configured in `scripts/test_with_coverage.sh` through
  `LINE_COVERAGE_MIN`, `BRANCH_COVERAGE_MIN`, and `GCOVR_IGNORE_PARSE_ERRORS`.
- Docs generation accepts `DOCSGEN_BUILD_DIR` in `scripts/generate_docs.sh`.
- Gate timeout control lives in `scripts/quality_gates.sh` through
  `EMEL_QUALITY_GATES_TIMEOUT`.
- Reference pins live in `tools/bench/reference_ref.txt` and
  `tools/paritychecker/reference_ref.txt`.
- Snapshot/config artifacts live in `snapshots/bench/`, `snapshots/lint/clang_format.txt`, and
  `snapshots/quality_gates/timing.txt`.

## Notable Non-Core Backends

- Backend-specific machine families exist in `src/emel/kernel/aarch64/`, `src/emel/kernel/x86_64/`,
  `src/emel/kernel/wasm/`, `src/emel/kernel/cuda/`, `src/emel/kernel/metal/`, and
  `src/emel/kernel/vulkan/`.
- Those backend directories are present in the core source tree, but the root build in
  `CMakeLists.txt` does not currently link CUDA, Metal, Vulkan, or Emscripten SDKs as explicit
  external packages.
