# Integrations

## Scope

This map covers external libraries, automation/tool integrations, reference-model hookups, and
the main boundaries between `src/emel/` and the outside world.

## External Libraries

- Boost.SML is the primary external framework. `CMakeLists.txt` and `tools/docsgen/CMakeLists.txt`
  fetch it from the repository pinned in `cmake/sml_version.cmake`, and every machine family under
  `src/emel/**/sm.hpp` depends on that integration pattern.
- `third_party/doctest/doctest/doctest.h` is a vendored test dependency used by the root test
  target in `CMakeLists.txt` and by `tools/paritychecker/CMakeLists.txt`.
- `tools/bench/CMakeLists.txt` optionally locates `nlohmann/json.hpp` through `find_path`, so JSON
  support is an optional bench-tool integration rather than a core runtime dependency.

## Reference Implementation Integrations

- `tools/bench/CMakeLists.txt` and `tools/paritychecker/CMakeLists.txt` integrate the upstream
  reference implementation from `https://github.com/ggml-org/llama.cpp.git` through
  `FetchContent` or a local checkout at `tmp/llama.cpp`.
- Both tool CMake files pin the upstream ref through `tools/bench/reference_ref.txt` and
  `tools/paritychecker/reference_ref.txt`.
- `tools/bench/CMakeLists.txt` pulls in upstream `ggml` and `llama` CMake targets with
  `add_subdirectory(${reference_impl_SOURCE_DIR}/ggml ggml)` and
  `add_subdirectory(${reference_impl_SOURCE_DIR}/src llama_src)`.
- `tools/paritychecker/CMakeLists.txt` does the same, then links EMEL against `ggml` and `llama`
  only inside the `paritychecker` and `paritychecker_tests` executables.
- `tools/bench/CMakeLists.txt` and `tools/paritychecker/CMakeLists.txt` both compile upstream
  Jinja implementation files from `common/jinja/*.cpp` for comparison tooling.
- `tools/paritychecker/tokenizer_parity_common.cpp` integrates directly with upstream
  `llama-vocab.h` to compare tokenizer behavior.
- `tools/bench/kernel/bench_common.hpp`, `tools/bench/memory/bench_common.hpp`, and
  `tools/paritychecker/parity_runner.cpp` call into `ggml` and `llama` APIs for apples-to-apples
  benchmark and parity runs.

## Tooling Integrations

- `scripts/quality_gates.sh` is the orchestration point that integrates build, coverage, parity,
  fuzz, benchmark, and docs tooling into one repo-level gate.
- `scripts/build_with_zig.sh`, `scripts/paritychecker.sh`, and `scripts/generate_docs.sh`
  integrate Zig as the default compiler frontend and Ninja as the generator/build runner.
- `scripts/test_with_coverage.sh` integrates native `gcc`/`g++`, `ctest`, `gcovr`, `llvm-cov`,
  and `llvm-profdata`.
- `scripts/fuzz_smoke.sh` integrates Clang/libFuzzer, ASan, and UBSan with the fuzz targets in
  `tests/fuzz/*.cpp`.
- `scripts/lint_snapshot.sh` integrates `clang-format` and Git snapshot baselines stored in
  `snapshots/lint/clang_format.txt`.
- `tools/docsgen/docsgen.cpp` integrates build-time machine introspection, template rendering from
  `docs/templates/README.md.j2` and `docs/templates/benchmarks.md.j2`, and writes generated docs
  into `docs/architecture/` and `docs/benchmarks.md`.

## GitHub And AI Automation

- `.github/workflows/gemini-review.yml` and `.github/workflows/gemini-invoke.yml` integrate
  `google-github-actions/run-gemini-cli@v0` into GitHub Actions.
- `.github/workflows/gemini-review.yml` and `.github/workflows/gemini-invoke.yml` also integrate
  `actions/create-github-app-token` and `actions/checkout` for GitHub-scoped credentials and repo
  checkout.
- The Gemini workflows configure a Docker-hosted GitHub MCP server from
  `ghcr.io/github/github-mcp-server:v0.27.0`, which is an external automation boundary even
  though it is CI-only.
- `.github/commands/gemini-review.toml` and related files under `.github/commands/` provide the
  prompt/program layer consumed by those workflows.

## Model And Data Integrations

- The main model format boundary is GGUF. Production loaders live in `src/emel/gguf/loader/sm.hpp`,
  `src/emel/model/loader/sm.hpp`, and `src/emel/model/weight_loader/sm.hpp`.
- Test and benchmark model artifacts are stored locally in `tests/models/`, with provenance,
  license, and download URLs documented in `tests/models/README.md`.
- `tools/bench/memory/kv_bench.cpp`, `tools/bench/memory/recurrent_bench.cpp`, and
  `tools/bench/memory/hybrid_bench.cpp` default to local GGUF files under `tests/models/` and
  allow overrides via environment variables.
- `tests/text/tokenizer/tokenizer_parity_tests.cpp` and the tooling under `tools/paritychecker/`
  form the tokenizer/reference-model integration surface.

## Public API And System Boundaries

- The public C ABI boundary starts in `include/emel/emel.h`; it uses `extern "C"` and fixed-width
  integer types rather than exposing templates or STL containers.
- Error-code normalization is shared through `include/emel/error/error.hpp`.
- Immediate synchronous callback handoff is exposed through `include/emel/callback.hpp`, which is
  used to bridge internal machines and callers without introducing queues.
- `src/emel/` is the implementation boundary for production code, while `tools/bench/`,
  `tools/paritychecker/`, and `tools/docsgen/` are intentionally separate integration zones with
  looser dependency rules.

## External Services And Network Use

- There is no evidence of HTTP client code or always-on external service calls in `src/emel/` or
  `include/emel/`; runtime integration is file-based and in-process.
- Network use is concentrated in build and CI edges: `cmake/sml_version.cmake` fetches Boost.SML,
  `tools/bench/CMakeLists.txt` and `tools/paritychecker/CMakeLists.txt` fetch `llama.cpp` when
  `tmp/llama.cpp` is absent, and `tests/models/README.md` records external Hugging Face download
  sources for local test assets.
- `.github/workflows/gemini-review.yml` and `.github/workflows/gemini-invoke.yml` add CI-only
  service integrations with GitHub, Google/Gemini credentials, and optional Vertex/Google API
  configuration.

## Notable Boundaries To Watch

- `src/emel/kernel/cuda/`, `src/emel/kernel/metal/`, `src/emel/kernel/vulkan/`, and
  `src/emel/kernel/wasm/` define backend-specific state-machine boundaries, but explicit vendor SDK
  linkage is not yet wired into the root `CMakeLists.txt`.
- `tools/mock_main.cpp` is a local harness that exercises model-loader and GGUF-loader boundaries
  directly through machine `process_event(...)` calls and is useful as a narrow integration probe.
