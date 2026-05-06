---
phase: 219-maintained-read-source-provenance
status: passed
verified: 2026-05-05T21:15:00Z
requirements:
  - PLAT-01
  - TIO-03
  - VAL-04
---

# Phase 219 Verification

## Requirement Status

| Requirement | Status | Source-Backed Evidence |
|-------------|--------|------------------------|
| PLAT-01 | Passed | Maintained tool lanes no longer own duplicated full-file read helpers for read/copy source spans. Phase 222 supersedes the helper placement; setup-time source bytes now come from public `src/emel/io/source/any.hpp`. Dispatch-time `io/read` still consumes event-provided source spans. |
| TIO-03 | Passed | Generation, Sortformer diarization, embedded probe, and paritychecker EMEL lanes keep strategy selection/reporting through public model-loader/I/O surfaces while using the maintained source-byte helper. |
| VAL-04 | Passed | Source guardrails reject tool-local `read_file_bytes` substitutes and, after Phase 222 supersession, require public `emel::io::source::load_file_bytes` usage in maintained read/copy evidence lanes. |

## Verification Commands

- `cmake --build build/zig --target emel_tests_bin --parallel`
  - Passed: `ninja: no work to do.`
- `ctest --test-dir build/zig --output-on-failure -R emel_tests_model_and_batch`
  - Passed: 1/1 tests passed in 53.02s.
- `scripts/check_domain_boundaries.sh`
  - Passed.
- `EMEL_QUALITY_GATES_CHANGED_FILES='src/emel/io/read/detail.hpp tests/model/loader/lifecycle_tests.cpp tools/bench/diarization/sortformer_fixture.hpp tools/bench/generation_bench.cpp tools/embedded_size/emel_probe/main.cpp tools/paritychecker/parity_assets.cpp tools/paritychecker/parity_assets.hpp tools/paritychecker/parity_engines.cpp tools/paritychecker/paritychecker_tests.cpp .planning/phases/219-maintained-read-source-provenance/219-CONTEXT.md .planning/phases/219-maintained-read-source-provenance/219-01-PLAN.md .planning/phases/219-maintained-read-source-provenance/219-01-SUMMARY.md .planning/phases/219-maintained-read-source-provenance/219-VERIFICATION.md .planning/phases/219-maintained-read-source-provenance/219-VALIDATION.md .planning/ROADMAP.md .planning/REQUIREMENTS.md .planning/STATE.md' scripts/quality_gates.sh`
  - Passed. The scoped gate rebuilt with the zig toolchain, passed the legacy SML surface scan, and skipped unrelated benchmark, coverage, paritychecker, fuzz, and docs lanes based on the changed-file scope.
- `node .codex/get-shit-done/bin/gsd-tools.cjs roadmap analyze`
  - Passed. Phase 219 reported `disk_status: complete` and `roadmap_complete: true`; phases 220 and 221 remain incomplete.
- `rg -n "bool read_file_bytes\\(|read_file_bytes\\(" tools/bench/generation_bench.cpp tools/bench/diarization/sortformer_fixture.hpp tools/embedded_size/emel_probe/main.cpp tools/paritychecker/parity_assets.cpp tools/paritychecker/parity_engines.cpp`
  - Passed by returning no matches.
- Superseded by Phase 222: `rg -n "emel::io::source::load_file_bytes" src/emel/io/source/any.hpp tools/bench/generation_bench.cpp tools/bench/diarization/sortformer_fixture.hpp tools/embedded_size/emel_probe/main.cpp tools/paritychecker/parity_engines.cpp tools/paritychecker/paritychecker_tests.cpp`
  - Passed; found maintained helper definition and maintained lane usage.
- `build/zig/emel_tests_bin --no-breaks --source-file='*tests/io/read/lifecycle_tests.cpp' --test-case='io read can capture same-RTC result without public callbacks'`
  - Passed: 1/1 test passed. This test belongs to Phase 220 edits but confirms the shared binary remained runnable.
- `build/zig/emel_tests_bin --no-breaks --source-file='*tests/diarization/request/lifecycle_tests.cpp' --test-case='diarization request rejects invalid sample rate'`
  - Passed: 1/1 test passed.
- `build/zig/emel_tests_bin --no-breaks --source-file='*tests/diarization/request/lifecycle_tests.cpp'`
  - Passed: 6/6 tests passed.
- `build/zig/emel_tests_bin --no-breaks --source-file='*tests/diarization/sortformer/modules/lifecycle_tests.cpp'`
  - Passed before later dyld launch instability: 5/5 tests passed.
- `build/zig/emel_tests_bin --no-breaks --source-file='*tests/diarization/sortformer/output/lifecycle_tests.cpp'`
  - Passed before later dyld launch instability: 5/5 tests passed.

## Diarization Shard Investigation

The full `emel_tests_diarization` shard was investigated because a rerun in the
shared workspace hung inside:

`build/zig/emel_tests_bin --no-breaks --source-file=*tests/diarization/*`

Targeted evidence shows this is not a Phase 219 source-provenance failure:

- The long-running test case
  `sortformer parity matches maintained real model and audio checksum baseline`
  completed all six fixture preparation assertions before it was terminated:
  model source load, model ready, PCM source load, PCM ready, baseline source
  load, and baseline ready.
- A process sample taken while that test was still running showed the stack in
  `sortformer_fixture::run_pipeline`, `pipeline::sm::process_event`, and
  `compute_encoder_frames_from_features`, with hot frames in encoder convolution,
  depthwise, pointwise, and AArch64 matmul kernels. The process footprint was
  about 1.0G.
- The maintained source fixtures involved are large enough to make this expected
  to be expensive on the debug/UBSan test binary: the Sortformer model fixture is
  449M, the audio fixture is 470K, and the baseline is 53B.
- The sampled stack was compute-bound after source loading had already passed,
  not blocked in source-byte loading, `std::fopen`, `std::fread`, or other
  Phase 219 source-provenance code.

Several rapid or concurrent direct launches of `emel_tests_bin` also hit the
known intermittent macOS dyld cache launch failure:

`dyld cache '(null)' not loaded: syscall to map cache into shared region failed`

That dyld launch failure was not treated as a source failure; sequential narrow
test invocations still passed as listed above.
