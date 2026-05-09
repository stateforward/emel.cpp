---
phase: 252
status: passed
requirements:
  - PERF-02
---

# Phase 252 Verification

## Result: Passed

`PERF-02` is satisfied by maintained constrained-RAM profiling evidence for the cooperative async
loading path. The benchmark reports the actual `cooperative_async` model-loader path, source
residency, model/tensor byte sizes, constrained window, chunk size, peak resident bytes, dispatch
counts, and load elapsed time.

## Evidence

| Check | Result | Evidence |
|-------|--------|----------|
| Maintained path | pass | Generation benchmark drives public model-loader load events into `io/loader` with `load_strategy=cooperative_async` |
| Truthful residency | pass | Profile labels `source_residency=setup_time_fixture_file_image_and_full_target_storage`; no true larger-than-RAM source-residency claim is made |
| Required metadata | pass | Output reports model file bytes, tensor data bytes, effective RAM constraint, async chunk bytes, peak resident bytes, dispatch counts, and load elapsed ns |
| Scoped optimization | pass | 64 KiB window: 11,165 load dispatches and 26,901,458 ns load time; 1 MiB window: 780 dispatches and 23,317,750 ns load time |
| Benchmark command | pass | `EMEL_MODEL_LOAD_IO_STRATEGY=cooperative_async EMEL_MODEL_LOAD_CONSTRAINED_RAM_BYTES=1048576 EMEL_MODEL_LOAD_CHUNK_BYTES=1048576 scripts/bench.sh --snapshot --compare --suite=generation` exited 0 |
| Build | pass | `cmake --build build/bench_tools_ninja_generation --target bench_runner` exited 0 |
| Scoped quality gate | pass | Changed-file scoped `scripts/quality_gates.sh` exited 0 |

## Notes

This is constrained-RAM emulation on the maintained publication fixture. It does not claim
device-specific async loading, OS-backed larger-than-RAM streaming, or broader async inference.
