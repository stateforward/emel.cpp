---
phase: 90
status: passed
verified: 2026-04-23
requirements:
  - PRF-01
  - PRF-02
  - BEN-01
  - DOC-01
  - RUN-03
  - OUT-03
---

# Phase 90 Verification

## Requirement Evidence

| Requirement | Evidence | Status |
|-------------|----------|--------|
| PRF-01 | `tests/diarization/sortformer/parity/lifecycle_tests.cpp` drives `sortformer/pipeline::sm` against the canonical PCM fixture and compares emitted segments to the recorded baseline. | Passed |
| PRF-02 | `tools/bench/diarization/sortformer_bench.cpp` keeps EMEL pipeline execution and recorded reference baseline construction in separate lanes without shared runtime/cache/model state. | Passed |
| BEN-01 | `EMEL_BENCH_SUITE=diarization_sortformer build/bench_tools_ninja/bench_runner --mode=compare` publishes pipeline workload metadata, output checksum `13736986938186292950`, and matching EMEL/reference output dimensions. | Passed |
| DOC-01 | `docs/templates/benchmarks.md.j2` and generated `docs/benchmarks.md` describe the pipeline-backed EMEL lane, recorded-baseline reference limitation, and current checksum. | Passed |
| RUN-03 | Benchmark and parity harnesses now enter the maintained runtime through `pipeline::sm::process_event(...)` rather than output helpers or fixed probability fills. | Passed |
| OUT-03 | Repeated pipeline fixture runs produce byte-stable probabilities/segments in the pipeline lifecycle test and a stable benchmark checksum. | Passed |

## Commands

- `git diff --check -- CMakeLists.txt tests/diarization/sortformer/pipeline/lifecycle_tests.cpp tests/diarization/sortformer/parity/lifecycle_tests.cpp tools/bench/diarization/sortformer_bench.cpp tools/bench/diarization/sortformer_fixture.hpp docs/templates/benchmarks.md.j2 docs/benchmarks.md`
- `cmake --build build/coverage --target emel_tests_bin -j 6`
- `ctest --test-dir build/coverage --output-on-failure -R '^emel_tests_diarization$' -j 1`
- `cmake --build build/bench_tools_ninja --target bench_runner -j 6`
- `EMEL_BENCH_SUITE=diarization_sortformer build/bench_tools_ninja/bench_runner --mode=compare`
- `scripts/generate_docs.sh`
- `scripts/quality_gates.sh`

## Results

- Focused diarization shard: passed.
- Focused Sortformer benchmark compare: passed with matching checksum `13736986938186292950`.
- Full quality gate: passed with 6/6 test shards, changed-source line coverage `95.0%`, branch
  coverage `68.1%`, paritychecker/fuzz/docs stages complete, and tolerated benchmark snapshot
  warnings for existing/new benchmark entries.
