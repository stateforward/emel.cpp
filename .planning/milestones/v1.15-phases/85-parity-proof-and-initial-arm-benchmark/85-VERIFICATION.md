# Phase 85 Verification

## Commands

- `git diff --check -- CMakeLists.txt tools/bench/bench_cases.hpp tools/bench/CMakeLists.txt tools/bench/bench_main.cpp tools/bench/diarization/sortformer_bench.cpp tests/diarization/sortformer/parity/lifecycle_tests.cpp docs/templates/benchmarks.md.j2 docs/benchmarks.md .planning/phases/85-parity-proof-and-initial-arm-benchmark snapshots/quality_gates/timing.txt`
  - Result: passed.
- `cmake --build build/coverage --target emel_tests_bin -j 6`
  - Result: passed.
- `cmake --build build/bench_tools_ninja --target bench_runner -j 6`
  - Result: passed.
- `ctest --test-dir build/coverage --output-on-failure -R '^emel_tests_diarization$' -j 1`
  - Result: passed, 1/1 tests.
- `EMEL_BENCH_SUITE=diarization_sortformer EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 build/bench_tools_ninja/bench_runner --mode=compare`
  - Result: passed; emitted EMEL and reference metadata with matching checksum
    `15712531076325547939`.
- `scripts/quality_gates.sh`
  - Result: exited 0.
  - Coverage: changed-source line coverage 96.3%, branch coverage 68.2%.
  - Notes: benchmark step reported `logits/sampler_sml/vocab_128000` variance and the expected
    `new benchmark entry without baseline` for
    `diarization/sortformer/canonical_multispeaker_16khz_15s_v1_profile_188x192x4`; the quality
    gate script continued and printed `warning: benchmark snapshot regression ignored by quality
    gates`.
- `scripts/generate_docs.sh`
  - Result: passed after moving Sortformer benchmark documentation into
    `docs/templates/benchmarks.md.j2`.

## Success Criteria

1. Repository tests compare EMEL output against a trusted reference baseline for one canonical
   multi-speaker fixture.
   - Covered by `tests/diarization/sortformer/parity/lifecycle_tests.cpp`.
2. EMEL/reference lanes remain isolated.
   - Covered by distinct benchmark append functions: EMEL decodes probabilities through EMEL output
     code; reference uses a fixed segment baseline and does not call EMEL.
3. One maintained ARM benchmark reports fixture identity, profile parameters, timing, and proof
   status.
   - Covered by `diarization_sortformer` suite and `# diarization_sortformer:` metadata output.
4. Documentation states supported model, input contract, output contract, and known
   pre-optimization limitations.
   - Covered by generated `docs/benchmarks.md` from `docs/templates/benchmarks.md.j2`.
