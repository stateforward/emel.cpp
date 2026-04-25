# Phase 86 Verification

## Commands

- `git diff --check -- tools/bench/bench_main.cpp tools/bench/diarization/sortformer_bench.cpp docs/templates/benchmarks.md.j2 docs/benchmarks.md .planning/phases/86-arm-sortformer-profiling-baseline`
  - Result: passed.
- `cmake --build build/bench_tools_ninja --target bench_runner -j 6`
  - Result: passed.
- `EMEL_BENCH_SUITE=diarization_sortformer EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 build/bench_tools_ninja/bench_runner --mode=compare`
  - Result: passed.
  - Evidence: emitted `feature_ns`, `encoder_ns`, `modules_cache_ns`, `transformer_ns`,
    `output_ns`, and `end_to_end_ns` for
    `diarization/sortformer/stage_profile_8f_profile_188x192x4`.
- `scripts/generate_docs.sh`
  - Result: passed.
- `scripts/quality_gates.sh`
  - Result: exited 0.
  - Coverage: changed-source line coverage 96.3%, branch coverage 68.2%.
  - Notes: benchmark step reported expected unsnapshotted entries for
    `diarization/sortformer/canonical_multispeaker_16khz_15s_v1_profile_188x192x4` and
    `diarization/sortformer/stage_profile_8f_profile_188x192x4`; the quality gate script continued
    and printed `warning: benchmark snapshot regression ignored by quality gates`.

## Success Criteria

1. Maintained ARM benchmark reports end-to-end timing plus stage attribution.
   - Covered by the `stage_profile_8f_profile_188x192x4` metadata line.
2. Profiling evidence identifies hotspots with enough specificity to choose an ownership layer.
   - Covered by owner hints for feature extractor, encoder, modules/cache, transformer, and output.
3. Profiling harness remains measurement-only.
   - Covered by benchmark code calling existing EMEL-owned stage functions and adding no production
     compute or fallback path in tools.
