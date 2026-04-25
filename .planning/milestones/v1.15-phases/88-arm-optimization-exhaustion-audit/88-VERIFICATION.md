# Phase 88 Verification

## Commands

- `EMEL_BENCH_SUITE=diarization_sortformer EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 build/bench_tools_ninja/bench_runner --mode=compare`
  - Result: passed.
  - Final evidence: `end_to_end_ns=2609458`, `transformer_ns=2356041`, checksum
    `15712531076325547939`.
- `scripts/generate_docs.sh`
  - Result: passed.
- `git diff --check -- docs/templates/benchmarks.md.j2 docs/benchmarks.md .planning/phases/88-arm-optimization-exhaustion-audit`
  - Result: passed.
- Latest implementation gate from Phase 87:
  - `scripts/quality_gates.sh` exited 0 after the dense-helper optimization.
  - Coverage: changed-source line coverage 96.6%, branch coverage 68.5%.
  - Benchmark warnings were limited to the expected unsnapshotted `diarization/sortformer` rows.

## Success Criteria

1. Final ARM profile shows no remaining material hotspot with a clear repo-owned optimization path
   under the maintained constraints.
   - Covered: transformer remains material, but the clear path is a broader kernel-owned
     dense/matmul contract rather than another local loop item.
2. Benchmark evidence records final timing, optimization history, rejected candidates, and
   remaining opportunities.
   - Covered in `docs/templates/benchmarks.md.j2` and generated `docs/benchmarks.md`.
3. Closeout docs state supported model, contracts, evidence, limitations, and future work.
   - Covered by the Sortformer Diarization Baseline section in generated benchmark docs.
