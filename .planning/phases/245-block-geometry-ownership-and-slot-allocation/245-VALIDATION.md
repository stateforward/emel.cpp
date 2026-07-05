# Phase 245 Validation

**Date:** 2026-07-04
**Host:** arm64 macOS (worktree hopeful-hodgkin-0166f9)

## Gate evidence

- Build: `cmake --build build/zig` clean (compile verified by Building-line count,
  not just exit code).
- `ctest emel_tests` + `lint_snapshot`: 14/14 pass (1811 doctest cases across the
  13 shards — real full run, not shard-vacuous).
- Scoped quality gates (`EMEL_QUALITY_GATES_CHANGED_FILES`, comma-separated):
  - `test_with_coverage`: PASS (335s)
  - `paritychecker`: PASS (93s) — LFM2.5 lane green after the tool block-math fix
  - `fuzz_smoke`: skipped (no fuzz-affecting files)
  - `bench_snapshot`: FAIL — **pre-existing main breakage, not this phase** (below)

## bench_snapshot disposition (pre-existing, evidence-backed)

Two independent failures, both reproduced on PRISTINE origin/main sources
(`git stash push -- src tools tests` + targeted rebuild + run → exit 138):

1. **Reference-lane SIGBUS:** `bench_runner --mode=compare` crashes in
   `libggml-base.0.dylib::ggml_is_quantized` (EXC_BAD_ACCESS, wild ggml_type;
   single-frame unwind) on the LFM2.5 generation workload, deterministic on this
   arm64 host with the current reference pin (ggml 0.10.2, commit c5a3bc39b).
   EMEL-only mode is clean for 1 and 1000-token workloads under lldb. A bench_runner
   binary built 2026-05-05 from the older pin runs compare clean — the regression
   came with the reference pin bump, not with EMEL changes. This also explains the
   previously "unreproduced SIGBUS in a full-gate generation run".
2. **Missing baselines:** suites added by PR #89 (graph/processor_*,
   decode_wavefront/*) have zero rows in `snapshots/bench/benchmarks.txt` →
   "new benchmark entry without baseline". Snapshot update requires explicit
   consent.

Both are tracked for follow-up (background task chip spawned; owner decision
needed on baseline regeneration and reference-pin fix). Phase 245's own runtime
surface is covered by the passing unit/coverage/parity lanes; renewed bench
evidence is owed at Phase 250 (KVE-01) once the pre-existing lane breakage is
repaired upstream.

## Isolation methodology (for the record)

HEAD-tools-only stash → rebuild → still SIGBUS (tools exonerated); full
src+tools+tests stash → rebuild → still SIGBUS (all phase changes exonerated);
main-checkout May-5 binary → clean (pin regression window).
