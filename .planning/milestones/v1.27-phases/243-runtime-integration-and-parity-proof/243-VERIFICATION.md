# Phase 243 Verification

status: passed

All commands were run from:
`/shared/stateforward/emel.cpp`

## Must-Have Verification

| Must-have | Evidence | Status |
|-----------|----------|--------|
| Shipped generator chain selects x86_64 optimized q2/q3/q6 paths | `tests/text/generator/lifecycle_tests.cpp` requires positive optimized q2/q3/q6 counters and zero shared q2/q3/q6 counters on the maintained quantized-contract fixture | PASS |
| Runtime proof uses public machine dispatch and diagnostics | Generator lifecycle tests drive `process_event(...)` and `capture_diagnostics`; paritychecker reads maintained generator diagnostics instead of actor private helpers | PASS |
| Paritychecker publishes x86_64 attribution | `tools/paritychecker/parity_engines.cpp` prints `quantized_dispatch:` counters and requires x86_64 native q2/q3/q6 optimized counters when those native tensor types are present | PASS |
| Maintained generation parity covers 1, 10, 100, and 1000 token runs | Live EMEL/reference generation parity matched at `--max-tokens` 1, 10, 100, and 1000; 10/100/1000 are blocked only by stale checked-in generation baselines | PASS |
| Supported and fallback/no-claim behavior remain deterministic | `tools/paritychecker/paritychecker_tests.cpp` asserts x86 optimized counters are positive only when native tensor types are present and shared q2/q3/q6 counters stay zero | PASS |
| Required quality gate | approved `kernel_x86_64` benchmark/parity snapshots landed; scoped quality gate passed all selected lanes | PASS |

## Evidence Summary

- `cmake --build build/phase239 --target emel_tests_bin -j2` passes.
- `cmake --build build/paritychecker_zig --target paritychecker paritychecker_tests -j2`
  passes.
- Focused generator/model tests pass:
  `generator_generate_quantized_contract_fixture_preserves_zero_disallowed_fallback`,
  `generator_generate_runs_native_generator_contract`,
  `generator_detail_lfm2_attention_uses_neox_rope_layout`,
  `generator_detail_qwen3_generator_applies_per_head_qk_norm_before_rope`,
  `generator_detail_gemma4_generator_applies_per_head_qk_norm_before_rope`,
  and the Qwen3, Gemma4, and LFM2 model hparam binding tests.
- `build/paritychecker_zig/paritychecker_tests` passes.
- `build/paritychecker_zig/paritychecker_tests --test-case="paritychecker matches current maintained generation publication against live reference" --no-skipped-summary`
  passes.
- Live generation parity:
  - `--max-tokens 1`: EMEL and reference match and the checked-in baseline matches.
  - `--max-tokens 10`: EMEL and reference match; checked-in generation baseline is stale.
  - `--max-tokens 100`: EMEL and reference match; checked-in generation baseline is stale.
  - `--max-tokens 1000`: EMEL and reference match; checked-in generation baseline is stale.
- `scripts/check_domain_boundaries.sh` passes.
- Unsupported x86 flag scan passes:
  `rg -n -- '-mavx512|-mavxvnni|-mamx|-mavx512bf16|-mavx512fp16|-mavx512vnni|-mavx512f' CMakeLists.txt src tests tools/paritychecker`
  returns no matches.
- `scripts/lint_snapshot.sh` passes with the maintained local tool PATH and without
  updating snapshots.
- `git diff --check` passes.
- Scoped `scripts/quality_gates.sh` passes build, coverage, paritychecker, lint
  snapshot, docs generation, and fuzz routing. Coverage evidence from the scoped gate:
  `changed-line coverage: lines 715/744 (96.1%), branches 171/240 (71.2%)`.
- Approved benchmark snapshots now include the `kernel/x86_64/*` benchmark suite
  entries, and maintained LFM2 generation publication baselines are current for
  `10`, `100`, and `1000` token runs.

## Final Verification

User approved snapshot updates. `scripts/bench.sh --snapshot --update
--suite=kernel_x86_64` updated the benchmark baseline, maintained generation
publication baselines were updated, and the changed-file scoped quality gate
passed with `EMEL_QUALITY_GATES_BENCH_SUITE=kernel_x86_64`.

Phase 243 is fully verified for `XRT-01`, `XRT-02`, and `XRT-03`.
