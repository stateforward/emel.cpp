# Phase 240 Verification

status: passed

All commands were run from:
`/shared/stateforward/emel.cpp`

## Must-Have Verification

| Must-have | Evidence | Status |
|-----------|----------|--------|
| Supported x86_64 flash route is optimized | `src/emel/kernel/x86_64/guards.hpp`, `sm.hpp`, `actions.hpp`; `tests/kernel/x86_64_tests.cpp` route-counter tests | PASS |
| Optimized helper is EMEL-owned AVX2/FMA/F16C source code | `run_flash_attn_ext_f16kv_one_chunk_avx2_fma_f16c_unchecked` and conversion/dot/axpy helpers in `src/emel/kernel/x86_64/actions.hpp` | PASS |
| Shared fallback/no-claim behavior remains explicit | feature-disabled x86_64 flash test increments shared counter and not optimized counter | PASS |
| Persistent workspace reuse is preserved | x86_64 flash workspace reuse test observes prepared-token and reuse counters through actor accessors | PASS |
| Numeric behavior matches maintained oracle | x86_64 flash tests compare fixture and masked-token output to flash reference helpers | PASS |
| Required quality gate | approved `kernel_x86_64` benchmark/parity snapshots landed; scoped quality gate passed all selected lanes | PASS |

## Evidence Summary

- Failing-first compile proof captured after adding tests:
  `cmake --build build/phase239 --target CMakeFiles/emel_tests_bin.dir/tests/kernel/x86_64_tests.cpp.o`
  failed before implementation on missing x86_64 flash counters/accessors and
  route support.
- `cmake --build build/phase239 --target CMakeFiles/emel_tests_bin.dir/tests/kernel/x86_64_tests.cpp.o`
  passes after implementation.
- `cmake --build build/phase239 --target emel_tests_bin -j2` passes.
- `ctest --test-dir build/phase239 --output-on-failure -R '^emel_tests_kernel_and_graph$'`
  passes.
- Source-only unsupported x86 flag scan passes:
  `rg -n -- '-mavx512|-mavxvnni|-mamx|-mavx512bf16|-mavx512fp16|-mavx512vnni|-mavx512f' CMakeLists.txt src tests tools/paritychecker`
  returns no matches.
- `scripts/lint_snapshot.sh` passes without updating
  `snapshots/lint/clang_format.txt`.
- Scoped `scripts/quality_gates.sh` passes all non-benchmark lanes:
  coverage, paritychecker, fuzz skip, lint snapshot, and docs generation.
- Coverage evidence from the scoped gate:
  `changed-line coverage: lines 381/406 (93.8%), branches 86/124 (69.4%)`.
- Approved benchmark snapshots now include the `kernel/x86_64/*` benchmark
  suite entries.

## Final Verification

User approved snapshot updates. `scripts/bench.sh --snapshot --update
--suite=kernel_x86_64` updated the benchmark baseline, maintained generation
publication baselines were updated, and the changed-file scoped quality gate
passed with `EMEL_QUALITY_GATES_BENCH_SUITE=kernel_x86_64`.

Phase 240 is fully verified for `XFL-01` and `XFL-02`.
