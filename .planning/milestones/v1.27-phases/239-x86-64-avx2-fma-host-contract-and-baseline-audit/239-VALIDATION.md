---
phase: 239
slug: x86-64-avx2-fma-host-contract-and-baseline-audit
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-25
---

# Phase 239 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | doctest, CTest, CMake configure/build, source scans, quality gate |
| Config file | `CMakeLists.txt`; `scripts/quality_gates.sh` |
| Quick run command | `ctest --test-dir build/phase239 --output-on-failure -R '^emel_tests_kernel_and_graph$'` |
| Gate command | `EMEL_QUALITY_GATES_CHANGED_FILES="<phase 239 implementation files>" scripts/quality_gates.sh` |
| Current gate status | passed after approved x86_64 benchmark baseline update |

## Per-Task Verification Map

| Task ID | Requirement | Test Type | Automated Command | Status |
|---------|-------------|-----------|-------------------|--------|
| 239-01-01 | X86-01 | failing-first compile proof | `ninja -C build/phase239 CMakeFiles/emel_tests_bin.dir/tests/kernel/x86_64_tests.cpp.o` before implementation | red captured |
| 239-01-02 | X86-01 | focused compile/test | `ninja -C build/phase239 CMakeFiles/emel_tests_bin.dir/tests/kernel/x86_64_tests.cpp.o`; `ctest --test-dir build/phase239 --output-on-failure -R '^emel_tests_kernel_and_graph$'` | green |
| 239-01-03 | X86-02 | configure/build/source scan | `CC='/shared/zig/zig cc' CXX='/shared/zig/zig c++' cmake -S . -B build/phase239 -G Ninja -DEMEL_ENABLE_TESTS=ON`; `cmake --build build/phase239 --target emel_tests_bin -j2`; unsupported-flag scan | green |
| 239-01-04 | X86-01, X86-02 | artifact/source audit | `239-X86-BASELINE-AUDIT.md`; `git diff --check` | green |
| 239-01-05 | quality gate | scoped quality gate | `EMEL_QUALITY_GATES_CHANGED_FILES="..." scripts/quality_gates.sh` | green |

## Command Results

```bash
CC='/shared/zig/zig cc' CXX='/shared/zig/zig c++' cmake -S . -B build/phase239 -G Ninja -DEMEL_ENABLE_TESTS=ON
```

Result: PASS. Configure reported
`EMEL enabling x86_64 host compile flags: -mavx2;-mfma;-mf16c`.

```bash
ninja -C build/phase239 CMakeFiles/emel_tests_bin.dir/tests/kernel/x86_64_tests.cpp.o
```

Result: PASS after implementation. The pre-fix run failed on missing x86_64
host feature contract/accessors and FMA/F16C detection.

```bash
cmake --build build/phase239 --target emel_tests_bin -j2
```

Result: PASS after repairing x86 host compile portability gaps.

```bash
ctest --test-dir build/phase239 --output-on-failure -R '^emel_tests_kernel_and_graph$'
```

Result: PASS, `100% tests passed`.

```bash
rg -n -- '-mavx512|-mavxvnni|-mamx|-mavx512bf16|-mavx512fp16|-mavx512vnni|-mavx512f' CMakeLists.txt src tests tools/paritychecker .planning/phases/239-x86-64-avx2-fma-host-contract-and-baseline-audit
```

Result: PASS, no unsupported x86 compile flags found.

```bash
scripts/paritychecker.sh --runner=kernel
```

Result: PASS. The paritychecker builds and the kernel parity runner passes.

```bash
PATH="/shared/stateforward/.tools/clang-format-venv/bin:/shared/stateforward/.tools/llvm18/root/usr/lib/llvm-18/bin:/shared/stateforward/.tools/git-lfs/git-lfs-3.7.1:$PATH" \
EMEL_QUALITY_GATES_TIMEOUT="3600s" \
EMEL_QUALITY_GATES_BENCH_SUITE="kernel_x86_64" \
EMEL_QUALITY_GATES_CHANGED_FILES="CMakeLists.txt:src/emel/kernel/x86_64/actions.hpp:src/emel/kernel/x86_64/context.hpp:src/emel/kernel/x86_64/sm.hpp:tests/kernel/x86_64_tests.cpp:src/emel/kernel/aarch64/actions.hpp:src/emel/diarization/sortformer/detail.cpp:src/emel/text/generator/detail.hpp:src/emel/text/generator/context.hpp:src/emel/embeddings/generator/detail.hpp:tests/kernel/aarch64_tests.cpp:tests/kernel/test_helpers.hpp:tests/text/generator/detail_tests.cpp:tests/text/generator/lifecycle_tests.cpp:tests/embeddings/vision_embedding_lane_tests.cpp:tests/embeddings/text_embedding_lane_tests.cpp:tools/paritychecker/CMakeLists.txt:tools/paritychecker/paritychecker_tests.cpp:tools/paritychecker/parity_engines.cpp:tools/bench/CMakeLists.txt:tools/bench/quality_gates_tests.cpp:scripts/test_with_coverage.sh:tests/diarization/request/lifecycle_tests.cpp:tests/diarization/sortformer/encoder/lifecycle_tests.cpp:tests/diarization/sortformer/modules/lifecycle_tests.cpp:tests/diarization/sortformer/output/lifecycle_tests.cpp:tests/diarization/sortformer/transformer/lifecycle_tests.cpp:tests/graph/assembler/assembler_tests.cpp:tests/graph/graph_tests.cpp:tests/model/loader/lifecycle_tests.cpp:tests/embeddings/te_fixture_data.hpp" \
scripts/quality_gates.sh
```

Initial result before snapshot approval: the scoped gate passed all
non-benchmark lanes:

- `test_with_coverage`: PASS. CTest shards
  `generator_and_runtime`, `diarization`, and `kernel_and_graph` pass.
  Changed-line coverage is `73/78` lines (`93.6%`) and `18/34` branches
  (`52.9%`).
- `paritychecker`: PASS. Full paritychecker tests pass.
- `fuzz_smoke`: skipped because no fuzz-affecting files changed.
- `lint_snapshot`: PASS after using the existing `tests/kernel/test_helpers.hpp`
  instead of adding a new helper file.
- `generate_docs`: PASS.

The only failing lane was `bench_snapshot`: the new `kernel_x86_64` suite emits
15 `kernel/x86_64/*` entries, and `snapshots/bench/benchmarks.txt` had no
matching baselines. User approval was granted and the snapshot baseline was
updated.

```bash
EMEL_BENCH_SUITE=kernel_x86_64 \
EMEL_BENCH_ITERS=100 \
EMEL_BENCH_RUNS=3 \
EMEL_BENCH_WARMUP_ITERS=10 \
EMEL_BENCH_WARMUP_RUNS=1 \
build/bench_tools_ninja_kernel_x86_64/bench_runner --mode=emel
```

Result: PASS as a non-mutating measurement. It emits x86_64 benchmark entries
for `op_add`, `op_cos`, `op_div`, `op_dup`, `op_log`, `op_mul`, `op_mul_mat`,
`op_sin`, `op_soft_max`, `op_sqr`, `op_sqrt`, `op_sub`, `op_unary_exp`,
`op_unary_neg`, and `op_unary_relu`.

## Manual-Only Verifications

- Approve the benchmark snapshot baseline update.
- Run `scripts/bench.sh --snapshot --update --suite=kernel_x86_64`.
- Re-run the scoped quality gate above.

## Validation Sign-Off

- [x] Host feature contract has automated/source-backed validation.
- [x] x86_64 host-tuned build config has configure/build/source-scan evidence.
- [x] Focused x86_64 and kernel/graph tests pass.
- [x] Snapshot updates were explicitly approved and applied.
- [x] Scoped quality gate passes after approved benchmark baseline update.
- [x] `nyquist_compliant: true` and `wave_0_complete: true` are set in
  frontmatter.
- [x] Rule-compliance evidence is recorded through Zig configure/build,
  unsupported x86 feature scans, and source-backed host-contract validation.

**Approval:** granted by user; snapshots updated.
