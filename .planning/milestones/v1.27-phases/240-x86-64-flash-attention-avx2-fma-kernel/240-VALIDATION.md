---
phase: 240
slug: x86-64-flash-attention-avx2-fma-kernel
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-25
---

# Phase 240 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | doctest, CTest, source scans, lint snapshot, quality gate |
| Config file | `CMakeLists.txt`; `scripts/quality_gates.sh` |
| Quick run command | `ctest --test-dir build/phase239 --output-on-failure -R '^emel_tests_kernel_and_graph$'` |
| Gate command | `EMEL_QUALITY_GATES_CHANGED_FILES="<phase 240 files>" scripts/quality_gates.sh` |
| Current gate status | passed after approved x86_64 benchmark baseline update |

## Per-Task Verification Map

| Task ID | Requirement | Test Type | Automated Command | Status |
|---------|-------------|-----------|-------------------|--------|
| 240-01-01 | XFL-01, XFL-02 | failing-first compile proof | `cmake --build build/phase239 --target CMakeFiles/emel_tests_bin.dir/tests/kernel/x86_64_tests.cpp.o` before implementation | red captured |
| 240-01-02 | XFL-01 | focused compile/test | x86_64 test object build; `emel_tests_kernel_and_graph` | green |
| 240-01-03 | XFL-01, XFL-02 | public actor route proof | optimized/shared counter tests in `tests/kernel/x86_64_tests.cpp` | green |
| 240-01-04 | quality gate | scoped quality gate | `EMEL_QUALITY_GATES_CHANGED_FILES="..." scripts/quality_gates.sh` | green |

## Command Results

```bash
cmake --build build/phase239 --target CMakeFiles/emel_tests_bin.dir/tests/kernel/x86_64_tests.cpp.o
```

Result: PASS after implementation. The pre-fix run failed on missing x86_64
optimized/shared flash route counters and accessors.

```bash
cmake --build build/phase239 --target emel_tests_bin -j2
```

Result: PASS.

```bash
ctest --test-dir build/phase239 --output-on-failure -R '^emel_tests_kernel_and_graph$'
```

Result: PASS, `100% tests passed`.

```bash
rg -n -- '-mavx512|-mavxvnni|-mamx|-mavx512bf16|-mavx512fp16|-mavx512vnni|-mavx512f' CMakeLists.txt src tests tools/paritychecker
```

Result: PASS, no unsupported x86 compile flags found.

```bash
scripts/lint_snapshot.sh
```

Result: PASS. No lint snapshot update was made.

```bash
PATH="/shared/stateforward/.tools/clang-format-venv/bin:/shared/stateforward/.tools/llvm18/root/usr/lib/llvm-18/bin:/shared/stateforward/.tools/git-lfs/git-lfs-3.7.1:$PATH" \
EMEL_QUALITY_GATES_TIMEOUT="3600s" \
EMEL_QUALITY_GATES_BENCH_SUITE="kernel_x86_64" \
EMEL_QUALITY_GATES_CHANGED_FILES="src/emel/kernel/x86_64/actions.hpp:src/emel/kernel/x86_64/context.hpp:src/emel/kernel/x86_64/guards.hpp:src/emel/kernel/x86_64/sm.hpp:tests/kernel/x86_64_tests.cpp" \
scripts/quality_gates.sh
```

Initial result before snapshot approval: the scoped gate passed all
non-benchmark lanes:

- `test_with_coverage`: PASS. CTest shard `kernel_and_graph` passes.
  Changed-line coverage is `381/406` lines (`93.8%`) and `86/124` branches
  (`69.4%`).
- `paritychecker`: PASS. Kernel parity runner passes.
- `fuzz_smoke`: skipped because no fuzz-affecting files changed.
- `lint_snapshot`: PASS without snapshot update.
- `generate_docs`: PASS.

The only failing lane was `bench_snapshot`: the `kernel_x86_64` suite still emits
15 `kernel/x86_64/*` entries without baselines in
`snapshots/bench/benchmarks.txt`. User approval was granted and the snapshot
baseline was updated.

## Validation Sign-Off

- [x] x86_64 optimized flash route has automated/source-backed validation.
- [x] x86_64 shared fallback/no-claim path has automated validation.
- [x] Focused x86_64 and kernel/graph tests pass.
- [x] Coverage, parity, lint, fuzz routing, and docs lanes pass in scoped gate.
- [x] Snapshot updates were explicitly approved and applied.
- [x] Scoped quality gate passes after approved benchmark baseline update.
- [x] `nyquist_compliant: true` and `wave_0_complete: true` are set in
  frontmatter.
- [x] Rule-compliance evidence is recorded through explicit x86_64
  guards/transitions, unsupported feature scans, focused actor tests, and lint.

**Approval:** granted by user; snapshots updated.
