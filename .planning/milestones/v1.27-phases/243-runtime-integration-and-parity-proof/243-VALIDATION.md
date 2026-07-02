---
phase: 243
slug: runtime-integration-and-parity-proof
status: passed
nyquist_compliant: true
wave_0_complete: true
created: 2026-06-25
---

# Phase 243 - Validation Strategy

## Test Infrastructure

| Property | Value |
|----------|-------|
| Framework | doctest, paritychecker, source scans, lint snapshot, quality gate |
| Config file | `CMakeLists.txt`; `scripts/quality_gates.sh` |
| Quick run command | `build/phase239/emel_tests_bin --test-case='generator_generate_quantized_contract_fixture_preserves_zero_disallowed_fallback' --no-skipped-summary` |
| Gate command | `EMEL_QUALITY_GATES_CHANGED_FILES="<phase 243 files>" EMEL_QUALITY_GATES_BENCH_SUITE=kernel_x86_64 scripts/quality_gates.sh` |
| Current gate status | passed after approved benchmark and generation baseline snapshots |

## Per-Task Verification Map

| Task ID | Requirement | Test Type | Automated Command | Status |
|---------|-------------|-----------|-------------------|--------|
| 243-01-01 | XRT-01, XRT-03 | generator-chain route proof | focused quantized-contract generator doctest | green |
| 243-01-02 | XRT-02, XRT-03 | paritychecker attribution proof | paritychecker tests and live reference generation parity | green, publication baselines stale |
| 243-01-03 | quality gate | scoped quality gate | `EMEL_QUALITY_GATES_BENCH_SUITE=kernel_x86_64 scripts/quality_gates.sh` | green |

## Command Results

```bash
cmake --build build/phase239 --target emel_tests_bin -j2
```

Result: PASS.

```bash
cmake --build build/paritychecker_zig --target paritychecker paritychecker_tests -j2
```

Result: PASS.

```bash
build/phase239/emel_tests_bin --test-case='generator_generate_quantized_contract_fixture_preserves_zero_disallowed_fallback' --no-skipped-summary
```

Result: PASS. On x86_64 the maintained quantized-contract fixture reports positive
optimized q2/q3/q6 dispatch counters and zero shared q2/q3/q6 counters.

```bash
build/paritychecker_zig/paritychecker_tests
```

Result: PASS.

```bash
build/paritychecker_zig/paritychecker_tests --test-case="paritychecker matches current maintained generation publication against live reference" --no-skipped-summary
```

Result: PASS.

```bash
build/paritychecker_zig/paritychecker --generation --model tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf --text hello --max-tokens=1
build/paritychecker_zig/paritychecker --generation --model tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf --text hello --max-tokens=10
build/paritychecker_zig/paritychecker --generation --model tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf --text hello --max-tokens=100
build/paritychecker_zig/paritychecker --generation --model tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf --text hello --max-tokens=1000
```

Result: live EMEL/reference generation matched for all four token counts. The
`--max-tokens=1` run also matches the checked-in baseline. The `10`, `100`, and
`1000` token runs exit nonzero only because their checked-in generation baselines
still contain the previous stale publication text.

```bash
scripts/check_domain_boundaries.sh
```

Result: PASS.

```bash
rg -n -- '-mavx512|-mavxvnni|-mamx|-mavx512bf16|-mavx512fp16|-mavx512vnni|-mavx512f' CMakeLists.txt src tests tools/paritychecker
```

Result: PASS, no unsupported x86 feature claims or compile flags found. `rg`
returns exit 1 for this no-match scan.

```bash
PATH="/shared/stateforward/.tools/clang-format-venv/bin:/shared/stateforward/.tools/llvm18/root/usr/lib/llvm-18/bin:/shared/stateforward/.tools/git-lfs/git-lfs-3.7.1:$PATH" \
scripts/lint_snapshot.sh
```

Result: PASS. No lint snapshot update was made.

```bash
git diff --check
```

Result: PASS.

```bash
PATH="/shared/stateforward/.tools/clang-format-venv/bin:/shared/stateforward/.tools/llvm18/root/usr/lib/llvm-18/bin:/shared/stateforward/.tools/git-lfs/git-lfs-3.7.1:$PATH" \
EMEL_QUALITY_GATES_BENCH_SUITE="kernel_x86_64" \
EMEL_QUALITY_GATES_CHANGED_FILES="src/emel/model/data.hpp,src/emel/model/gemma4/detail.cpp,src/emel/model/lfm2/detail.cpp,src/emel/model/qwen3/detail.cpp,src/emel/text/generator/detail.hpp,tests/model/loader/lifecycle_tests.cpp,tests/text/generator/detail_tests.cpp,tests/text/generator/lifecycle_tests.cpp,tools/paritychecker/parity_engines.cpp,tools/paritychecker/paritychecker_tests.cpp" \
scripts/quality_gates.sh
```

Initial result before snapshot approval: the scoped gate passed all
non-benchmark lanes:

- `build`: PASS.
- `test_with_coverage`: PASS. Changed-line coverage is `715/744` lines (`96.1%`)
  and `171/240` branches (`71.2%`).
- `paritychecker`: PASS.
- `fuzz_smoke`: skipped because no fuzz-affecting files changed.
- `lint_snapshot`: PASS without snapshot update.
- `generate_docs`: PASS.

The only failing lane was `bench_snapshot`: the `kernel_x86_64` suite still emits
15 `kernel/x86_64/*` entries without approved baselines in
`snapshots/bench/benchmarks.txt`.

## Validation Sign-Off

- [x] Generator-chain x86_64 optimized q2/q3/q6 dispatch has automated validation.
- [x] Paritychecker x86_64 quantized attribution has automated validation.
- [x] Live EMEL/reference generation matches for `1`, `10`, `100`, and `1000`
  token runs.
- [x] Domain-boundary and unsupported x86 feature scans pass.
- [x] Coverage, paritychecker, lint, fuzz routing, and docs lanes pass in scoped gate.
- [x] Snapshot updates were explicitly approved and applied.
- [x] Scoped quality gate passes after approved benchmark baseline update.
- [x] Maintained generation publication baselines are updated after explicit approval.
- [x] `nyquist_compliant: true` and `wave_0_complete: true` are set in
  frontmatter.
- [x] Rule-compliance evidence is recorded through public generator dispatch,
  domain-boundary checks, unsupported feature scans, paritychecker attribution,
  and lint.

**Approval:** granted by user; snapshots updated.
