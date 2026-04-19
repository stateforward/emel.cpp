---
phase: 66-repair-unified-compare-publication
status: complete
verified: 2026-04-18T00:40:14Z
---

# Phase 66 Verification

## Commands

- `cmake --build build/bench_tools_ninja --parallel --target embedding_compare_tests`
- `ctest --test-dir build/bench_tools_ninja --output-on-failure -R '^embedding_compare_tests$'`
- `python3 tools/bench/embedding_compare.py --reference-backend liquid_cpp --emel-runner build/bench_tools_ninja/embedding_generator_bench_runner --output-dir build/embedding_compare/liquid_cpp_compare_text_repaired --case-filter text`

## Requirements

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| `CPP-01` | `66-01-PLAN.md` | Existing C++ reference lanes can run through the same pluggable backend contract and produce the canonical comparison output. | passed | Repaired `compare_summary.json` preserves both `cpp.reference.arctic_s` and `cpp.reference.embeddinggemma_300m` records for the maintained text run. |
| `CMP-01` | `66-01-PLAN.md` | Operator can run one consistent EMEL-vs-reference compare workflow regardless of backend language. | passed | The maintained `liquid_cpp` compare command exits `0` and now prints two truthful baseline lines for the shared text compare group. |
| `CMP-02` | `66-01-PLAN.md` | Compare artifacts publish backend identity, fixture identity, and enough similarity and configuration metadata to reproduce results across inference engines. | passed | `build/embedding_compare/liquid_cpp_compare_text_repaired/compare_summary.json` records both baseline backend identities and their distinct fixture metadata. |

## Results

- The new duplicate-group regression test fails on the pre-fix driver and now passes on the
  repaired implementation.
- `embedding_compare.py` no longer collapses repeated reference records down to the first match in
  a shared `compare_group`.
- The repaired maintained C++ compare artifact preserves both baseline records:
  - `cpp.reference.arctic_s`
  - `cpp.reference.embeddinggemma_300m`
- The repaired stdout output shows two truthful `non_parity_backend` lines for
  `text/red_square/full_dim`.
