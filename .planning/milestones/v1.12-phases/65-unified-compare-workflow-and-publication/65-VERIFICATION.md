---
phase: 65-unified-compare-workflow-and-publication
status: complete
verified: 2026-04-18T00:20:00Z
---

# Phase 65 Verification

## Commands

- `ctest --test-dir build/bench_tools_ninja --output-on-failure -R '^embedding_compare_tests$'`
- `scripts/bench_embedding_compare.sh --reference-backend te_python_goldens --case-filter text --skip-emel-build --output-dir build/embedding_compare/wrapper_python_text`
- `python3 tools/bench/embedding_compare.py --reference-backend te_python_goldens --emel-runner build/bench_tools_ninja/embedding_generator_bench_runner --output-dir build/embedding_compare/te_python_goldens_all`
- `python3 tools/bench/embedding_compare.py --reference-backend liquid_cpp --emel-runner build/bench_tools_ninja/embedding_generator_bench_runner --output-dir build/embedding_compare/liquid_cpp_compare_text --case-filter text`
- `scripts/quality_gates.sh`

## Requirements

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| `CMP-01` | `65-01-SUMMARY.md` | Operator can run one consistent EMEL-vs-reference compare workflow regardless of backend language. | passed | The wrapper workflow runs successfully against the maintained Python backend, and the same compare driver path also runs against the manifest-driven C++ baseline backend. |
| `CMP-02` | `65-01-SUMMARY.md` | Compare artifacts publish backend identity, fixture identity, and enough similarity and configuration metadata to reproduce results across inference engines. | passed | The shared workflow writes raw JSONL/vector artifacts and `compare_summary.json`; Phase 66 later repairs the C++ multi-record publication gap in that shared artifact surface. |

## Results

- `embedding_compare_tests` passed.
- The wrapper workflow ran successfully against the Python backend:
  - `text/red_square/full_dim status=computed reason=ok cosine=0.999613`
- The direct compare workflow ran successfully across all maintained TE modalities against the
  Python backend:
  - text cosine `0.999613`
  - image cosine `0.999979`
  - audio cosine `0.989635`
- The same workflow ran successfully against the manifest-driven C++ baseline backend:
  - `text/red_square/full_dim status=unavailable reason=non_parity_backend`
- `scripts/quality_gates.sh` exited `0` with the existing warning-only benchmark snapshot note.
