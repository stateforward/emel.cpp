---
phase: 62-reference-backend-contract
status: complete
verified: 2026-04-17T23:35:00Z
---

# Phase 62 Verification

## Commands

- `cmake --build build/bench_tools_ninja --parallel --target embedding_generator_bench_runner`
- `python3 tools/bench/embedding_compare.py --reference-backend te_python_goldens --emel-runner build/bench_tools_ninja/embedding_generator_bench_runner --output-dir build/embedding_compare/te_python_goldens_all`

## Requirements

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| `REF-01` | `62-01-SUMMARY.md` | Operator can select a reference backend for parity or benchmark runs without changing the EMEL lane implementation. | passed | The maintained compare command selects `te_python_goldens` through tooling while keeping the EMEL runner fixed at `embedding_generator_bench_runner`. |
| `REF-02` | `62-01-SUMMARY.md` | Python and C++ reference backends emit one canonical comparison contract for embeddings, timings, and backend metadata. | passed | The Phase 62 verification command consumes EMEL and Python reference JSONL under the shared `embedding_compare/v1` schema. |
| `ISO-01` | `62-01-SUMMARY.md` | The EMEL lane remains isolated from reference-engine model, tokenizer, cache, and runtime objects. | passed | The proof path rebuilds the EMEL runner separately and compares it against a manifest-selected reference lane without sharing lane-owned runtime state. |

## Results

- `embedding_generator_bench_runner` rebuilt successfully with the shared compare-contract wiring.
- The unified compare driver consumed EMEL JSONL output from the maintained runner and a Python
  reference backend under the same schema.
- The contract produced real similarity output on the maintained TE anchors:
  - text cosine: `0.999613`
  - image cosine: `0.999979`
  - audio cosine: `0.989635`
