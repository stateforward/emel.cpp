---
phase: 65-unified-compare-workflow-and-publication
plan: 01
status: complete
completed: 2026-04-17
requirements-completed:
  - CMP-01
  - CMP-02
---

# Phase 65 Summary

## Outcome

Phase 65 is complete. The repo now has one maintained compare workflow for EMEL-vs-reference
embedding comparisons, regardless of whether the selected reference backend is Python or C++.

## Delivered

- Added `tools/bench/embedding_compare.py` as the unified compare driver.
- Added `scripts/bench_embedding_compare.sh` as the operator-facing workflow entrypoint.
- Added `embedding_compare_tests` and wired it into `tools/bench/CMakeLists.txt`.
- Published raw compare artifacts under `build/embedding_compare/`:
  - raw JSONL records per lane
  - dumped binary output vectors
  - machine-readable `compare_summary.json`

## Published Compare Evidence

- Python parity backend `te_python_goldens` produced maintained full-modality TE compare evidence:
  - text cosine `0.999613`
  - image cosine `0.999979`
  - audio cosine `0.989635`
- C++ baseline backend `liquid_cpp` ran through the same workflow and was truthfully marked
  `non_parity_backend` for similarity.

## Gate Result

- `scripts/quality_gates.sh` exited `0`.
- The gate still emitted the existing warning-only benchmark snapshot note:
  - `text/encoders/bpe_long`
  - `logits/sampler_sml/vocab_256000`
  - `logits/sampler_sml/vocab_128000`
