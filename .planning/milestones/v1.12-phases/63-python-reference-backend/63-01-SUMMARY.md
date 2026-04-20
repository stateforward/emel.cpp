---
phase: 63-python-reference-backend
plan: 01
status: complete
completed: 2026-04-17
requirements-completed:
  - PY-01
  - PY-02
---

# Phase 63 Summary

## Outcome

Phase 63 is complete. The compare architecture now has maintained Python backends: a deterministic
stored-golden TE backend for real compare runs and an optional live TE backend that reports
explicit errors when its Python environment is not ready.

## Delivered

- Added `tools/bench/embedding_reference_python.py` as the shared Python reference runner.
- Added `te_python_goldens` and `te_python_live` backend manifests under
  `tools/bench/reference_backends/`.
- Documented that the maintained Python golden backend consumes the stored upstream TE vectors in
  `tests/embeddings/fixtures/te75m/README.md`.

## Maintained Python Truth

- `te_python_goldens` is the maintained backend for deterministic TE parity comparisons.
- `te_python_live` is optional and emits explicit error records if its live Python generator stack
  cannot run.
- Python backends emit the same schema and vector-dump shape as the EMEL and C++ lanes.
