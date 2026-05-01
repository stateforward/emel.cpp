---
phase: 144
plan: 01
status: complete
requirements-completed: []
superseded-by: 146
verification: superseded_by_phase_146
validation: superseded_by_phase_146
---

# Phase 144 Summary: Text Generator Runtime Route Ownership Closure

## Implemented

- Replaced generic decode `run_kernel_flash` / `run_kernel_nonflash` action
  bindings with explicit route-specific decode actions.
- Replaced scalar prefill generic kernel bindings with explicit scalar
  `packed_q8_0`, `q8_k`, `native_quantized`, and `kernel` routes.
- Removed the generic prefill-vs-decode `run_kernel_mode` dispatch from
  `detail.hpp`; each public graph kernel wrapper now validates one expected
  `step_kind`.
- Split scalar matmul execution so the fixed kernel fallback no longer chooses
  packed/q8 routes internally.
- Added SML guard and transition rows for materialized logits and preselected
  argmax route ownership in parent generator and prefill machines.
- Added source regressions in `tests/text/generator/lifecycle_tests.cpp`.

## Notes

The maintained quantized fixture requires a mixed `native_quantized` route:
some matrices use q8-k RHS kernels and others use the native kernel path. The
route is now selected explicitly by guards and transition rows instead of being
hidden behind the old generic action binding.

## Superseded Closeout

Phase 144 closed the first generic route ownership gap but did not fully close
`TEXTGEN-04` / `TEXTGEN-07` alone. Phase 145 and Phase 146 are the follow-up
closure phases. Phase 146 provides the final source-backed compute outcome
modeling validation and is the phase mapped to the completed requirements.
