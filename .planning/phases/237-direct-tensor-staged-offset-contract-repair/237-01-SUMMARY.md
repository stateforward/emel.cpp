---
phase: 237
status: complete
requirements-completed:
  - TNX-01
  - TNX-03
  - TNX-04
  - TST-01
  - TST-02
---

# Phase 237 Summary

## What Changed

- Added a public regression doctest for direct
  `model/tensor::event::request_staged_load` with nonzero `file_offset` and a
  whole-file source buffer.
- Repaired the direct tensor staged-load dispatch to pass an offset-adjusted
  source span and logical window length into `io/staged_read`.
- Strengthened tensor staged-load validation so source coverage must include
  `file_offset + byte_size` before pointer arithmetic occurs.
- Preserved explicit `_done` and `_error` outcomes for direct staged-load success
  and validation failure.

## Requirement Evidence

- `TNX-01`: direct tensor staged loads use public `io/staged_read` events through
  the injected actor and `process_event(...)`.
- `TNX-03`: success is visible through `request_staged_load_done`, resident tensor
  state, and copied `cdef` bytes from a source offset of `2`.
- `TNX-04`: staged-read validation errors remain mapped to
  `request_staged_load_error`.
- `TST-01`: the new nonzero-offset success doctest exercises public dispatch and
  SML state inspection.
- `TST-02`: the direct staged-load validation-error doctest exercises the public
  failure path.

## Validation

- Failing-first reproduction: offset doctest failed before the source repair.
- `cmake --build build --target emel_tests_bin` - pass.
- `./build/emel_tests_bin --test-case="model_tensor_request_staged_load_applies_nonzero_file_offset"` - pass.
- `./build/emel_tests_bin --test-case="model_tensor_request_staged_load_*"` - pass.
- `ctest --test-dir build -R '^emel_tests_model_and_batch$' --output-on-failure` - pass.
- Scoped `scripts/quality_gates.sh` with Phase 237 changed files - pass.

## Closeout Status

Phase 237 is complete. The direct tensor staged nonzero-offset blocker from
`.planning/v1.26-MILESTONE-AUDIT.md` is closed; Phase 238 remains for audit
artifact and probe-reporting cleanup.
