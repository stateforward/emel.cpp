---
phase: 225-read-closeout-runtime-validation-and-sml-repair
reviewed: 2026-05-06T15:57:49Z
depth: standard
files_reviewed: 22
files_reviewed_list:
  - src/emel/io/events.hpp
  - src/emel/io/loader/actions.hpp
  - src/emel/io/loader/detail.hpp
  - src/emel/io/loader/events.hpp
  - src/emel/io/loader/guards.hpp
  - src/emel/io/loader/sm.hpp
  - src/emel/io/read/actions.hpp
  - src/emel/io/read/detail.hpp
  - src/emel/io/read/events.hpp
  - src/emel/io/read/guards.hpp
  - src/emel/io/read/sm.hpp
  - src/emel/model/loader/actions.hpp
  - src/emel/model/loader/events.hpp
  - src/emel/model/loader/guards.hpp
  - src/emel/model/loader/sm.hpp
  - tests/io/loader/lifecycle_tests.cpp
  - tests/io/read/lifecycle_tests.cpp
  - tests/model/loader/lifecycle_tests.cpp
  - tools/bench/diarization/sortformer_fixture.hpp
  - tools/bench/generation_bench.cpp
  - tools/embedded_size/emel_probe/main.cpp
  - tools/paritychecker/parity_engines.cpp
findings:
  critical: 1
  warning: 1
  info: 0
  total: 2
status: issues_found
---

# Phase 225: Code Review Report

**Reviewed:** 2026-05-06T15:57:49Z
**Depth:** standard
**Files Reviewed:** 22
**Status:** issues_found

## Summary

Reviewed the Phase 225 SML repair path, including the public `load_tensor_batch`
and `read_tensor_batch` routes, model-loader `io_load_spans` wiring, maintained
bench/parity callers, and lifecycle tests. The batch dispatch shape matches the
phase intent: model-loader dispatches one public IO loader batch event, callers
provide request-owned `io_load_spans`, and the concrete copy loop stays in
`io/read`.

Two correctness gaps remain: one null-callback crash in the model loader's parse
entry and one missing terminal path for unknown read source errors.

## Critical Issues

### CR-01: Model Loader Can Call An Empty Parse Callback

**File:** `src/emel/model/loader/actions.hpp:230`
**Issue:** `run_parse` unconditionally invokes `ev.request.parse_model(ev.request)`,
but neither `event::load` construction nor the `valid_request` guard verifies that
the required callback is actually bound. `emel::callback::operator()` calls the
stored thunk directly, so an empty `parse_model_fn` causes a null function-pointer
call during dispatch instead of an explicit invalid-request transition.
**Fix:**
```cpp
// guards.hpp
struct parse_model_present {
  bool operator()(const event::load_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.parse_model);
  }
};

struct valid_request {
  bool operator()(const event::load_runtime &ev,
                  const action::context &) const noexcept {
    return parse_model_present{}(ev) &&
           (has_model_path{}(ev) || has_file_image{}(ev));
  }
};
```
Add a failing lifecycle test that passes an empty `parse_model_fn` with an
otherwise valid file image and asserts `process_event` returns false, publishes
`error::invalid_request`, and returns to `ready`.

## Warnings

### WR-01: Unknown Source Errors Strand `io/read` In A Mid-Chain State

**File:** `src/emel/io/read/guards.hpp:316`
**Issue:** `file_open_succeeded` and `file_seek_succeeded` treat any non-`none`
error except their own category as success, but `file_read_failed`,
`file_read_short`, and `file_read_succeeded` only recognize known read outcomes.
If `source_error` carries any other nonzero value, no transition is enabled from
`state_file_read_decision`; the actor stays out of `state_ready` and no error
callback fires. The same gap exists for batch spans via
`batch_span_file_read_failed` / `batch_span_short_read` /
`batch_span_file_read_succeeded`.
**Fix:** Add an explicit unclassified-source-error guard and transition for both
single and batch paths, or broaden the read-failed predicates to catch every
non-`none` source error not handled by open/seek/short-read guards. Cover both
`read_tensor` and `read_tensor_batch` with tests using a synthetic nonzero
`source_error` and asserting a deterministic error callback plus recovery to
`state_ready`.

---

_Reviewed: 2026-05-06T15:57:49Z_
_Reviewer: the agent (gsd-code-reviewer)_
_Depth: standard_
