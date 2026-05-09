# Phase 240 Validation

| ID | Plan | Requirement | Claim | Method | Evidence | Result |
|----|------|-------------|-------|--------|----------|--------|
| 240-01-01 | 01 | CO-02 | Project-owned `emel::co_sm` wrapper exists parallel to `emel::sm`. | source review, build | `src/emel/sm.hpp`, Zig/native builds | pass |
| 240-01-02 | 01 | CO-03 | Scheduler policies expose strict FIFO/single-consumer/RTC contracts. | compile-time test | `co_sm_policy_aliases_expose_strict_scheduler_contracts` | pass |
| 240-01-03 | 01 | CO-04 | Default EMEL coroutine allocator has no heap fallback. | unit test | `fixed_coroutine_allocator_has_no_heap_fallback` | pass |
| 240-01-04 | 01 | CO-04 | Immediate FIFO async dispatch does not allocate coroutine frames. | unit test | `co_sm_default_fifo_path_avoids_coroutine_frame_allocation_when_immediate` | pass |
| 240-01-05 | 01 | CO-05 | Inline async dispatch is deterministic through public machine API. | unit test | `co_sm_inline_scheduler_async_dispatch_runs_immediately` | pass |
| 240-01-06 | 01 | TST-01 | Public wrapper tests cover state inspection, sync dispatch normalization, context injection, and async dispatch. | unit tests, ctest | `tests/sm/sm_policy_tests.cpp`, 13/13 ctest shards | pass |
