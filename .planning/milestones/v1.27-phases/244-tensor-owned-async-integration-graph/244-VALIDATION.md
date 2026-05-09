# Phase 244 Validation

| ID | Plan | Requirement | Claim | Method | Evidence | Result |
|----|------|-------------|-------|--------|----------|--------|
| 244-01-01 | 01 | TNX-01 | `model/tensor` initiates async loading only by constructing and dispatching a public `emel::io::async::event::load_window`. | source review, unit test | `effect_attempt_request_async_load_dispatch`, `model_tensor_request_async_load_progresses_then_commits_residency` | pass |
| 244-01-02 | 01 | TNX-02 | Tensor residency remains owned by `model/tensor`; async I/O only reports progress or terminal result. | source review, state assertion | `effect_commit_request_async_load`, `capture_tensor_state` assertions | pass |
| 244-01-03 | 01 | TNX-03 | Tensor integrators observe partial progress, terminal success, and terminal errors through explicit tensor events. | unit tests | `request_async_load_progress_done`, `request_async_load_done`, `request_async_load_error` callbacks | pass |
| 244-01-04 | 01 | TST-03 | Public dispatch tests cover unsupported actor, partial progress, terminal residency commit, and async failure. | doctest | `*model_tensor_request_async*` focused test run | pass |
