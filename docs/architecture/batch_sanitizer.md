# batch_sanitizer

Source: [`emel/batch/sanitizer/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> initialized
  initialized --> sanitizing : sanitize_decode [valid_request_] / begin_sanitize_
  initialized --> errored : sanitize_decode [invalid_request_] / reject_invalid_sanitize_
  sanitizing --> sanitize_decision : [always] / run_sanitize_decode_
  sanitize_decision --> errored : [phase_failed_] / none
  sanitize_decision --> done : [phase_ok_] / mark_done_
  done --> sanitizing : sanitize_decode [valid_request_] / begin_sanitize_
  done --> errored : sanitize_decode [invalid_request_] / reject_invalid_sanitize_
  errored --> sanitizing : sanitize_decode [valid_request_] / begin_sanitize_
  errored --> errored : sanitize_decode [invalid_request_] / reject_invalid_sanitize_
  unexpected --> sanitizing : sanitize_decode [valid_request_] / begin_sanitize_
  unexpected --> errored : sanitize_decode [invalid_request_] / reject_invalid_sanitize_
  initialized --> unexpected : _ [always] / on_unexpected_
  sanitizing --> unexpected : _ [always] / on_unexpected_
  sanitize_decision --> unexpected : _ [always] / on_unexpected_
  done --> unexpected : _ [always] / on_unexpected_
  errored --> unexpected : _ [always] / on_unexpected_
  unexpected --> unexpected : _ [always] / on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`sanitize_decode`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`valid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`begin_sanitize>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`sanitizing`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`sanitize_decode`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`invalid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`reject_invalid_sanitize>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) |
| [`sanitizing`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`run_sanitize_decode>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`sanitize_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) |
| [`sanitize_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | - | [`phase_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) |
| [`sanitize_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | - | [`phase_ok>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`mark_done>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) |
| [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`sanitize_decode`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`valid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`begin_sanitize>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`sanitizing`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) |
| [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`sanitize_decode`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`invalid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`reject_invalid_sanitize>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) |
| [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`sanitize_decode`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`valid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`begin_sanitize>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`sanitizing`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) |
| [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`sanitize_decode`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`invalid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`reject_invalid_sanitize>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) |
| [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`sanitize_decode`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`valid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`begin_sanitize>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`sanitizing`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) |
| [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`sanitize_decode`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`invalid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`reject_invalid_sanitize>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) |
| [`sanitizing`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) |
| [`sanitize_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) |
| [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) |
| [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) |
| [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/batch/sanitizer/sm.hpp) |
