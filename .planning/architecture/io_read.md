# io_read

Source: [`emel/io/read/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> state_ready
  state_ready --> state_request_decision : read_tensor_runtime [always] / effect_begin_read_tensor_
  state_request_decision --> state_unsupported_platform_error_decision : completion_read_tensor_runtime_ [guard_request_accepted_] / effect_mark_unsupported_platform_
  state_unsupported_platform_error_decision --> state_error_callback : completion_read_tensor_runtime_ [error_callback_present_] / effect_publish_read_tensor_error_
  state_unsupported_platform_error_decision --> state_ready : completion_read_tensor_runtime_ [error_callback_absent_] / effect_record_read_tensor_error_
  state_error_callback --> state_ready : completion_read_tensor_runtime_ [always] / effect_record_read_tensor_error_
  state_ready --> state_ready : _ [always] / effect_on_unexpected_
  state_request_decision --> state_ready : _ [always] / effect_on_unexpected_
  state_unsupported_platform_error_decision --> state_ready : _ [always] / effect_on_unexpected_
  state_error_callback --> state_ready : _ [always] / effect_on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`read_tensor_runtime`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`effect_begin_read_tensor>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`state_request_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) |
| [`state_request_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`completion<read_tensor_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`guard_request_accepted>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`effect_mark_unsupported_platform>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`state_unsupported_platform_error_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) |
| [`state_unsupported_platform_error_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`completion<read_tensor_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`error_callback_present>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`effect_publish_read_tensor_error>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`state_error_callback`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) |
| [`state_unsupported_platform_error_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`completion<read_tensor_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`error_callback_absent>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`effect_record_read_tensor_error>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) |
| [`state_error_callback`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`completion<read_tensor_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`effect_record_read_tensor_error>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`effect_on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) |
| [`state_request_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`effect_on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) |
| [`state_unsupported_platform_error_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`effect_on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) |
| [`state_error_callback`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`effect_on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/io/read/sm.hpp) |
