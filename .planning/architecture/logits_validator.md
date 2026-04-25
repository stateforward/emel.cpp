# logits_validator

Source: [`emel/logits/validator/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> ready
  ready --> request_decision : build_runtime [always] / begin_build_
  request_decision --> done : completion_build_runtime_ [valid_request_] / execute_build_
  request_decision --> errored : completion_build_runtime_ [invalid_request_] / mark_invalid_request_
  done --> ready : completion_build_runtime_ [always] / publish_done_
  errored --> ready : completion_build_runtime_ [always] / publish_error_
  ready --> ready : _ [always] / on_unexpected_
  request_decision --> ready : _ [always] / on_unexpected_
  done --> ready : _ [always] / on_unexpected_
  errored --> ready : _ [always] / on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`build_runtime`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`begin_build>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`request_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) |
| [`request_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`completion<build_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`valid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`execute_build>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) |
| [`request_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`completion<build_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`invalid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`mark_invalid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) |
| [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`completion<build_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`publish_done>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) |
| [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`completion<build_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`publish_error>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) |
| [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) |
| [`request_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) |
| [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) |
| [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) | [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/validator/sm.hpp) |
