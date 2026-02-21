# sampler_token_selector

Source: [`emel/sampler/token_selector/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> initialized
  initialized --> validating : select_token [always] / begin_select_token_
  validating --> validate_decision : [always] / run_validate_
  validate_decision --> errored : [phase_failed_] / none
  validate_decision --> selecting_token : [phase_ok_] / none
  selecting_token --> select_decision : [always] / run_select_
  select_decision --> errored : [phase_failed_] / none
  select_decision --> done : [phase_ok_] / none
  done --> initialized : [always] / publish_done_
  errored --> initialized : [always] / publish_error_
  initialized --> errored : _ [always] / on_unexpected_
  validating --> errored : _ [always] / on_unexpected_
  validate_decision --> errored : _ [always] / on_unexpected_
  selecting_token --> errored : _ [always] / on_unexpected_
  select_decision --> errored : _ [always] / on_unexpected_
  done --> errored : _ [always] / on_unexpected_
  errored --> errored : _ [always] / on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`select_token`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`begin_select_token>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`validating`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) |
| [`validating`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`run_validate>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`validate_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) |
| [`validate_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | - | [`phase_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) |
| [`validate_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | - | [`phase_ok>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`selecting_token`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) |
| [`selecting_token`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`run_select>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`select_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) |
| [`select_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | - | [`phase_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) |
| [`select_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | - | [`phase_ok>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) |
| [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`publish_done>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) |
| [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`publish_error>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) |
| [`validating`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) |
| [`validate_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) |
| [`selecting_token`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) |
| [`select_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) |
| [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) |
| [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/sampler/token_selector/sm.hpp) |
