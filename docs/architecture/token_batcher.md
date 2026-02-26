# token_batcher

Source: [`emel/token/batcher/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> initialized
  initialized --> batching : batch [valid_request_] / begin_batch_
  initialized --> errored : batch [invalid_request_] / reject_invalid_batch_
  batching --> batch_decision : [always] / run_batch_
  batch_decision --> errored : [phase_failed_] / none
  batch_decision --> done : [phase_ok_] / mark_done_
  done --> batching : batch [valid_request_] / begin_batch_
  done --> errored : batch [invalid_request_] / reject_invalid_batch_
  errored --> batching : batch [valid_request_] / begin_batch_
  errored --> errored : batch [invalid_request_] / reject_invalid_batch_
  unexpected --> batching : batch [valid_request_] / begin_batch_
  unexpected --> errored : batch [invalid_request_] / reject_invalid_batch_
  initialized --> unexpected : _ [always] / on_unexpected_
  batching --> unexpected : _ [always] / on_unexpected_
  batch_decision --> unexpected : _ [always] / on_unexpected_
  done --> unexpected : _ [always] / on_unexpected_
  errored --> unexpected : _ [always] / on_unexpected_
  unexpected --> unexpected : _ [always] / on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`batch`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`valid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`begin_batch>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`batching`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`batch`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`invalid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`reject_invalid_batch>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) |
| [`batching`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`run_batch>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`batch_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) |
| [`batch_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | - | [`phase_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) |
| [`batch_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | - | [`phase_ok>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`mark_done>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) |
| [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`batch`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`valid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`begin_batch>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`batching`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) |
| [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`batch`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`invalid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`reject_invalid_batch>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) |
| [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`batch`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`valid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`begin_batch>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`batching`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) |
| [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`batch`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`invalid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`reject_invalid_batch>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) |
| [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`batch`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`valid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`begin_batch>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`batching`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) |
| [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`batch`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`invalid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`reject_invalid_batch>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) |
| [`batching`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) |
| [`batch_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) |
| [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) |
| [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) |
| [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/token/batcher/sm.hpp) |
