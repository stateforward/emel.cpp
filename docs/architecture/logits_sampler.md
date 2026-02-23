# logits_sampler

Source: [`emel/logits/sampler/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> initialized
  initialized --> preparing_candidates : sample [always] / begin_sample_
  preparing_candidates --> prepare_decision : [always] / run_prepare_candidates_
  prepare_decision --> errored : [phase_failed_] / none
  prepare_decision --> sampling : [phase_ok_and_has_more_samplers_] / none
  prepare_decision --> selecting_token : [phase_ok_and_no_more_samplers_] / none
  sampling --> sampling_decision : [always] / run_apply_sampling_
  sampling_decision --> errored : [phase_failed_] / none
  sampling_decision --> sampling : [phase_ok_and_has_more_samplers_] / none
  sampling_decision --> selecting_token : [phase_ok_and_no_more_samplers_] / none
  selecting_token --> select_decision : [always] / run_select_token_
  select_decision --> errored : [phase_failed_] / none
  select_decision --> done : [phase_ok_] / none
  done --> initialized : [always] / publish_done_
  errored --> initialized : [always] / publish_error_
  initialized --> errored : _ [always] / on_unexpected_
  preparing_candidates --> errored : _ [always] / on_unexpected_
  prepare_decision --> errored : _ [always] / on_unexpected_
  sampling --> errored : _ [always] / on_unexpected_
  sampling_decision --> errored : _ [always] / on_unexpected_
  selecting_token --> errored : _ [always] / on_unexpected_
  select_decision --> errored : _ [always] / on_unexpected_
  done --> errored : _ [always] / on_unexpected_
  errored --> errored : _ [always] / on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`sample`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`begin_sample>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`preparing_candidates`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`preparing_candidates`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`run_prepare_candidates>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`prepare_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`prepare_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | - | [`phase_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`prepare_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | - | [`phase_ok_and_has_more_samplers>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`sampling`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`prepare_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | - | [`phase_ok_and_no_more_samplers>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`selecting_token`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`sampling`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`run_apply_sampling>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`sampling_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`sampling_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | - | [`phase_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`sampling_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | - | [`phase_ok_and_has_more_samplers>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`sampling`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`sampling_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | - | [`phase_ok_and_no_more_samplers>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`selecting_token`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`selecting_token`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`run_select_token>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`select_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`select_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | - | [`phase_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`select_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | - | [`phase_ok>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`publish_done>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`publish_error>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`preparing_candidates`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`prepare_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`sampling`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`sampling_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`selecting_token`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`select_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
| [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/logits/sampler/sm.hpp) |
