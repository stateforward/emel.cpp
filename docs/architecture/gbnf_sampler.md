# gbnf_sampler

Source: [`emel/gbnf/sampler/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> ready
  ready --> request_decision : sample_runtime [always] / begin_sample_
  request_decision --> filter_candidates : completion_sample_runtime_ [valid_sample_request_] / none
  request_decision --> errored : completion_sample_runtime_ [invalid_sample_request_] / mark_invalid_request_
  filter_candidates --> finalize_decision : completion_sample_runtime_ [always] / filter_candidates_
  finalize_decision --> done : completion_sample_runtime_ [filtered_candidates_available_] / none
  finalize_decision --> errored : completion_sample_runtime_ [no_filtered_candidates_] / mark_parse_failed_
  done --> ready : completion_sample_runtime_ [always] / publish_done_
  errored --> ready : completion_sample_runtime_ [always] / publish_error_
  ready --> ready : _ [always] / on_unexpected_
  request_decision --> ready : _ [always] / on_unexpected_
  filter_candidates --> ready : _ [always] / on_unexpected_
  finalize_decision --> ready : _ [always] / on_unexpected_
  done --> ready : _ [always] / on_unexpected_
  errored --> ready : _ [always] / on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`sample_runtime`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`begin_sample>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`request_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) |
| [`request_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`completion<sample_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`valid_sample_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`filter_candidates`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) |
| [`request_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`completion<sample_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`invalid_sample_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`mark_invalid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) |
| [`filter_candidates`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`completion<sample_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`filter_candidates>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`finalize_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) |
| [`finalize_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`completion<sample_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`filtered_candidates_available>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) |
| [`finalize_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`completion<sample_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`no_filtered_candidates>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`mark_parse_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) |
| [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`completion<sample_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`publish_done>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) |
| [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`completion<sample_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`publish_error>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) |
| [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) |
| [`request_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) |
| [`filter_candidates`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) |
| [`finalize_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) |
| [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) |
| [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) | [`ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/sm.hpp) |
