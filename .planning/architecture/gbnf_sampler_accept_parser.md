# gbnf_sampler_accept_parser

Source: [`emel/gbnf/sampler/accept_parser/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> deciding
  deciding --> parsed : completion_sample_runtime_ [token_accepted_by_grammar_] / consume_accepted_
  deciding --> parsed : completion_sample_runtime_ [token_rejected_by_grammar_] / consume_rejected_
  deciding --> parse_failed : completion_sample_runtime_ [parse_failed_] / dispatch_parse_failed_
  parsed --> terminate : [always] / none
  parse_failed --> terminate : [always] / none
  deciding --> parse_failed : _ [always] / on_unexpected_
  parsed --> parse_failed : _ [always] / on_unexpected_
  parse_failed --> parse_failed : _ [always] / on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`completion<sample_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`token_accepted_by_grammar>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`consume_accepted>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`parsed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`completion<sample_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`token_rejected_by_grammar>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`consume_rejected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`parsed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`completion<sample_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`parse_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`dispatch_parse_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) |
| [`parsed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`terminate`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) |
| [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`terminate`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) |
| [`parsed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) |
| [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) | [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/accept_parser/sm.hpp) |
