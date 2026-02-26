# gbnf_sampler_matcher_parser

Source: [`emel/gbnf/sampler/matcher_parser/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> deciding
  deciding --> parsed : completion [token_text_] / consume_match_accepted_
  deciding --> parsed : completion [token_empty_] / consume_match_rejected_
  deciding --> parse_failed : completion [parse_failed_] / dispatch_parse_failed_
  parsed --> terminate : [always] / none
  parse_failed --> terminate : [always] / none
  deciding --> unexpected_event : _ [always] / on_unexpected_
  parsed --> unexpected_event : _ [always] / on_unexpected_
  parse_failed --> unexpected_event : _ [always] / on_unexpected_
  unexpected_event --> unexpected_event : _ [always] / on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`completion`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`token_text>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`consume_match_accepted>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`parsed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`completion`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`token_empty>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`consume_match_rejected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`parsed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`completion`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`parse_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`dispatch_parse_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) |
| [`parsed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`terminate`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) |
| [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`terminate`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) |
| [`parsed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) |
| [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) |
| [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/sampler/matcher_parser/sm.hpp) |
