# text_jinja_parser_lexer

Source: [`emel/text/jinja/parser/lexer/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> deciding
  deciding --> tokenize_exec : completion_parse_runtime_ [always] / run_tokenize_
  tokenize_exec --> tokenize_result_decision : completion_parse_runtime_ [always] / none
  tokenize_result_decision --> tokenized : completion_parse_runtime_ [tokenize_succeeded_] / none
  tokenize_result_decision --> parse_failed : completion_parse_runtime_ [tokenize_failed_] / mark_parse_failed_
  tokenized --> terminate : [always] / none
  parse_failed --> terminate : [always] / none
  deciding --> unexpected_event : _ [always] / on_unexpected_
  tokenize_exec --> unexpected_event : _ [always] / on_unexpected_
  tokenize_result_decision --> unexpected_event : _ [always] / on_unexpected_
  tokenized --> unexpected_event : _ [always] / on_unexpected_
  parse_failed --> unexpected_event : _ [always] / on_unexpected_
  unexpected_event --> unexpected_event : _ [always] / on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`completion<parse_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`run_tokenize>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`tokenize_exec`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) |
| [`tokenize_exec`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`completion<parse_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`tokenize_result_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) |
| [`tokenize_result_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`completion<parse_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`tokenize_succeeded>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`tokenized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) |
| [`tokenize_result_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`completion<parse_runtime>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`tokenize_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`mark_parse_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) |
| [`tokenized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`terminate`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) |
| [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`terminate`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) |
| [`tokenize_exec`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) |
| [`tokenize_result_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) |
| [`tokenized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) |
| [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) |
| [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/jinja/parser/lexer/sm.hpp) |
