# gbnf_rule_parser_expression_parser

Source: [`emel/gbnf/rule_parser/expression_parser/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> deciding
  deciding --> parsed_identifier : completion_parse_rules_ [token_identifier_] / consume_identifier_
  deciding --> parsed_non_identifier : completion_parse_rules_ [token_non_identifier_] / consume_non_identifier_
  deciding --> parse_failed : completion_parse_rules_ [parse_failed_] / dispatch_parse_failed_
  parsed_identifier --> terminate : [always] / none
  parsed_non_identifier --> terminate : [always] / none
  parse_failed --> terminate : [always] / none
  deciding --> unexpected_event : _ [always] / on_unexpected_
  parsed_identifier --> unexpected_event : _ [always] / on_unexpected_
  parsed_non_identifier --> unexpected_event : _ [always] / on_unexpected_
  parse_failed --> unexpected_event : _ [always] / on_unexpected_
  unexpected_event --> unexpected_event : _ [always] / on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`completion<parse_rules>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`token_identifier>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`consume_identifier>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`parsed_identifier`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`completion<parse_rules>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`token_non_identifier>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`consume_non_identifier>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`parsed_non_identifier`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`completion<parse_rules>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`parse_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`dispatch_parse_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) |
| [`parsed_identifier`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`terminate`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) |
| [`parsed_non_identifier`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`terminate`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) |
| [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`terminate`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) |
| [`parsed_identifier`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) |
| [`parsed_non_identifier`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) |
| [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) |
| [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/expression_parser/sm.hpp) |
