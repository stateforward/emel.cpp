# gbnf_rule_parser_nonterm_parser

Source: [`emel/gbnf/rule_parser/nonterm_parser/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> deciding
  deciding --> parsed : completion_parse_rules_ [definition_existing_valid_] / consume_definition_existing_
  deciding --> parsed : completion_parse_rules_ [definition_new_valid_] / consume_definition_new_
  deciding --> parsed : completion_parse_rules_ [reference_existing_valid_] / consume_reference_existing_
  deciding --> parsed : completion_parse_rules_ [reference_new_valid_] / consume_reference_new_
  deciding --> parse_failed : completion_parse_rules_ [parse_failed_] / dispatch_parse_failed_
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
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`completion<parse_rules>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`definition_existing_valid>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`consume_definition_existing>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`parsed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`completion<parse_rules>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`definition_new_valid>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`consume_definition_new>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`parsed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`completion<parse_rules>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`reference_existing_valid>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`consume_reference_existing>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`parsed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`completion<parse_rules>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`reference_new_valid>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`consume_reference_new>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`parsed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`completion<parse_rules>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`parse_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`dispatch_parse_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) |
| [`parsed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`terminate`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) |
| [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`terminate`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) |
| [`parsed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) |
| [`parse_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) |
| [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp) |
