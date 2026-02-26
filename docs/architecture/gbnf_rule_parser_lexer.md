# gbnf_rule_parser_lexer

Source: [`emel/gbnf/rule_parser/lexer/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> initialized
  initialized --> initialized : next [invalid_next_] / reject_invalid_next_
  initialized --> initialized : next [invalid_cursor_position_] / reject_invalid_cursor_
  initialized --> scanning : next [has_remaining_input_] / emit_next_token_
  initialized --> scanning : next [at_eof_] / emit_eof_
  scanning --> scanning : next [invalid_next_] / reject_invalid_next_
  scanning --> scanning : next [invalid_cursor_position_] / reject_invalid_cursor_
  scanning --> scanning : next [has_remaining_input_] / emit_next_token_
  scanning --> scanning : next [at_eof_] / emit_eof_
  initialized --> initialized : _ [always] / on_unexpected_
  scanning --> scanning : _ [always] / on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`next`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`invalid_next>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`reject_invalid_next>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`next`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`invalid_cursor_position>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`reject_invalid_cursor>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`next`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`has_remaining_input>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`emit_next_token>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`scanning`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`next`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`at_eof>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`emit_eof>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`scanning`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) |
| [`scanning`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`next`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`invalid_next>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`reject_invalid_next>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`scanning`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) |
| [`scanning`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`next`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`invalid_cursor_position>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`reject_invalid_cursor>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`scanning`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) |
| [`scanning`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`next`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`has_remaining_input>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`emit_next_token>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`scanning`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) |
| [`scanning`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`next`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`at_eof>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`emit_eof>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`scanning`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) |
| [`scanning`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) | [`scanning`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/gbnf/rule_parser/lexer/sm.hpp) |
