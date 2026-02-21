# jinja_parser

Source: [`emel/jinja/parser/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> initialized
  initialized --> parse_decision : parse [valid_parse_] / run_parse_
  initialized --> errored : parse [invalid_parse_] / reject_invalid_parse_
  done --> parse_decision : parse [valid_parse_] / run_parse_
  done --> errored : parse [invalid_parse_] / reject_invalid_parse_
  errored --> parse_decision : parse [valid_parse_] / run_parse_
  errored --> errored : parse [invalid_parse_] / reject_invalid_parse_
  unexpected --> parse_decision : parse [valid_parse_] / run_parse_
  unexpected --> unexpected : parse [invalid_parse_] / reject_invalid_parse_
  parse_decision --> done : [phase_ok_] / none
  parse_decision --> errored : [phase_failed_] / none
  initialized --> unexpected : _ [always] / on_unexpected_
  parse_decision --> unexpected : _ [always] / on_unexpected_
  done --> unexpected : _ [always] / on_unexpected_
  errored --> unexpected : _ [always] / on_unexpected_
  unexpected --> unexpected : _ [always] / on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`parse`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`valid_parse>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`run_parse>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`parse_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`parse`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`invalid_parse>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`reject_invalid_parse>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) |
| [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`parse`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`valid_parse>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`run_parse>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`parse_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) |
| [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`parse`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`invalid_parse>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`reject_invalid_parse>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) |
| [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`parse`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`valid_parse>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`run_parse>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`parse_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) |
| [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`parse`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`invalid_parse>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`reject_invalid_parse>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) |
| [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`parse`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`valid_parse>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`run_parse>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`parse_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) |
| [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`parse`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`invalid_parse>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`reject_invalid_parse>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) |
| [`parse_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | - | [`phase_ok>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) |
| [`parse_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | - | [`phase_failed>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) |
| [`parse_decision`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) |
| [`done`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) |
| [`errored`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) |
| [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) | [`unexpected`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/jinja/parser/sm.hpp) |
