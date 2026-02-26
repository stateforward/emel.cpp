# graph_assembler_assemble_validate_pass

Source: [`emel/graph/assembler/assemble_validate_pass/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> deciding
  deciding --> assembled : completion [phase_done_] / mark_done_
  deciding --> assemble_failed : completion [phase_invalid_request_] / mark_failed_invalid_request_
  assembled --> terminate : [always] / none
  assemble_failed --> terminate : [always] / none
  deciding --> unexpected_event : _ [always] / on_unexpected_
  assembled --> unexpected_event : _ [always] / on_unexpected_
  assemble_failed --> unexpected_event : _ [always] / on_unexpected_
  unexpected_event --> unexpected_event : _ [always] / on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`completion`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`phase_done>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`mark_done>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`assembled`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`completion`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`phase_invalid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`mark_failed_invalid_request>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`assemble_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) |
| [`assembled`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`terminate`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) |
| [`assemble_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | - | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`terminate`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) |
| [`deciding`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) |
| [`assembled`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) |
| [`assemble_failed`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) |
| [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) | [`unexpected_event`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/graph/assembler/assemble_validate_pass/sm.hpp) |
