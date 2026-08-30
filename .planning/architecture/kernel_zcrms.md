# kernel_zcrms

Source: [`emel/kernel/zcrms/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/zcrms/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> state_ready
  state_ready --> state_ready : execute_norm_rows [guard_execute_norm_rows_] / effect_execute_norm_rows_
  state_ready --> state_ready : execute_unit_rows [guard_execute_unit_rows_] / effect_execute_unit_rows_
  state_ready --> state_ready : _ [always] / effect_on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/zcrms/sm.hpp) | [`execute_norm_rows`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/zcrms/sm.hpp) | [`guard_execute_norm_rows>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/zcrms/sm.hpp) | [`effect_execute_norm_rows>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/zcrms/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/zcrms/sm.hpp) |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/zcrms/sm.hpp) | [`execute_unit_rows`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/zcrms/sm.hpp) | [`guard_execute_unit_rows>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/zcrms/sm.hpp) | [`effect_execute_unit_rows>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/zcrms/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/zcrms/sm.hpp) |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/zcrms/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/zcrms/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/zcrms/sm.hpp) | [`effect_on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/zcrms/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/zcrms/sm.hpp) |
