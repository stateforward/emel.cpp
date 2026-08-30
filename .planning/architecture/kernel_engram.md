# kernel_engram

Source: [`emel/kernel/engram/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> state_ready
  state_ready --> state_ready : execute_hash_rows [guard_execute_hash_rows_] / effect_execute_hash_rows_
  state_ready --> state_ready : execute_conv_taps [guard_execute_conv_taps_] / effect_execute_conv_taps_
  state_ready --> state_ready : execute_alpha_gate [guard_execute_alpha_gate_] / effect_execute_alpha_gate_
  state_ready --> state_ready : _ [always] / effect_on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) | [`execute_hash_rows`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) | [`guard_execute_hash_rows>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) | [`effect_execute_hash_rows>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) | [`execute_conv_taps`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) | [`guard_execute_conv_taps>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) | [`effect_execute_conv_taps>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) | [`execute_alpha_gate`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) | [`guard_execute_alpha_gate>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) | [`effect_execute_alpha_gate>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) | [`effect_on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/engram/sm.hpp) |
