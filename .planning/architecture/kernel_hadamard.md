# kernel_hadamard

Source: [`emel/kernel/hadamard/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/hadamard/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> state_ready
  state_ready --> state_ready : execute_mlp_row [guard_execute_mlp_row_] / effect_execute_mlp_row_
  state_ready --> state_ready : _ [always] / effect_on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/hadamard/sm.hpp) | [`execute_mlp_row`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/hadamard/sm.hpp) | [`guard_execute_mlp_row>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/hadamard/sm.hpp) | [`effect_execute_mlp_row>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/hadamard/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/hadamard/sm.hpp) |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/hadamard/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/hadamard/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/hadamard/sm.hpp) | [`effect_on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/hadamard/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/hadamard/sm.hpp) |
