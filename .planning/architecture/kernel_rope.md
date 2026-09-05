# kernel_rope

Source: [`emel/kernel/rope/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/rope/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> state_ready
  state_ready --> state_ready : execute_precompute [guard_execute_precompute_] / effect_execute_precompute_
  state_ready --> state_ready : execute_apply_rows [guard_execute_apply_rows_] / effect_execute_apply_rows_
  state_ready --> state_ready : _ [always] / effect_on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/rope/sm.hpp) | [`execute_precompute`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/rope/sm.hpp) | [`guard_execute_precompute>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/rope/sm.hpp) | [`effect_execute_precompute>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/rope/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/rope/sm.hpp) |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/rope/sm.hpp) | [`execute_apply_rows`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/rope/sm.hpp) | [`guard_execute_apply_rows>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/rope/sm.hpp) | [`effect_execute_apply_rows>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/rope/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/rope/sm.hpp) |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/rope/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/rope/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/rope/sm.hpp) | [`effect_on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/rope/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/rope/sm.hpp) |
