# kernel_mhc

Source: [`emel/kernel/mhc/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> state_ready
  state_ready --> state_ready : execute_pre_mix [guard_execute_pre_mix_] / effect_execute_pre_mix_
  state_ready --> state_ready : execute_post_mix [guard_execute_post_mix_] / effect_execute_post_mix_
  state_ready --> state_ready : execute_mean_lanes [guard_execute_mean_lanes_] / effect_execute_mean_lanes_
  state_ready --> state_ready : _ [always] / effect_on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) | [`execute_pre_mix`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) | [`guard_execute_pre_mix>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) | [`effect_execute_pre_mix>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) | [`execute_post_mix`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) | [`guard_execute_post_mix>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) | [`effect_execute_post_mix>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) | [`execute_mean_lanes`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) | [`guard_execute_mean_lanes>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) | [`effect_execute_mean_lanes>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) | [`effect_on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/mhc/sm.hpp) |
