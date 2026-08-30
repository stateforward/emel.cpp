# kernel_swa

Source: [`emel/kernel/swa/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> state_ready
  state_ready --> state_ready : execute_attend [guard_execute_attend_] / effect_execute_attend_
  state_ready --> state_ready : execute_cache_write [guard_execute_cache_write_] / effect_execute_cache_write_
  state_ready --> state_ready : execute_gate_mul [guard_execute_gate_mul_] / effect_execute_gate_mul_
  state_ready --> state_ready : execute_residual_gate [guard_execute_residual_gate_] / effect_execute_residual_gate_
  state_ready --> state_ready : _ [always] / effect_on_unexpected_
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`execute_attend`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`guard_execute_attend>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`effect_execute_attend>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`execute_cache_write`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`guard_execute_cache_write>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`effect_execute_cache_write>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`execute_gate_mul`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`guard_execute_gate_mul>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`effect_execute_gate_mul>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`execute_residual_gate`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`guard_execute_residual_gate>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`effect_execute_residual_gate>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) |
| [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`effect_on_unexpected>`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) | [`state_ready`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/swa/sm.hpp) |
