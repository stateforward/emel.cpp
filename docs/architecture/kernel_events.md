# kernel_events

Source: [`emel/kernel/events/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/events/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> idle
  idle --> idle : scaffold [always] / none
  idle --> idle : _ [always] / none
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`idle`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/events/sm.hpp) | [`scaffold`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/events/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/events/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/events/sm.hpp) | [`idle`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/events/sm.hpp) |
| [`idle`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/events/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/events/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/events/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/events/sm.hpp) | [`idle`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/kernel/events/sm.hpp) |
