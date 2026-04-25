# text_formatter

Source: [`emel/text/formatter/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/formatter/sm.hpp)

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
| [`idle`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/formatter/sm.hpp) | [`scaffold`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/formatter/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/formatter/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/formatter/sm.hpp) | [`idle`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/formatter/sm.hpp) |
| [`idle`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/formatter/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/formatter/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/formatter/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/formatter/sm.hpp) | [`idle`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/formatter/sm.hpp) |
