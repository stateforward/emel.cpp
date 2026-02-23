# text_tokenizer_preprocessor

Source: [`emel/text/tokenizer/preprocessor/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/tokenizer/preprocessor/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  direction TB
  [*] --> idle
  idle --> idle : preprocess [always] / none
  idle --> idle : _ [always] / none
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`idle`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/tokenizer/preprocessor/sm.hpp) | [`preprocess`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/tokenizer/preprocessor/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/tokenizer/preprocessor/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/tokenizer/preprocessor/sm.hpp) | [`idle`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/tokenizer/preprocessor/sm.hpp) |
| [`idle`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/tokenizer/preprocessor/sm.hpp) | [`_`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/tokenizer/preprocessor/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/tokenizer/preprocessor/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/tokenizer/preprocessor/sm.hpp) | [`idle`](https://github.com/stateforward/emel.cpp/blob/main/src/emel/text/tokenizer/preprocessor/sm.hpp) |
