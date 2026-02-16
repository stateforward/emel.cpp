# generator

Source: [`emel/generator/sm.hpp`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp)

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> initialized
  initialized --> tokenizing_prompt : emel__generator__event__generate [boost__sml__front__always] / boost__sml__front__none
  tokenizing_prompt --> prefilling : emel__generator__event__prompt_tokenized_done [boost__sml__front__always] / boost__sml__front__none
  tokenizing_prompt --> errored : emel__generator__event__prompt_tokenized_error [boost__sml__front__always] / boost__sml__front__none
  prefilling --> decoding : emel__generator__event__prefill_done [boost__sml__front__always] / boost__sml__front__none
  prefilling --> errored : emel__generator__event__prefill_error [boost__sml__front__always] / boost__sml__front__none
  decoding --> decoding : emel__generator__event__decode_step_done [boost__sml__aux__zero_wrapper_emel__generator__guard___lambda_at__Users_gabrielwillen_VSCode_stateforward_emel_emel_cpp_src_emel_generator_guards_hpp_8_48__] / boost__sml__front__none
  decoding --> done : emel__generator__event__stop_condition_met [boost__sml__front__always] / boost__sml__front__none
  decoding --> errored : emel__generator__event__decode_step_error [boost__sml__front__always] / boost__sml__front__none
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| [`initialized`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`generate`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`tokenizing_prompt`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) |
| [`tokenizing_prompt`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`prompt_tokenized_done`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`prefilling`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) |
| [`tokenizing_prompt`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`prompt_tokenized_error`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) |
| [`prefilling`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`prefill_done`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`decoding`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) |
| [`prefilling`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`prefill_error`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) |
| [`decoding`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`decode_step_done`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/generator/guards.hpp:8:48)>`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`decoding`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) |
| [`decoding`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`stop_condition_met`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`done`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) |
| [`decoding`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`decode_step_error`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`always`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`none`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) | [`errored`](https://github.com/stateforward/emel.cpp/blob/main/emel/generator/sm.hpp) |
