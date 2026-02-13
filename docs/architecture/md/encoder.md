# encoder

Source: `emel/encoder/sm.hpp`

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> initialized
  initialized --> pretokenizing : emel::encoder::event::encode [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:5:45)>
  pretokenizing --> selecting_algorithm : emel::encoder::event::pretokenizing_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:6:47)>
  pretokenizing --> errored : emel::encoder::event::pretokenizing_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:7:48)>
  selecting_algorithm --> merging : emel::encoder::event::algorithm_selected [boost::sml::aux::zero_wrapper<emel::encoder::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/guards.hpp:9:37)>] / boost::sml::front::none
  selecting_algorithm --> searching : emel::encoder::event::algorithm_selected [boost::sml::aux::zero_wrapper<emel::encoder::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/guards.hpp:12:39)>] / boost::sml::front::none
  selecting_algorithm --> scanning : emel::encoder::event::algorithm_selected [boost::sml::aux::zero_wrapper<emel::encoder::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/guards.hpp:15:38)>] / boost::sml::front::none
  merging --> emitting_tokens : emel::encoder::event::algorithm_step_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:8:48)>
  merging --> errored : emel::encoder::event::algorithm_step_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:9:49)>
  searching --> emitting_tokens : emel::encoder::event::algorithm_step_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:8:48)>
  searching --> errored : emel::encoder::event::algorithm_step_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:9:49)>
  scanning --> emitting_tokens : emel::encoder::event::algorithm_step_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:8:48)>
  scanning --> errored : emel::encoder::event::algorithm_step_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:9:49)>
  emitting_tokens --> applying_backend_postrules : emel::encoder::event::emission_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:10:42)>
  emitting_tokens --> errored : emel::encoder::event::emission_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:11:43)>
  applying_backend_postrules --> done : emel::encoder::event::postrules_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:12:43)>
  applying_backend_postrules --> errored : emel::encoder::event::postrules_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:13:44)>
  done --> done : emel::encoder::event::tokenized_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:14:57)>
  errored --> errored : emel::encoder::event::tokenized_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:15:58)>
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| `initialized` | `emel::encoder::event::encode` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:5:45)>` | `pretokenizing` |
| `pretokenizing` | `emel::encoder::event::pretokenizing_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:6:47)>` | `selecting_algorithm` |
| `pretokenizing` | `emel::encoder::event::pretokenizing_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:7:48)>` | `errored` |
| `selecting_algorithm` | `emel::encoder::event::algorithm_selected` | `boost::sml::aux::zero_wrapper<emel::encoder::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/guards.hpp:9:37)>` | `boost::sml::front::none` | `merging` |
| `selecting_algorithm` | `emel::encoder::event::algorithm_selected` | `boost::sml::aux::zero_wrapper<emel::encoder::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/guards.hpp:12:39)>` | `boost::sml::front::none` | `searching` |
| `selecting_algorithm` | `emel::encoder::event::algorithm_selected` | `boost::sml::aux::zero_wrapper<emel::encoder::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/guards.hpp:15:38)>` | `boost::sml::front::none` | `scanning` |
| `merging` | `emel::encoder::event::algorithm_step_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:8:48)>` | `emitting_tokens` |
| `merging` | `emel::encoder::event::algorithm_step_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:9:49)>` | `errored` |
| `searching` | `emel::encoder::event::algorithm_step_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:8:48)>` | `emitting_tokens` |
| `searching` | `emel::encoder::event::algorithm_step_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:9:49)>` | `errored` |
| `scanning` | `emel::encoder::event::algorithm_step_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:8:48)>` | `emitting_tokens` |
| `scanning` | `emel::encoder::event::algorithm_step_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:9:49)>` | `errored` |
| `emitting_tokens` | `emel::encoder::event::emission_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:10:42)>` | `applying_backend_postrules` |
| `emitting_tokens` | `emel::encoder::event::emission_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:11:43)>` | `errored` |
| `applying_backend_postrules` | `emel::encoder::event::postrules_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:12:43)>` | `done` |
| `applying_backend_postrules` | `emel::encoder::event::postrules_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:13:44)>` | `errored` |
| `done` | `emel::encoder::event::tokenized_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:14:57)>` | `done` |
| `errored` | `emel::encoder::event::tokenized_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::encoder::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/encoder/actions.hpp:15:58)>` | `errored` |
