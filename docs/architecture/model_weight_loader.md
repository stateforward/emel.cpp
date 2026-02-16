# model_weight_loader

Source: `emel/model/weight_loader/sm.hpp`

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> initialized
  initialized --> loading_mmap : emel::model::weight_loader::event::load_weights [boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/weight_loader/guards.hpp:7:34)>] / boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/weight_loader/actions.hpp:5:42)>
  initialized --> loading_streamed : emel::model::weight_loader::event::load_weights [boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/weight_loader/guards.hpp:19:38)>] / boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/weight_loader/actions.hpp:6:47)>
  loading_mmap --> done : emel::model::weight_loader::event::weights_loaded [boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/weight_loader/guards.hpp:23:34)>] / boost::sml::front::none
  loading_mmap --> errored : emel::model::weight_loader::event::weights_loaded [boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/weight_loader/guards.hpp:26:35)>] / boost::sml::front::none
  loading_streamed --> done : emel::model::weight_loader::event::weights_loaded [boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/weight_loader/guards.hpp:23:34)>] / boost::sml::front::none
  loading_streamed --> errored : emel::model::weight_loader::event::weights_loaded [boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/weight_loader/guards.hpp:26:35)>] / boost::sml::front::none
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| `initialized` | `emel::model::weight_loader::event::load_weights` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/weight_loader/guards.hpp:7:34)>` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/weight_loader/actions.hpp:5:42)>` | `loading_mmap` |
| `initialized` | `emel::model::weight_loader::event::load_weights` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/weight_loader/guards.hpp:19:38)>` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/weight_loader/actions.hpp:6:47)>` | `loading_streamed` |
| `loading_mmap` | `emel::model::weight_loader::event::weights_loaded` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/weight_loader/guards.hpp:23:34)>` | `boost::sml::front::none` | `done` |
| `loading_mmap` | `emel::model::weight_loader::event::weights_loaded` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/weight_loader/guards.hpp:26:35)>` | `boost::sml::front::none` | `errored` |
| `loading_streamed` | `emel::model::weight_loader::event::weights_loaded` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/weight_loader/guards.hpp:23:34)>` | `boost::sml::front::none` | `done` |
| `loading_streamed` | `emel::model::weight_loader::event::weights_loaded` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/weight_loader/guards.hpp:26:35)>` | `boost::sml::front::none` | `errored` |
