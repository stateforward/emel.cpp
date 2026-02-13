# loader

Source: `emel/model/loader/sm.hpp`

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> initialized
  initialized --> mapping_parser : emel::model::loader::event::load [boost::sml::front::always] / boost::sml::front::none
  mapping_parser --> parsing : emel::model::loader::event::mapping_parser_done [boost::sml::front::always] / boost::sml::front::none
  mapping_parser --> errored : emel::model::loader::event::unsupported_format_error [boost::sml::front::always] / boost::sml::front::none
  parsing --> loading_weights : emel::model::parser::events::parsing_done [boost::sml::front::always] / boost::sml::front::none
  parsing --> errored : emel::model::parser::events::parsing_error [boost::sml::front::always] / boost::sml::front::none
  loading_weights --> mapping_layers : emel::model::weight_loader::events::loading_done [boost::sml::front::always] / boost::sml::front::none
  loading_weights --> errored : emel::model::weight_loader::events::loading_error [boost::sml::front::always] / boost::sml::front::none
  mapping_layers --> validating_structure : emel::model::loader::event::layers_mapped [boost::sml::aux::zero_wrapper<emel::model::loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/loader/guards.hpp:5:34)>] / boost::sml::front::none
  mapping_layers --> errored : emel::model::loader::event::layers_mapped [boost::sml::aux::zero_wrapper<emel::model::loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/loader/guards.hpp:7:35)>] / boost::sml::front::none
  validating_structure --> validating_architecture : emel::model::loader::event::structure_validated [boost::sml::aux::zero_wrapper<emel::model::loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/loader/guards.hpp:9:56)>] / boost::sml::front::none
  validating_structure --> done : emel::model::loader::event::structure_validated [boost::sml::aux::zero_wrapper<emel::model::loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/loader/guards.hpp:12:55)>] / boost::sml::front::none
  validating_structure --> errored : emel::model::loader::event::structure_validated [boost::sml::aux::zero_wrapper<emel::model::loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/loader/guards.hpp:7:35)>] / boost::sml::front::none
  validating_architecture --> done : emel::model::loader::event::architecture_validated [boost::sml::aux::zero_wrapper<emel::model::loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/loader/guards.hpp:5:34)>] / boost::sml::front::none
  validating_architecture --> errored : emel::model::loader::event::architecture_validated [boost::sml::aux::zero_wrapper<emel::model::loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/loader/guards.hpp:7:35)>] / boost::sml::front::none
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| `initialized` | `emel::model::loader::event::load` | `boost::sml::front::always` | `boost::sml::front::none` | `mapping_parser` |
| `mapping_parser` | `emel::model::loader::event::mapping_parser_done` | `boost::sml::front::always` | `boost::sml::front::none` | `parsing` |
| `mapping_parser` | `emel::model::loader::event::unsupported_format_error` | `boost::sml::front::always` | `boost::sml::front::none` | `errored` |
| `parsing` | `emel::model::parser::events::parsing_done` | `boost::sml::front::always` | `boost::sml::front::none` | `loading_weights` |
| `parsing` | `emel::model::parser::events::parsing_error` | `boost::sml::front::always` | `boost::sml::front::none` | `errored` |
| `loading_weights` | `emel::model::weight_loader::events::loading_done` | `boost::sml::front::always` | `boost::sml::front::none` | `mapping_layers` |
| `loading_weights` | `emel::model::weight_loader::events::loading_error` | `boost::sml::front::always` | `boost::sml::front::none` | `errored` |
| `mapping_layers` | `emel::model::loader::event::layers_mapped` | `boost::sml::aux::zero_wrapper<emel::model::loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/loader/guards.hpp:5:34)>` | `boost::sml::front::none` | `validating_structure` |
| `mapping_layers` | `emel::model::loader::event::layers_mapped` | `boost::sml::aux::zero_wrapper<emel::model::loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/loader/guards.hpp:7:35)>` | `boost::sml::front::none` | `errored` |
| `validating_structure` | `emel::model::loader::event::structure_validated` | `boost::sml::aux::zero_wrapper<emel::model::loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/loader/guards.hpp:9:56)>` | `boost::sml::front::none` | `validating_architecture` |
| `validating_structure` | `emel::model::loader::event::structure_validated` | `boost::sml::aux::zero_wrapper<emel::model::loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/loader/guards.hpp:12:55)>` | `boost::sml::front::none` | `done` |
| `validating_structure` | `emel::model::loader::event::structure_validated` | `boost::sml::aux::zero_wrapper<emel::model::loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/loader/guards.hpp:7:35)>` | `boost::sml::front::none` | `errored` |
| `validating_architecture` | `emel::model::loader::event::architecture_validated` | `boost::sml::aux::zero_wrapper<emel::model::loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/loader/guards.hpp:5:34)>` | `boost::sml::front::none` | `done` |
| `validating_architecture` | `emel::model::loader::event::architecture_validated` | `boost::sml::aux::zero_wrapper<emel::model::loader::guard::(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/model/loader/guards.hpp:7:35)>` | `boost::sml::front::none` | `errored` |
