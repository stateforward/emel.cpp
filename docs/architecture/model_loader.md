# model_loader

Source: `emel/model/loader/sm.hpp`

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> emel::model::loader::initialized
  emel::model::loader::initialized --> emel::model::loader::mapping_parser : emel::model::loader::event::load [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::start_map_parser>
  emel::model::loader::mapping_parser --> emel::model::loader::parsing : emel::model::loader::events::mapping_parser_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::parse>
  emel::model::loader::mapping_parser --> emel::model::loader::errored : emel::model::loader::events::mapping_parser_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::dispatch_error>
  emel::model::loader::parsing --> emel::model::loader::loading_weights : emel::model::loader::events::parsing_done [boost::sml::aux::zero_wrapper<emel::model::loader::guard::should_load_weights>] / boost::sml::aux::zero_wrapper<emel::model::loader::action::load_weights>
  emel::model::loader::parsing --> emel::model::loader::validating_structure : emel::model::loader::events::parsing_done [boost::sml::aux::zero_wrapper<emel::model::loader::guard::skip_weights>] / boost::sml::aux::zero_wrapper<emel::model::loader::action::validate_structure>
  emel::model::loader::parsing --> emel::model::loader::errored : emel::model::loader::events::parsing_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::dispatch_error>
  emel::model::loader::loading_weights --> emel::model::loader::mapping_layers : emel::model::loader::events::loading_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::store_and_map_layers>
  emel::model::loader::loading_weights --> emel::model::loader::errored : emel::model::loader::events::loading_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::dispatch_error>
  emel::model::loader::mapping_layers --> emel::model::loader::validating_structure : emel::model::loader::events::layers_mapped [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::validate_structure>
  emel::model::loader::mapping_layers --> emel::model::loader::errored : emel::model::loader::events::layers_map_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::dispatch_error>
  emel::model::loader::validating_structure --> emel::model::loader::validating_architecture : emel::model::loader::events::structure_validated [boost::sml::aux::zero_wrapper<emel::model::loader::guard::has_arch_validate>] / boost::sml::aux::zero_wrapper<emel::model::loader::action::validate_architecture>
  emel::model::loader::validating_structure --> emel::model::loader::done : emel::model::loader::events::structure_validated [boost::sml::aux::zero_wrapper<emel::model::loader::guard::no_arch_validate>] / boost::sml::aux::zero_wrapper<emel::model::loader::action::dispatch_done>
  emel::model::loader::validating_structure --> emel::model::loader::errored : emel::model::loader::events::structure_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::dispatch_error>
  emel::model::loader::validating_architecture --> emel::model::loader::done : emel::model::loader::events::architecture_validated [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::dispatch_done>
  emel::model::loader::validating_architecture --> emel::model::loader::errored : emel::model::loader::events::architecture_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::dispatch_error>
  emel::model::loader::initialized --> emel::model::loader::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>
  emel::model::loader::mapping_parser --> emel::model::loader::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>
  emel::model::loader::parsing --> emel::model::loader::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>
  emel::model::loader::loading_weights --> emel::model::loader::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>
  emel::model::loader::mapping_layers --> emel::model::loader::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>
  emel::model::loader::validating_structure --> emel::model::loader::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>
  emel::model::loader::validating_architecture --> emel::model::loader::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>
  emel::model::loader::done --> emel::model::loader::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>
  emel::model::loader::errored --> emel::model::loader::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| `emel::model::loader::initialized` | `emel::model::loader::event::load` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::start_map_parser>` | `emel::model::loader::mapping_parser` |
| `emel::model::loader::mapping_parser` | `emel::model::loader::events::mapping_parser_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::parse>` | `emel::model::loader::parsing` |
| `emel::model::loader::mapping_parser` | `emel::model::loader::events::mapping_parser_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::dispatch_error>` | `emel::model::loader::errored` |
| `emel::model::loader::parsing` | `emel::model::loader::events::parsing_done` | `boost::sml::aux::zero_wrapper<emel::model::loader::guard::should_load_weights>` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::load_weights>` | `emel::model::loader::loading_weights` |
| `emel::model::loader::parsing` | `emel::model::loader::events::parsing_done` | `boost::sml::aux::zero_wrapper<emel::model::loader::guard::skip_weights>` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::validate_structure>` | `emel::model::loader::validating_structure` |
| `emel::model::loader::parsing` | `emel::model::loader::events::parsing_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::dispatch_error>` | `emel::model::loader::errored` |
| `emel::model::loader::loading_weights` | `emel::model::loader::events::loading_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::store_and_map_layers>` | `emel::model::loader::mapping_layers` |
| `emel::model::loader::loading_weights` | `emel::model::loader::events::loading_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::dispatch_error>` | `emel::model::loader::errored` |
| `emel::model::loader::mapping_layers` | `emel::model::loader::events::layers_mapped` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::validate_structure>` | `emel::model::loader::validating_structure` |
| `emel::model::loader::mapping_layers` | `emel::model::loader::events::layers_map_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::dispatch_error>` | `emel::model::loader::errored` |
| `emel::model::loader::validating_structure` | `emel::model::loader::events::structure_validated` | `boost::sml::aux::zero_wrapper<emel::model::loader::guard::has_arch_validate>` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::validate_architecture>` | `emel::model::loader::validating_architecture` |
| `emel::model::loader::validating_structure` | `emel::model::loader::events::structure_validated` | `boost::sml::aux::zero_wrapper<emel::model::loader::guard::no_arch_validate>` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::dispatch_done>` | `emel::model::loader::done` |
| `emel::model::loader::validating_structure` | `emel::model::loader::events::structure_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::dispatch_error>` | `emel::model::loader::errored` |
| `emel::model::loader::validating_architecture` | `emel::model::loader::events::architecture_validated` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::dispatch_done>` | `emel::model::loader::done` |
| `emel::model::loader::validating_architecture` | `emel::model::loader::events::architecture_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::dispatch_error>` | `emel::model::loader::errored` |
| `emel::model::loader::initialized` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>` | `emel::model::loader::errored` |
| `emel::model::loader::mapping_parser` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>` | `emel::model::loader::errored` |
| `emel::model::loader::parsing` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>` | `emel::model::loader::errored` |
| `emel::model::loader::loading_weights` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>` | `emel::model::loader::errored` |
| `emel::model::loader::mapping_layers` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>` | `emel::model::loader::errored` |
| `emel::model::loader::validating_structure` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>` | `emel::model::loader::errored` |
| `emel::model::loader::validating_architecture` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>` | `emel::model::loader::errored` |
| `emel::model::loader::done` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>` | `emel::model::loader::errored` |
| `emel::model::loader::errored` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::loader::action::on_unexpected>` | `emel::model::loader::errored` |
