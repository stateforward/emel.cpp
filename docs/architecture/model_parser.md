# model_parser

Source: `emel/model/parser/sm.hpp`

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> emel::model::parser::initialized
  emel::model::parser::initialized --> emel::model::parser::parsing_architecture : emel::model::parser::event::parse_model [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::parse_architecture>
  emel::model::parser::parsing_architecture --> emel::model::parser::mapping_architecture : emel::model::parser::events::parse_architecture_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::map_architecture>
  emel::model::parser::parsing_architecture --> emel::model::parser::errored : emel::model::parser::events::parse_architecture_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::dispatch_error>
  emel::model::parser::mapping_architecture --> emel::model::parser::parsing_hparams : emel::model::parser::events::map_architecture_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::parse_hparams>
  emel::model::parser::mapping_architecture --> emel::model::parser::errored : emel::model::parser::events::map_architecture_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::dispatch_error>
  emel::model::parser::parsing_hparams --> emel::model::parser::parsing_vocab : emel::model::parser::events::parse_hparams_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::parse_vocab>
  emel::model::parser::parsing_hparams --> emel::model::parser::errored : emel::model::parser::events::parse_hparams_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::dispatch_error>
  emel::model::parser::parsing_vocab --> emel::model::parser::mapping_tensors : emel::model::parser::events::parse_vocab_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::map_tensors>
  emel::model::parser::parsing_vocab --> emel::model::parser::errored : emel::model::parser::events::parse_vocab_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::dispatch_error>
  emel::model::parser::mapping_tensors --> emel::model::parser::done : emel::model::parser::events::map_tensors_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::dispatch_done>
  emel::model::parser::mapping_tensors --> emel::model::parser::errored : emel::model::parser::events::map_tensors_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::dispatch_error>
  emel::model::parser::initialized --> emel::model::parser::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::on_unexpected>
  emel::model::parser::parsing_architecture --> emel::model::parser::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::on_unexpected>
  emel::model::parser::mapping_architecture --> emel::model::parser::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::on_unexpected>
  emel::model::parser::parsing_hparams --> emel::model::parser::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::on_unexpected>
  emel::model::parser::parsing_vocab --> emel::model::parser::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::on_unexpected>
  emel::model::parser::mapping_tensors --> emel::model::parser::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::on_unexpected>
  emel::model::parser::done --> emel::model::parser::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::on_unexpected>
  emel::model::parser::errored --> emel::model::parser::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::parser::action::on_unexpected>
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| `emel::model::parser::initialized` | `emel::model::parser::event::parse_model` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::parse_architecture>` | `emel::model::parser::parsing_architecture` |
| `emel::model::parser::parsing_architecture` | `emel::model::parser::events::parse_architecture_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::map_architecture>` | `emel::model::parser::mapping_architecture` |
| `emel::model::parser::parsing_architecture` | `emel::model::parser::events::parse_architecture_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::dispatch_error>` | `emel::model::parser::errored` |
| `emel::model::parser::mapping_architecture` | `emel::model::parser::events::map_architecture_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::parse_hparams>` | `emel::model::parser::parsing_hparams` |
| `emel::model::parser::mapping_architecture` | `emel::model::parser::events::map_architecture_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::dispatch_error>` | `emel::model::parser::errored` |
| `emel::model::parser::parsing_hparams` | `emel::model::parser::events::parse_hparams_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::parse_vocab>` | `emel::model::parser::parsing_vocab` |
| `emel::model::parser::parsing_hparams` | `emel::model::parser::events::parse_hparams_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::dispatch_error>` | `emel::model::parser::errored` |
| `emel::model::parser::parsing_vocab` | `emel::model::parser::events::parse_vocab_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::map_tensors>` | `emel::model::parser::mapping_tensors` |
| `emel::model::parser::parsing_vocab` | `emel::model::parser::events::parse_vocab_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::dispatch_error>` | `emel::model::parser::errored` |
| `emel::model::parser::mapping_tensors` | `emel::model::parser::events::map_tensors_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::dispatch_done>` | `emel::model::parser::done` |
| `emel::model::parser::mapping_tensors` | `emel::model::parser::events::map_tensors_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::dispatch_error>` | `emel::model::parser::errored` |
| `emel::model::parser::initialized` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::on_unexpected>` | `emel::model::parser::errored` |
| `emel::model::parser::parsing_architecture` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::on_unexpected>` | `emel::model::parser::errored` |
| `emel::model::parser::mapping_architecture` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::on_unexpected>` | `emel::model::parser::errored` |
| `emel::model::parser::parsing_hparams` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::on_unexpected>` | `emel::model::parser::errored` |
| `emel::model::parser::parsing_vocab` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::on_unexpected>` | `emel::model::parser::errored` |
| `emel::model::parser::mapping_tensors` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::on_unexpected>` | `emel::model::parser::errored` |
| `emel::model::parser::done` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::on_unexpected>` | `emel::model::parser::errored` |
| `emel::model::parser::errored` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::parser::action::on_unexpected>` | `emel::model::parser::errored` |
