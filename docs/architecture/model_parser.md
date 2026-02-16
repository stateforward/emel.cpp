# model_parser

Source: `emel/model/parser/sm.hpp`

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> initialized
  initialized --> parsing_architecture : emel::model::parser::event::parse_model [boost::sml::front::always] / boost::sml::front::none
  parsing_architecture --> mapping_architecture : emel::model::parser::event::parse_architecture_done [boost::sml::front::always] / boost::sml::front::none
  parsing_architecture --> errored : emel::model::parser::event::parse_architecture_error [boost::sml::front::always] / boost::sml::front::none
  mapping_architecture --> parsing_hparams : emel::model::parser::event::map_architecture_done [boost::sml::front::always] / boost::sml::front::none
  mapping_architecture --> errored : emel::model::parser::event::map_architecture_error [boost::sml::front::always] / boost::sml::front::none
  parsing_hparams --> parsing_vocab : emel::model::parser::event::parse_hparams_done [boost::sml::front::always] / boost::sml::front::none
  parsing_hparams --> errored : emel::model::parser::event::parse_hparams_error [boost::sml::front::always] / boost::sml::front::none
  parsing_vocab --> mapping_tensors : emel::model::parser::event::parse_vocab_done [boost::sml::front::always] / boost::sml::front::none
  parsing_vocab --> errored : emel::model::parser::event::parse_vocab_error [boost::sml::front::always] / boost::sml::front::none
  mapping_tensors --> done : emel::model::parser::event::map_tensors_done [boost::sml::front::always] / boost::sml::front::none
  mapping_tensors --> errored : emel::model::parser::event::map_tensors_error [boost::sml::front::always] / boost::sml::front::none
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| `initialized` | `emel::model::parser::event::parse_model` | `boost::sml::front::always` | `boost::sml::front::none` | `parsing_architecture` |
| `parsing_architecture` | `emel::model::parser::event::parse_architecture_done` | `boost::sml::front::always` | `boost::sml::front::none` | `mapping_architecture` |
| `parsing_architecture` | `emel::model::parser::event::parse_architecture_error` | `boost::sml::front::always` | `boost::sml::front::none` | `errored` |
| `mapping_architecture` | `emel::model::parser::event::map_architecture_done` | `boost::sml::front::always` | `boost::sml::front::none` | `parsing_hparams` |
| `mapping_architecture` | `emel::model::parser::event::map_architecture_error` | `boost::sml::front::always` | `boost::sml::front::none` | `errored` |
| `parsing_hparams` | `emel::model::parser::event::parse_hparams_done` | `boost::sml::front::always` | `boost::sml::front::none` | `parsing_vocab` |
| `parsing_hparams` | `emel::model::parser::event::parse_hparams_error` | `boost::sml::front::always` | `boost::sml::front::none` | `errored` |
| `parsing_vocab` | `emel::model::parser::event::parse_vocab_done` | `boost::sml::front::always` | `boost::sml::front::none` | `mapping_tensors` |
| `parsing_vocab` | `emel::model::parser::event::parse_vocab_error` | `boost::sml::front::always` | `boost::sml::front::none` | `errored` |
| `mapping_tensors` | `emel::model::parser::event::map_tensors_done` | `boost::sml::front::always` | `boost::sml::front::none` | `done` |
| `mapping_tensors` | `emel::model::parser::event::map_tensors_error` | `boost::sml::front::always` | `boost::sml::front::none` | `errored` |
