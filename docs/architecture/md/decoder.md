# decoder

Source: `emel/decoder/sm.hpp`

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> initialized
  initialized --> detokenizing : emel::decoder::event::decode [boost::sml::front::always] / boost::sml::front::none
  detokenizing --> done : emel::decoder::event::detokenized_done [boost::sml::front::always] / boost::sml::front::none
  detokenizing --> errored : emel::decoder::event::detokenized_error [boost::sml::front::always] / boost::sml::front::none
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| `initialized` | `emel::decoder::event::decode` | `boost::sml::front::always` | `boost::sml::front::none` | `detokenizing` |
| `detokenizing` | `emel::decoder::event::detokenized_done` | `boost::sml::front::always` | `boost::sml::front::none` | `done` |
| `detokenizing` | `emel::decoder::event::detokenized_error` | `boost::sml::front::always` | `boost::sml::front::none` | `errored` |
