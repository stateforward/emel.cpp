# buffer_realloc_analyzer

Source: `emel/buffer/realloc_analyzer/sm.hpp`

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> idle
  idle --> validating : emel::buffer::realloc_analyzer::event::analyze [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_analyze>
  validating --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/realloc_analyzer/sm.hpp:65:11)>
  validating --> validating : emel::buffer::realloc_analyzer::event::validate [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::run_validate>
  validating --> evaluating : emel::buffer::realloc_analyzer::events::validate_done [boost::sml::front::always] / boost::sml::front::none
  validating --> failed : emel::buffer::realloc_analyzer::events::validate_error [boost::sml::front::always] / boost::sml::front::none
  evaluating --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/realloc_analyzer/sm.hpp:94:11)>
  evaluating --> evaluating : emel::buffer::realloc_analyzer::event::evaluate [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::run_evaluate>
  evaluating --> publishing : emel::buffer::realloc_analyzer::events::evaluate_done [boost::sml::front::always] / boost::sml::front::none
  evaluating --> failed : emel::buffer::realloc_analyzer::events::evaluate_error [boost::sml::front::always] / boost::sml::front::none
  publishing --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/realloc_analyzer/sm.hpp:123:11)>
  publishing --> publishing : emel::buffer::realloc_analyzer::event::publish [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::run_publish>
  publishing --> done : emel::buffer::realloc_analyzer::events::publish_done [boost::sml::front::always] / boost::sml::front::none
  publishing --> failed : emel::buffer::realloc_analyzer::events::publish_error [boost::sml::front::always] / boost::sml::front::none
  done --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/realloc_analyzer/sm.hpp:148:11)>
  done --> idle : emel::buffer::realloc_analyzer::events::analyze_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_analyze_done>
  done --> idle : emel::buffer::realloc_analyzer::events::analyze_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_analyze_error>
  idle --> resetting : emel::buffer::realloc_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>
  validating --> resetting : emel::buffer::realloc_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>
  evaluating --> resetting : emel::buffer::realloc_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>
  publishing --> resetting : emel::buffer::realloc_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>
  done --> resetting : emel::buffer::realloc_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>
  failed --> resetting : emel::buffer::realloc_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>
  resetting --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/realloc_analyzer/sm.hpp:172:11)>
  resetting --> idle : emel::buffer::realloc_analyzer::events::reset_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_reset_done>
  resetting --> failed : emel::buffer::realloc_analyzer::events::reset_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_reset_error>
  failed --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/realloc_analyzer/sm.hpp:184:11)>
  failed --> idle : emel::buffer::realloc_analyzer::events::analyze_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_analyze_error>
  idle --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_unexpected>
  validating --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_unexpected>
  evaluating --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_unexpected>
  publishing --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_unexpected>
  done --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_unexpected>
  resetting --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_unexpected>
  failed --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_unexpected>
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| `idle` | `emel::buffer::realloc_analyzer::event::analyze` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_analyze>` | `validating` |
| `validating` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/realloc_analyzer/sm.hpp:65:11)>` | `boost::sml::front::internal` |
| `validating` | `emel::buffer::realloc_analyzer::event::validate` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::run_validate>` | `validating` |
| `validating` | `emel::buffer::realloc_analyzer::events::validate_done` | `boost::sml::front::always` | `boost::sml::front::none` | `evaluating` |
| `validating` | `emel::buffer::realloc_analyzer::events::validate_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `evaluating` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/realloc_analyzer/sm.hpp:94:11)>` | `boost::sml::front::internal` |
| `evaluating` | `emel::buffer::realloc_analyzer::event::evaluate` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::run_evaluate>` | `evaluating` |
| `evaluating` | `emel::buffer::realloc_analyzer::events::evaluate_done` | `boost::sml::front::always` | `boost::sml::front::none` | `publishing` |
| `evaluating` | `emel::buffer::realloc_analyzer::events::evaluate_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `publishing` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/realloc_analyzer/sm.hpp:123:11)>` | `boost::sml::front::internal` |
| `publishing` | `emel::buffer::realloc_analyzer::event::publish` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::run_publish>` | `publishing` |
| `publishing` | `emel::buffer::realloc_analyzer::events::publish_done` | `boost::sml::front::always` | `boost::sml::front::none` | `done` |
| `publishing` | `emel::buffer::realloc_analyzer::events::publish_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `done` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/realloc_analyzer/sm.hpp:148:11)>` | `boost::sml::front::internal` |
| `done` | `emel::buffer::realloc_analyzer::events::analyze_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_analyze_done>` | `idle` |
| `done` | `emel::buffer::realloc_analyzer::events::analyze_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_analyze_error>` | `idle` |
| `idle` | `emel::buffer::realloc_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>` | `resetting` |
| `validating` | `emel::buffer::realloc_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>` | `resetting` |
| `evaluating` | `emel::buffer::realloc_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>` | `resetting` |
| `publishing` | `emel::buffer::realloc_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>` | `resetting` |
| `done` | `emel::buffer::realloc_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>` | `resetting` |
| `failed` | `emel::buffer::realloc_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>` | `resetting` |
| `resetting` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/realloc_analyzer/sm.hpp:172:11)>` | `boost::sml::front::internal` |
| `resetting` | `emel::buffer::realloc_analyzer::events::reset_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_reset_done>` | `idle` |
| `resetting` | `emel::buffer::realloc_analyzer::events::reset_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_reset_error>` | `failed` |
| `failed` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/realloc_analyzer/sm.hpp:184:11)>` | `boost::sml::front::internal` |
| `failed` | `emel::buffer::realloc_analyzer::events::analyze_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_analyze_error>` | `idle` |
| `idle` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_unexpected>` | `failed` |
| `validating` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_unexpected>` | `failed` |
| `evaluating` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_unexpected>` | `failed` |
| `publishing` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_unexpected>` | `failed` |
| `done` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_unexpected>` | `failed` |
| `resetting` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_unexpected>` | `failed` |
| `failed` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_unexpected>` | `failed` |
