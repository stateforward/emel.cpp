# buffer_realloc_analyzer

Source: `emel/buffer/realloc_analyzer/sm.hpp`

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> idle
  idle --> validating : emel::buffer::realloc_analyzer::event::analyze [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_analyze>
  validating --> validating : emel::buffer::realloc_analyzer::event::validate [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::run_validate>
  validating --> evaluating : emel::buffer::realloc_analyzer::events::validate_done [boost::sml::front::always] / boost::sml::front::none
  validating --> failed : emel::buffer::realloc_analyzer::events::validate_error [boost::sml::front::always] / boost::sml::front::none
  evaluating --> evaluating : emel::buffer::realloc_analyzer::event::evaluate [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::run_evaluate>
  evaluating --> publishing : emel::buffer::realloc_analyzer::events::evaluate_done [boost::sml::front::always] / boost::sml::front::none
  evaluating --> failed : emel::buffer::realloc_analyzer::events::evaluate_error [boost::sml::front::always] / boost::sml::front::none
  publishing --> publishing : emel::buffer::realloc_analyzer::event::publish [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::run_publish>
  publishing --> done : emel::buffer::realloc_analyzer::events::publish_done [boost::sml::front::always] / boost::sml::front::none
  publishing --> failed : emel::buffer::realloc_analyzer::events::publish_error [boost::sml::front::always] / boost::sml::front::none
  done --> idle : emel::buffer::realloc_analyzer::events::analyze_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_analyze_done>
  failed --> idle : emel::buffer::realloc_analyzer::events::analyze_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_analyze_error>
  idle --> resetting : emel::buffer::realloc_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>
  validating --> resetting : emel::buffer::realloc_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>
  evaluating --> resetting : emel::buffer::realloc_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>
  publishing --> resetting : emel::buffer::realloc_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>
  done --> resetting : emel::buffer::realloc_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>
  failed --> resetting : emel::buffer::realloc_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>
  resetting --> idle : emel::buffer::realloc_analyzer::events::reset_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_reset_done>
  resetting --> failed : emel::buffer::realloc_analyzer::events::reset_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_reset_error>
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| `idle` | `emel::buffer::realloc_analyzer::event::analyze` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_analyze>` | `validating` |
| `validating` | `emel::buffer::realloc_analyzer::event::validate` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::run_validate>` | `validating` |
| `validating` | `emel::buffer::realloc_analyzer::events::validate_done` | `boost::sml::front::always` | `boost::sml::front::none` | `evaluating` |
| `validating` | `emel::buffer::realloc_analyzer::events::validate_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `evaluating` | `emel::buffer::realloc_analyzer::event::evaluate` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::run_evaluate>` | `evaluating` |
| `evaluating` | `emel::buffer::realloc_analyzer::events::evaluate_done` | `boost::sml::front::always` | `boost::sml::front::none` | `publishing` |
| `evaluating` | `emel::buffer::realloc_analyzer::events::evaluate_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `publishing` | `emel::buffer::realloc_analyzer::event::publish` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::run_publish>` | `publishing` |
| `publishing` | `emel::buffer::realloc_analyzer::events::publish_done` | `boost::sml::front::always` | `boost::sml::front::none` | `done` |
| `publishing` | `emel::buffer::realloc_analyzer::events::publish_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `done` | `emel::buffer::realloc_analyzer::events::analyze_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_analyze_done>` | `idle` |
| `failed` | `emel::buffer::realloc_analyzer::events::analyze_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_analyze_error>` | `idle` |
| `idle` | `emel::buffer::realloc_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>` | `resetting` |
| `validating` | `emel::buffer::realloc_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>` | `resetting` |
| `evaluating` | `emel::buffer::realloc_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>` | `resetting` |
| `publishing` | `emel::buffer::realloc_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>` | `resetting` |
| `done` | `emel::buffer::realloc_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>` | `resetting` |
| `failed` | `emel::buffer::realloc_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::begin_reset>` | `resetting` |
| `resetting` | `emel::buffer::realloc_analyzer::events::reset_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_reset_done>` | `idle` |
| `resetting` | `emel::buffer::realloc_analyzer::events::reset_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::realloc_analyzer::action::on_reset_error>` | `failed` |
