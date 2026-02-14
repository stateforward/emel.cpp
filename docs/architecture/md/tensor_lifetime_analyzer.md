# tensor_lifetime_analyzer

Source: `emel/tensor/lifetime_analyzer/sm.hpp`

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> idle
  idle --> validating : emel::tensor::lifetime_analyzer::event::analyze [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::begin_analyze>
  validating --> validating : emel::tensor::lifetime_analyzer::event::validate [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::run_validate>
  validating --> collecting_ranges : emel::tensor::lifetime_analyzer::events::validate_done [boost::sml::front::always] / boost::sml::front::none
  validating --> failed : emel::tensor::lifetime_analyzer::events::validate_error [boost::sml::front::always] / boost::sml::front::none
  collecting_ranges --> collecting_ranges : emel::tensor::lifetime_analyzer::event::collect_ranges [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::run_collect_ranges>
  collecting_ranges --> publishing : emel::tensor::lifetime_analyzer::events::collect_ranges_done [boost::sml::front::always] / boost::sml::front::none
  collecting_ranges --> failed : emel::tensor::lifetime_analyzer::events::collect_ranges_error [boost::sml::front::always] / boost::sml::front::none
  publishing --> publishing : emel::tensor::lifetime_analyzer::event::publish [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::run_publish>
  publishing --> done : emel::tensor::lifetime_analyzer::events::publish_done [boost::sml::front::always] / boost::sml::front::none
  publishing --> failed : emel::tensor::lifetime_analyzer::events::publish_error [boost::sml::front::always] / boost::sml::front::none
  done --> idle : emel::tensor::lifetime_analyzer::events::analyze_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::on_analyze_done>
  failed --> idle : emel::tensor::lifetime_analyzer::events::analyze_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::on_analyze_error>
  idle --> resetting : emel::tensor::lifetime_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::begin_reset>
  validating --> resetting : emel::tensor::lifetime_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::begin_reset>
  collecting_ranges --> resetting : emel::tensor::lifetime_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::begin_reset>
  publishing --> resetting : emel::tensor::lifetime_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::begin_reset>
  done --> resetting : emel::tensor::lifetime_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::begin_reset>
  failed --> resetting : emel::tensor::lifetime_analyzer::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::begin_reset>
  resetting --> idle : emel::tensor::lifetime_analyzer::events::reset_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::on_reset_done>
  resetting --> failed : emel::tensor::lifetime_analyzer::events::reset_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::on_reset_error>
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| `idle` | `emel::tensor::lifetime_analyzer::event::analyze` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::begin_analyze>` | `validating` |
| `validating` | `emel::tensor::lifetime_analyzer::event::validate` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::run_validate>` | `validating` |
| `validating` | `emel::tensor::lifetime_analyzer::events::validate_done` | `boost::sml::front::always` | `boost::sml::front::none` | `collecting_ranges` |
| `validating` | `emel::tensor::lifetime_analyzer::events::validate_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `collecting_ranges` | `emel::tensor::lifetime_analyzer::event::collect_ranges` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::run_collect_ranges>` | `collecting_ranges` |
| `collecting_ranges` | `emel::tensor::lifetime_analyzer::events::collect_ranges_done` | `boost::sml::front::always` | `boost::sml::front::none` | `publishing` |
| `collecting_ranges` | `emel::tensor::lifetime_analyzer::events::collect_ranges_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `publishing` | `emel::tensor::lifetime_analyzer::event::publish` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::run_publish>` | `publishing` |
| `publishing` | `emel::tensor::lifetime_analyzer::events::publish_done` | `boost::sml::front::always` | `boost::sml::front::none` | `done` |
| `publishing` | `emel::tensor::lifetime_analyzer::events::publish_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `done` | `emel::tensor::lifetime_analyzer::events::analyze_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::on_analyze_done>` | `idle` |
| `failed` | `emel::tensor::lifetime_analyzer::events::analyze_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::on_analyze_error>` | `idle` |
| `idle` | `emel::tensor::lifetime_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::begin_reset>` | `resetting` |
| `validating` | `emel::tensor::lifetime_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::begin_reset>` | `resetting` |
| `collecting_ranges` | `emel::tensor::lifetime_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::begin_reset>` | `resetting` |
| `publishing` | `emel::tensor::lifetime_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::begin_reset>` | `resetting` |
| `done` | `emel::tensor::lifetime_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::begin_reset>` | `resetting` |
| `failed` | `emel::tensor::lifetime_analyzer::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::begin_reset>` | `resetting` |
| `resetting` | `emel::tensor::lifetime_analyzer::events::reset_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::on_reset_done>` | `idle` |
| `resetting` | `emel::tensor::lifetime_analyzer::events::reset_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::lifetime_analyzer::action::on_reset_error>` | `failed` |
