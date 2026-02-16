# model_weight_loader

Source: `emel/model/weight_loader/sm.hpp`

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> emel::model::weight_loader::initialized
  emel::model::weight_loader::initialized --> emel::model::weight_loader::loading_mmap : emel::model::weight_loader::event::load_weights [boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::use_mmap>] / boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::load_mmap>
  emel::model::weight_loader::initialized --> emel::model::weight_loader::loading_streamed : emel::model::weight_loader::event::load_weights [boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::use_stream>] / boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::load_streamed>
  emel::model::weight_loader::loading_mmap --> emel::model::weight_loader::done : emel::model::weight_loader::events::weights_loaded [boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::no_error>] / boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::store_and_dispatch_done>
  emel::model::weight_loader::loading_mmap --> emel::model::weight_loader::errored : emel::model::weight_loader::events::weights_loaded [boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::has_error>] / boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::store_and_dispatch_error>
  emel::model::weight_loader::loading_streamed --> emel::model::weight_loader::done : emel::model::weight_loader::events::weights_loaded [boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::no_error>] / boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::store_and_dispatch_done>
  emel::model::weight_loader::loading_streamed --> emel::model::weight_loader::errored : emel::model::weight_loader::events::weights_loaded [boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::has_error>] / boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::store_and_dispatch_error>
  emel::model::weight_loader::initialized --> emel::model::weight_loader::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::on_unexpected>
  emel::model::weight_loader::loading_mmap --> emel::model::weight_loader::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::on_unexpected>
  emel::model::weight_loader::loading_streamed --> emel::model::weight_loader::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::on_unexpected>
  emel::model::weight_loader::done --> emel::model::weight_loader::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::on_unexpected>
  emel::model::weight_loader::errored --> emel::model::weight_loader::errored : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::on_unexpected>
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| `emel::model::weight_loader::initialized` | `emel::model::weight_loader::event::load_weights` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::use_mmap>` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::load_mmap>` | `emel::model::weight_loader::loading_mmap` |
| `emel::model::weight_loader::initialized` | `emel::model::weight_loader::event::load_weights` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::use_stream>` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::load_streamed>` | `emel::model::weight_loader::loading_streamed` |
| `emel::model::weight_loader::loading_mmap` | `emel::model::weight_loader::events::weights_loaded` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::no_error>` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::store_and_dispatch_done>` | `emel::model::weight_loader::done` |
| `emel::model::weight_loader::loading_mmap` | `emel::model::weight_loader::events::weights_loaded` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::has_error>` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::store_and_dispatch_error>` | `emel::model::weight_loader::errored` |
| `emel::model::weight_loader::loading_streamed` | `emel::model::weight_loader::events::weights_loaded` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::no_error>` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::store_and_dispatch_done>` | `emel::model::weight_loader::done` |
| `emel::model::weight_loader::loading_streamed` | `emel::model::weight_loader::events::weights_loaded` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::guard::has_error>` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::store_and_dispatch_error>` | `emel::model::weight_loader::errored` |
| `emel::model::weight_loader::initialized` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::on_unexpected>` | `emel::model::weight_loader::errored` |
| `emel::model::weight_loader::loading_mmap` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::on_unexpected>` | `emel::model::weight_loader::errored` |
| `emel::model::weight_loader::loading_streamed` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::on_unexpected>` | `emel::model::weight_loader::errored` |
| `emel::model::weight_loader::done` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::on_unexpected>` | `emel::model::weight_loader::errored` |
| `emel::model::weight_loader::errored` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::model::weight_loader::action::on_unexpected>` | `emel::model::weight_loader::errored` |
