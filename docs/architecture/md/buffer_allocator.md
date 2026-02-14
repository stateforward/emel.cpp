# buffer_allocator

Source: `emel/buffer/allocator/sm.hpp`

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> uninitialized
  uninitialized --> initializing : emel::buffer::allocator::event::initialize [boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::valid_initialize>] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_initialize>
  initializing --> ready : emel::buffer::allocator::events::initialize_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_initialize_done>
  initializing --> failed : emel::buffer::allocator::events::initialize_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_initialize_error>
  ready --> reserving_n_size : emel::buffer::allocator::event::reserve_n_size [boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::can_reserve_n_size>] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_reserve_n_size>
  allocated --> reserving_n_size : emel::buffer::allocator::event::reserve_n_size [boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::can_reserve_n_size>] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_reserve_n_size>
  reserving_n_size --> ready : emel::buffer::allocator::events::reserve_n_size_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_reserve_n_size_done>
  reserving_n_size --> failed : emel::buffer::allocator::events::reserve_n_size_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_reserve_n_size_error>
  ready --> reserving : emel::buffer::allocator::event::reserve_n [boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::can_reserve_n>] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_reserve_n>
  allocated --> reserving : emel::buffer::allocator::event::reserve_n [boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::can_reserve_n>] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_reserve_n>
  ready --> reserving : emel::buffer::allocator::event::reserve [boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::can_reserve>] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_reserve>
  allocated --> reserving : emel::buffer::allocator::event::reserve [boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::can_reserve>] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_reserve>
  reserving --> ready : emel::buffer::allocator::events::reserve_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_reserve_done>
  reserving --> failed : emel::buffer::allocator::events::reserve_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_reserve_error>
  ready --> allocating_graph : emel::buffer::allocator::event::alloc_graph [boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::can_alloc_graph>] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_alloc_graph>
  allocated --> allocating_graph : emel::buffer::allocator::event::alloc_graph [boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::can_alloc_graph>] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_alloc_graph>
  allocating_graph --> allocated : emel::buffer::allocator::events::alloc_graph_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_alloc_graph_done>
  allocating_graph --> failed : emel::buffer::allocator::events::alloc_graph_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_alloc_graph_error>
  uninitialized --> releasing : emel::buffer::allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_release>
  initializing --> releasing : emel::buffer::allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_release>
  ready --> releasing : emel::buffer::allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_release>
  reserving_n_size --> releasing : emel::buffer::allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_release>
  reserving --> releasing : emel::buffer::allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_release>
  allocating_graph --> releasing : emel::buffer::allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_release>
  allocated --> releasing : emel::buffer::allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_release>
  failed --> releasing : emel::buffer::allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_release>
  releasing --> uninitialized : emel::buffer::allocator::events::release_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_release_done>
  releasing --> failed : emel::buffer::allocator::events::release_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_release_error>
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| `uninitialized` | `emel::buffer::allocator::event::initialize` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::valid_initialize>` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_initialize>` | `initializing` |
| `initializing` | `emel::buffer::allocator::events::initialize_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_initialize_done>` | `ready` |
| `initializing` | `emel::buffer::allocator::events::initialize_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_initialize_error>` | `failed` |
| `ready` | `emel::buffer::allocator::event::reserve_n_size` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::can_reserve_n_size>` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_reserve_n_size>` | `reserving_n_size` |
| `allocated` | `emel::buffer::allocator::event::reserve_n_size` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::can_reserve_n_size>` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_reserve_n_size>` | `reserving_n_size` |
| `reserving_n_size` | `emel::buffer::allocator::events::reserve_n_size_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_reserve_n_size_done>` | `ready` |
| `reserving_n_size` | `emel::buffer::allocator::events::reserve_n_size_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_reserve_n_size_error>` | `failed` |
| `ready` | `emel::buffer::allocator::event::reserve_n` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::can_reserve_n>` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_reserve_n>` | `reserving` |
| `allocated` | `emel::buffer::allocator::event::reserve_n` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::can_reserve_n>` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_reserve_n>` | `reserving` |
| `ready` | `emel::buffer::allocator::event::reserve` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::can_reserve>` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_reserve>` | `reserving` |
| `allocated` | `emel::buffer::allocator::event::reserve` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::can_reserve>` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_reserve>` | `reserving` |
| `reserving` | `emel::buffer::allocator::events::reserve_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_reserve_done>` | `ready` |
| `reserving` | `emel::buffer::allocator::events::reserve_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_reserve_error>` | `failed` |
| `ready` | `emel::buffer::allocator::event::alloc_graph` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::can_alloc_graph>` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_alloc_graph>` | `allocating_graph` |
| `allocated` | `emel::buffer::allocator::event::alloc_graph` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::guard::can_alloc_graph>` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_alloc_graph>` | `allocating_graph` |
| `allocating_graph` | `emel::buffer::allocator::events::alloc_graph_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_alloc_graph_done>` | `allocated` |
| `allocating_graph` | `emel::buffer::allocator::events::alloc_graph_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_alloc_graph_error>` | `failed` |
| `uninitialized` | `emel::buffer::allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_release>` | `releasing` |
| `initializing` | `emel::buffer::allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_release>` | `releasing` |
| `ready` | `emel::buffer::allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_release>` | `releasing` |
| `reserving_n_size` | `emel::buffer::allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_release>` | `releasing` |
| `reserving` | `emel::buffer::allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_release>` | `releasing` |
| `allocating_graph` | `emel::buffer::allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_release>` | `releasing` |
| `allocated` | `emel::buffer::allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_release>` | `releasing` |
| `failed` | `emel::buffer::allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::begin_release>` | `releasing` |
| `releasing` | `emel::buffer::allocator::events::release_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_release_done>` | `uninitialized` |
| `releasing` | `emel::buffer::allocator::events::release_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::allocator::action::on_release_error>` | `failed` |
