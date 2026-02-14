# tensor_allocator

Source: `emel/tensor/allocator/sm.hpp`

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> idle
  idle --> validating : emel::tensor::allocator::event::allocate_tensors [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_allocate_tensors>
  validating --> validating : emel::tensor::allocator::event::validate [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::run_validate>
  validating --> scanning_tensors : emel::tensor::allocator::events::validate_done [boost::sml::front::always] / boost::sml::front::none
  validating --> failed : emel::tensor::allocator::events::validate_error [boost::sml::front::always] / boost::sml::front::none
  scanning_tensors --> scanning_tensors : emel::tensor::allocator::event::scan_tensors [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::run_scan_tensors>
  scanning_tensors --> partitioning_ranges : emel::tensor::allocator::events::scan_done [boost::sml::front::always] / boost::sml::front::none
  scanning_tensors --> failed : emel::tensor::allocator::events::scan_error [boost::sml::front::always] / boost::sml::front::none
  partitioning_ranges --> partitioning_ranges : emel::tensor::allocator::event::partition_ranges [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::run_partition_ranges>
  partitioning_ranges --> allocating_ranges : emel::tensor::allocator::events::partition_done [boost::sml::front::always] / boost::sml::front::none
  partitioning_ranges --> failed : emel::tensor::allocator::events::partition_error [boost::sml::front::always] / boost::sml::front::none
  allocating_ranges --> allocating_ranges : emel::tensor::allocator::event::allocate_ranges [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::run_allocate_ranges>
  allocating_ranges --> initializing_tensors : emel::tensor::allocator::events::allocate_ranges_done [boost::sml::front::always] / boost::sml::front::none
  allocating_ranges --> failed : emel::tensor::allocator::events::allocate_ranges_error [boost::sml::front::always] / boost::sml::front::none
  initializing_tensors --> initializing_tensors : emel::tensor::allocator::event::initialize_tensors [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::run_initialize_tensors>
  initializing_tensors --> assembling_result : emel::tensor::allocator::events::initialize_tensors_done [boost::sml::front::always] / boost::sml::front::none
  initializing_tensors --> failed : emel::tensor::allocator::events::initialize_tensors_error [boost::sml::front::always] / boost::sml::front::none
  assembling_result --> assembling_result : emel::tensor::allocator::event::assemble [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::run_assemble>
  assembling_result --> done : emel::tensor::allocator::events::assemble_done [boost::sml::front::always] / boost::sml::front::none
  assembling_result --> failed : emel::tensor::allocator::events::assemble_error [boost::sml::front::always] / boost::sml::front::none
  done --> idle : emel::tensor::allocator::events::allocate_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::on_allocate_done>
  failed --> idle : emel::tensor::allocator::events::allocate_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::on_allocate_error>
  idle --> releasing : emel::tensor::allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>
  validating --> releasing : emel::tensor::allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>
  scanning_tensors --> releasing : emel::tensor::allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>
  partitioning_ranges --> releasing : emel::tensor::allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>
  allocating_ranges --> releasing : emel::tensor::allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>
  initializing_tensors --> releasing : emel::tensor::allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>
  assembling_result --> releasing : emel::tensor::allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>
  done --> releasing : emel::tensor::allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>
  failed --> releasing : emel::tensor::allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>
  releasing --> idle : emel::tensor::allocator::events::release_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::on_release_done>
  releasing --> failed : emel::tensor::allocator::events::release_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::on_release_error>
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| `idle` | `emel::tensor::allocator::event::allocate_tensors` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_allocate_tensors>` | `validating` |
| `validating` | `emel::tensor::allocator::event::validate` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::run_validate>` | `validating` |
| `validating` | `emel::tensor::allocator::events::validate_done` | `boost::sml::front::always` | `boost::sml::front::none` | `scanning_tensors` |
| `validating` | `emel::tensor::allocator::events::validate_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `scanning_tensors` | `emel::tensor::allocator::event::scan_tensors` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::run_scan_tensors>` | `scanning_tensors` |
| `scanning_tensors` | `emel::tensor::allocator::events::scan_done` | `boost::sml::front::always` | `boost::sml::front::none` | `partitioning_ranges` |
| `scanning_tensors` | `emel::tensor::allocator::events::scan_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `partitioning_ranges` | `emel::tensor::allocator::event::partition_ranges` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::run_partition_ranges>` | `partitioning_ranges` |
| `partitioning_ranges` | `emel::tensor::allocator::events::partition_done` | `boost::sml::front::always` | `boost::sml::front::none` | `allocating_ranges` |
| `partitioning_ranges` | `emel::tensor::allocator::events::partition_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `allocating_ranges` | `emel::tensor::allocator::event::allocate_ranges` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::run_allocate_ranges>` | `allocating_ranges` |
| `allocating_ranges` | `emel::tensor::allocator::events::allocate_ranges_done` | `boost::sml::front::always` | `boost::sml::front::none` | `initializing_tensors` |
| `allocating_ranges` | `emel::tensor::allocator::events::allocate_ranges_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `initializing_tensors` | `emel::tensor::allocator::event::initialize_tensors` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::run_initialize_tensors>` | `initializing_tensors` |
| `initializing_tensors` | `emel::tensor::allocator::events::initialize_tensors_done` | `boost::sml::front::always` | `boost::sml::front::none` | `assembling_result` |
| `initializing_tensors` | `emel::tensor::allocator::events::initialize_tensors_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `assembling_result` | `emel::tensor::allocator::event::assemble` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::run_assemble>` | `assembling_result` |
| `assembling_result` | `emel::tensor::allocator::events::assemble_done` | `boost::sml::front::always` | `boost::sml::front::none` | `done` |
| `assembling_result` | `emel::tensor::allocator::events::assemble_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `done` | `emel::tensor::allocator::events::allocate_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::on_allocate_done>` | `idle` |
| `failed` | `emel::tensor::allocator::events::allocate_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::on_allocate_error>` | `idle` |
| `idle` | `emel::tensor::allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>` | `releasing` |
| `validating` | `emel::tensor::allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>` | `releasing` |
| `scanning_tensors` | `emel::tensor::allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>` | `releasing` |
| `partitioning_ranges` | `emel::tensor::allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>` | `releasing` |
| `allocating_ranges` | `emel::tensor::allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>` | `releasing` |
| `initializing_tensors` | `emel::tensor::allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>` | `releasing` |
| `assembling_result` | `emel::tensor::allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>` | `releasing` |
| `done` | `emel::tensor::allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>` | `releasing` |
| `failed` | `emel::tensor::allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::begin_release>` | `releasing` |
| `releasing` | `emel::tensor::allocator::events::release_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::on_release_done>` | `idle` |
| `releasing` | `emel::tensor::allocator::events::release_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::tensor::allocator::action::on_release_error>` | `failed` |
