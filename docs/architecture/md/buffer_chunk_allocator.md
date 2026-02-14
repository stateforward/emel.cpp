# buffer_chunk_allocator

Source: `emel/buffer/chunk_allocator/sm.hpp`

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> ready
  ready --> configuring : emel::buffer::chunk_allocator::event::configure [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_configure>
  configuring --> configuring : emel::buffer::chunk_allocator::event::validate_configure [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_validate_configure>
  configuring --> applying_configure : emel::buffer::chunk_allocator::events::validate_configure_done [boost::sml::front::always] / boost::sml::front::none
  configuring --> failed : emel::buffer::chunk_allocator::events::validate_configure_error [boost::sml::front::always] / boost::sml::front::none
  applying_configure --> applying_configure : emel::buffer::chunk_allocator::event::apply_configure [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_apply_configure>
  applying_configure --> configure_done : emel::buffer::chunk_allocator::events::apply_configure_done [boost::sml::front::always] / boost::sml::front::none
  applying_configure --> failed : emel::buffer::chunk_allocator::events::apply_configure_error [boost::sml::front::always] / boost::sml::front::none
  configure_done --> ready : emel::buffer::chunk_allocator::events::configure_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_configure_done>
  ready --> validating_allocate : emel::buffer::chunk_allocator::event::allocate [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_allocate>
  validating_allocate --> validating_allocate : emel::buffer::chunk_allocator::event::validate_allocate [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_validate_allocate>
  validating_allocate --> selecting_block : emel::buffer::chunk_allocator::events::validate_allocate_done [boost::sml::front::always] / boost::sml::front::none
  validating_allocate --> failed : emel::buffer::chunk_allocator::events::validate_allocate_error [boost::sml::front::always] / boost::sml::front::none
  selecting_block --> selecting_block : emel::buffer::chunk_allocator::event::select_block [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_select_block>
  selecting_block --> ensuring_chunk : emel::buffer::chunk_allocator::events::select_block_done [boost::sml::front::always] / boost::sml::front::none
  selecting_block --> failed : emel::buffer::chunk_allocator::events::select_block_error [boost::sml::front::always] / boost::sml::front::none
  ensuring_chunk --> ensuring_chunk : emel::buffer::chunk_allocator::event::ensure_chunk [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_ensure_chunk>
  ensuring_chunk --> committing_allocate : emel::buffer::chunk_allocator::events::ensure_chunk_done [boost::sml::front::always] / boost::sml::front::none
  ensuring_chunk --> failed : emel::buffer::chunk_allocator::events::ensure_chunk_error [boost::sml::front::always] / boost::sml::front::none
  committing_allocate --> committing_allocate : emel::buffer::chunk_allocator::event::commit_allocate [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_commit_allocate>
  committing_allocate --> allocate_done : emel::buffer::chunk_allocator::events::commit_allocate_done [boost::sml::front::always] / boost::sml::front::none
  committing_allocate --> failed : emel::buffer::chunk_allocator::events::commit_allocate_error [boost::sml::front::always] / boost::sml::front::none
  allocate_done --> ready : emel::buffer::chunk_allocator::events::allocate_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_allocate_done>
  ready --> validating_release : emel::buffer::chunk_allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_release>
  validating_release --> validating_release : emel::buffer::chunk_allocator::event::validate_release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_validate_release>
  validating_release --> merging_release : emel::buffer::chunk_allocator::events::validate_release_done [boost::sml::front::always] / boost::sml::front::none
  validating_release --> failed : emel::buffer::chunk_allocator::events::validate_release_error [boost::sml::front::always] / boost::sml::front::none
  merging_release --> merging_release : emel::buffer::chunk_allocator::event::merge_release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_merge_release>
  merging_release --> release_done : emel::buffer::chunk_allocator::events::merge_release_done [boost::sml::front::always] / boost::sml::front::none
  merging_release --> failed : emel::buffer::chunk_allocator::events::merge_release_error [boost::sml::front::always] / boost::sml::front::none
  release_done --> ready : emel::buffer::chunk_allocator::events::release_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_release_done>
  ready --> resetting : emel::buffer::chunk_allocator::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>
  configuring --> resetting : emel::buffer::chunk_allocator::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>
  applying_configure --> resetting : emel::buffer::chunk_allocator::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>
  configure_done --> resetting : emel::buffer::chunk_allocator::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>
  validating_allocate --> resetting : emel::buffer::chunk_allocator::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>
  selecting_block --> resetting : emel::buffer::chunk_allocator::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>
  ensuring_chunk --> resetting : emel::buffer::chunk_allocator::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>
  committing_allocate --> resetting : emel::buffer::chunk_allocator::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>
  allocate_done --> resetting : emel::buffer::chunk_allocator::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>
  validating_release --> resetting : emel::buffer::chunk_allocator::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>
  merging_release --> resetting : emel::buffer::chunk_allocator::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>
  release_done --> resetting : emel::buffer::chunk_allocator::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>
  failed --> resetting : emel::buffer::chunk_allocator::event::reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>
  resetting --> resetting : emel::buffer::chunk_allocator::event::apply_reset [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_apply_reset>
  resetting --> reset_done : emel::buffer::chunk_allocator::events::apply_reset_done [boost::sml::front::always] / boost::sml::front::none
  resetting --> failed : emel::buffer::chunk_allocator::events::apply_reset_error [boost::sml::front::always] / boost::sml::front::none
  reset_done --> ready : emel::buffer::chunk_allocator::events::reset_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_reset_done>
  failed --> ready : emel::buffer::chunk_allocator::events::configure_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_configure_error>
  failed --> ready : emel::buffer::chunk_allocator::events::allocate_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_allocate_error>
  failed --> ready : emel::buffer::chunk_allocator::events::release_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_release_error>
  failed --> ready : emel::buffer::chunk_allocator::events::reset_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_reset_error>
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| `ready` | `emel::buffer::chunk_allocator::event::configure` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_configure>` | `configuring` |
| `configuring` | `emel::buffer::chunk_allocator::event::validate_configure` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_validate_configure>` | `configuring` |
| `configuring` | `emel::buffer::chunk_allocator::events::validate_configure_done` | `boost::sml::front::always` | `boost::sml::front::none` | `applying_configure` |
| `configuring` | `emel::buffer::chunk_allocator::events::validate_configure_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `applying_configure` | `emel::buffer::chunk_allocator::event::apply_configure` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_apply_configure>` | `applying_configure` |
| `applying_configure` | `emel::buffer::chunk_allocator::events::apply_configure_done` | `boost::sml::front::always` | `boost::sml::front::none` | `configure_done` |
| `applying_configure` | `emel::buffer::chunk_allocator::events::apply_configure_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `configure_done` | `emel::buffer::chunk_allocator::events::configure_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_configure_done>` | `ready` |
| `ready` | `emel::buffer::chunk_allocator::event::allocate` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_allocate>` | `validating_allocate` |
| `validating_allocate` | `emel::buffer::chunk_allocator::event::validate_allocate` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_validate_allocate>` | `validating_allocate` |
| `validating_allocate` | `emel::buffer::chunk_allocator::events::validate_allocate_done` | `boost::sml::front::always` | `boost::sml::front::none` | `selecting_block` |
| `validating_allocate` | `emel::buffer::chunk_allocator::events::validate_allocate_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `selecting_block` | `emel::buffer::chunk_allocator::event::select_block` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_select_block>` | `selecting_block` |
| `selecting_block` | `emel::buffer::chunk_allocator::events::select_block_done` | `boost::sml::front::always` | `boost::sml::front::none` | `ensuring_chunk` |
| `selecting_block` | `emel::buffer::chunk_allocator::events::select_block_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `ensuring_chunk` | `emel::buffer::chunk_allocator::event::ensure_chunk` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_ensure_chunk>` | `ensuring_chunk` |
| `ensuring_chunk` | `emel::buffer::chunk_allocator::events::ensure_chunk_done` | `boost::sml::front::always` | `boost::sml::front::none` | `committing_allocate` |
| `ensuring_chunk` | `emel::buffer::chunk_allocator::events::ensure_chunk_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `committing_allocate` | `emel::buffer::chunk_allocator::event::commit_allocate` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_commit_allocate>` | `committing_allocate` |
| `committing_allocate` | `emel::buffer::chunk_allocator::events::commit_allocate_done` | `boost::sml::front::always` | `boost::sml::front::none` | `allocate_done` |
| `committing_allocate` | `emel::buffer::chunk_allocator::events::commit_allocate_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `allocate_done` | `emel::buffer::chunk_allocator::events::allocate_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_allocate_done>` | `ready` |
| `ready` | `emel::buffer::chunk_allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_release>` | `validating_release` |
| `validating_release` | `emel::buffer::chunk_allocator::event::validate_release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_validate_release>` | `validating_release` |
| `validating_release` | `emel::buffer::chunk_allocator::events::validate_release_done` | `boost::sml::front::always` | `boost::sml::front::none` | `merging_release` |
| `validating_release` | `emel::buffer::chunk_allocator::events::validate_release_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `merging_release` | `emel::buffer::chunk_allocator::event::merge_release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_merge_release>` | `merging_release` |
| `merging_release` | `emel::buffer::chunk_allocator::events::merge_release_done` | `boost::sml::front::always` | `boost::sml::front::none` | `release_done` |
| `merging_release` | `emel::buffer::chunk_allocator::events::merge_release_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `release_done` | `emel::buffer::chunk_allocator::events::release_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_release_done>` | `ready` |
| `ready` | `emel::buffer::chunk_allocator::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>` | `resetting` |
| `configuring` | `emel::buffer::chunk_allocator::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>` | `resetting` |
| `applying_configure` | `emel::buffer::chunk_allocator::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>` | `resetting` |
| `configure_done` | `emel::buffer::chunk_allocator::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>` | `resetting` |
| `validating_allocate` | `emel::buffer::chunk_allocator::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>` | `resetting` |
| `selecting_block` | `emel::buffer::chunk_allocator::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>` | `resetting` |
| `ensuring_chunk` | `emel::buffer::chunk_allocator::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>` | `resetting` |
| `committing_allocate` | `emel::buffer::chunk_allocator::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>` | `resetting` |
| `allocate_done` | `emel::buffer::chunk_allocator::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>` | `resetting` |
| `validating_release` | `emel::buffer::chunk_allocator::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>` | `resetting` |
| `merging_release` | `emel::buffer::chunk_allocator::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>` | `resetting` |
| `release_done` | `emel::buffer::chunk_allocator::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>` | `resetting` |
| `failed` | `emel::buffer::chunk_allocator::event::reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_reset>` | `resetting` |
| `resetting` | `emel::buffer::chunk_allocator::event::apply_reset` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_apply_reset>` | `resetting` |
| `resetting` | `emel::buffer::chunk_allocator::events::apply_reset_done` | `boost::sml::front::always` | `boost::sml::front::none` | `reset_done` |
| `resetting` | `emel::buffer::chunk_allocator::events::apply_reset_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `reset_done` | `emel::buffer::chunk_allocator::events::reset_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_reset_done>` | `ready` |
| `failed` | `emel::buffer::chunk_allocator::events::configure_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_configure_error>` | `ready` |
| `failed` | `emel::buffer::chunk_allocator::events::allocate_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_allocate_error>` | `ready` |
| `failed` | `emel::buffer::chunk_allocator::events::release_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_release_error>` | `ready` |
| `failed` | `emel::buffer::chunk_allocator::events::reset_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_reset_error>` | `ready` |
