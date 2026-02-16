# buffer_chunk_allocator

Source: `emel/buffer/chunk_allocator/sm.hpp`

## Mermaid

```mermaid
stateDiagram-v2
  [*] --> ready
  ready --> configuring : emel::buffer::chunk_allocator::event::configure [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_configure>
  configuring --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:89:11)>
  configuring --> configuring : emel::buffer::chunk_allocator::event::validate_configure [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_validate_configure>
  configuring --> applying_configure : emel::buffer::chunk_allocator::events::validate_configure_done [boost::sml::front::always] / boost::sml::front::none
  configuring --> failed : emel::buffer::chunk_allocator::events::validate_configure_error [boost::sml::front::always] / boost::sml::front::none
  applying_configure --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:112:11)>
  applying_configure --> applying_configure : emel::buffer::chunk_allocator::event::apply_configure [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_apply_configure>
  applying_configure --> configure_done : emel::buffer::chunk_allocator::events::apply_configure_done [boost::sml::front::always] / boost::sml::front::none
  applying_configure --> failed : emel::buffer::chunk_allocator::events::apply_configure_error [boost::sml::front::always] / boost::sml::front::none
  configure_done --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:137:11)>
  configure_done --> ready : emel::buffer::chunk_allocator::events::configure_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_configure_done>
  ready --> validating_allocate : emel::buffer::chunk_allocator::event::allocate [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_allocate>
  validating_allocate --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:151:11)>
  validating_allocate --> validating_allocate : emel::buffer::chunk_allocator::event::validate_allocate [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_validate_allocate>
  validating_allocate --> selecting_block : emel::buffer::chunk_allocator::events::validate_allocate_done [boost::sml::front::always] / boost::sml::front::none
  validating_allocate --> failed : emel::buffer::chunk_allocator::events::validate_allocate_error [boost::sml::front::always] / boost::sml::front::none
  selecting_block --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:182:11)>
  selecting_block --> selecting_block : emel::buffer::chunk_allocator::event::select_block [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_select_block>
  selecting_block --> ensuring_chunk : emel::buffer::chunk_allocator::events::select_block_done [boost::sml::front::always] / boost::sml::front::none
  selecting_block --> failed : emel::buffer::chunk_allocator::events::select_block_error [boost::sml::front::always] / boost::sml::front::none
  ensuring_chunk --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:206:11)>
  ensuring_chunk --> ensuring_chunk : emel::buffer::chunk_allocator::event::ensure_chunk [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_ensure_chunk>
  ensuring_chunk --> committing_allocate : emel::buffer::chunk_allocator::events::ensure_chunk_done [boost::sml::front::always] / boost::sml::front::none
  ensuring_chunk --> failed : emel::buffer::chunk_allocator::events::ensure_chunk_error [boost::sml::front::always] / boost::sml::front::none
  committing_allocate --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:229:11)>
  committing_allocate --> committing_allocate : emel::buffer::chunk_allocator::event::commit_allocate [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_commit_allocate>
  committing_allocate --> allocate_done : emel::buffer::chunk_allocator::events::commit_allocate_done [boost::sml::front::always] / boost::sml::front::none
  committing_allocate --> failed : emel::buffer::chunk_allocator::events::commit_allocate_error [boost::sml::front::always] / boost::sml::front::none
  allocate_done --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:254:11)>
  allocate_done --> ready : emel::buffer::chunk_allocator::events::allocate_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_allocate_done>
  ready --> validating_release : emel::buffer::chunk_allocator::event::release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_release>
  validating_release --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:282:11)>
  validating_release --> validating_release : emel::buffer::chunk_allocator::event::validate_release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_validate_release>
  validating_release --> merging_release : emel::buffer::chunk_allocator::events::validate_release_done [boost::sml::front::always] / boost::sml::front::none
  validating_release --> failed : emel::buffer::chunk_allocator::events::validate_release_error [boost::sml::front::always] / boost::sml::front::none
  merging_release --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:306:11)>
  merging_release --> merging_release : emel::buffer::chunk_allocator::event::merge_release [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_merge_release>
  merging_release --> release_done : emel::buffer::chunk_allocator::events::merge_release_done [boost::sml::front::always] / boost::sml::front::none
  merging_release --> failed : emel::buffer::chunk_allocator::events::merge_release_error [boost::sml::front::always] / boost::sml::front::none
  release_done --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:330:11)>
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
  resetting --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:369:11)>
  reset_done --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:387:11)>
  reset_done --> ready : emel::buffer::chunk_allocator::events::reset_done [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_reset_done>
  failed --> boost::sml::front::internal : on_entry [boost::sml::front::always] / boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:398:11)>
  failed --> ready : emel::buffer::chunk_allocator::events::configure_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_configure_error>
  failed --> ready : emel::buffer::chunk_allocator::events::allocate_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_allocate_error>
  failed --> ready : emel::buffer::chunk_allocator::events::release_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_release_error>
  failed --> ready : emel::buffer::chunk_allocator::events::reset_error [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_reset_error>
  ready --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>
  configuring --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>
  applying_configure --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>
  configure_done --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>
  validating_allocate --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>
  selecting_block --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>
  ensuring_chunk --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>
  committing_allocate --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>
  allocate_done --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>
  validating_release --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>
  merging_release --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>
  release_done --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>
  resetting --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>
  reset_done --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>
  failed --> failed : boost::sml::back::_ [boost::sml::front::always] / boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>
```

## Transitions

| Source | Event | Guard | Action | Target |
| --- | --- | --- | --- | --- |
| `ready` | `emel::buffer::chunk_allocator::event::configure` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_configure>` | `configuring` |
| `configuring` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:89:11)>` | `boost::sml::front::internal` |
| `configuring` | `emel::buffer::chunk_allocator::event::validate_configure` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_validate_configure>` | `configuring` |
| `configuring` | `emel::buffer::chunk_allocator::events::validate_configure_done` | `boost::sml::front::always` | `boost::sml::front::none` | `applying_configure` |
| `configuring` | `emel::buffer::chunk_allocator::events::validate_configure_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `applying_configure` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:112:11)>` | `boost::sml::front::internal` |
| `applying_configure` | `emel::buffer::chunk_allocator::event::apply_configure` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_apply_configure>` | `applying_configure` |
| `applying_configure` | `emel::buffer::chunk_allocator::events::apply_configure_done` | `boost::sml::front::always` | `boost::sml::front::none` | `configure_done` |
| `applying_configure` | `emel::buffer::chunk_allocator::events::apply_configure_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `configure_done` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:137:11)>` | `boost::sml::front::internal` |
| `configure_done` | `emel::buffer::chunk_allocator::events::configure_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_configure_done>` | `ready` |
| `ready` | `emel::buffer::chunk_allocator::event::allocate` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_allocate>` | `validating_allocate` |
| `validating_allocate` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:151:11)>` | `boost::sml::front::internal` |
| `validating_allocate` | `emel::buffer::chunk_allocator::event::validate_allocate` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_validate_allocate>` | `validating_allocate` |
| `validating_allocate` | `emel::buffer::chunk_allocator::events::validate_allocate_done` | `boost::sml::front::always` | `boost::sml::front::none` | `selecting_block` |
| `validating_allocate` | `emel::buffer::chunk_allocator::events::validate_allocate_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `selecting_block` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:182:11)>` | `boost::sml::front::internal` |
| `selecting_block` | `emel::buffer::chunk_allocator::event::select_block` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_select_block>` | `selecting_block` |
| `selecting_block` | `emel::buffer::chunk_allocator::events::select_block_done` | `boost::sml::front::always` | `boost::sml::front::none` | `ensuring_chunk` |
| `selecting_block` | `emel::buffer::chunk_allocator::events::select_block_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `ensuring_chunk` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:206:11)>` | `boost::sml::front::internal` |
| `ensuring_chunk` | `emel::buffer::chunk_allocator::event::ensure_chunk` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_ensure_chunk>` | `ensuring_chunk` |
| `ensuring_chunk` | `emel::buffer::chunk_allocator::events::ensure_chunk_done` | `boost::sml::front::always` | `boost::sml::front::none` | `committing_allocate` |
| `ensuring_chunk` | `emel::buffer::chunk_allocator::events::ensure_chunk_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `committing_allocate` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:229:11)>` | `boost::sml::front::internal` |
| `committing_allocate` | `emel::buffer::chunk_allocator::event::commit_allocate` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_commit_allocate>` | `committing_allocate` |
| `committing_allocate` | `emel::buffer::chunk_allocator::events::commit_allocate_done` | `boost::sml::front::always` | `boost::sml::front::none` | `allocate_done` |
| `committing_allocate` | `emel::buffer::chunk_allocator::events::commit_allocate_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `allocate_done` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:254:11)>` | `boost::sml::front::internal` |
| `allocate_done` | `emel::buffer::chunk_allocator::events::allocate_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_allocate_done>` | `ready` |
| `ready` | `emel::buffer::chunk_allocator::event::release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::begin_release>` | `validating_release` |
| `validating_release` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:282:11)>` | `boost::sml::front::internal` |
| `validating_release` | `emel::buffer::chunk_allocator::event::validate_release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_validate_release>` | `validating_release` |
| `validating_release` | `emel::buffer::chunk_allocator::events::validate_release_done` | `boost::sml::front::always` | `boost::sml::front::none` | `merging_release` |
| `validating_release` | `emel::buffer::chunk_allocator::events::validate_release_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `merging_release` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:306:11)>` | `boost::sml::front::internal` |
| `merging_release` | `emel::buffer::chunk_allocator::event::merge_release` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::run_merge_release>` | `merging_release` |
| `merging_release` | `emel::buffer::chunk_allocator::events::merge_release_done` | `boost::sml::front::always` | `boost::sml::front::none` | `release_done` |
| `merging_release` | `emel::buffer::chunk_allocator::events::merge_release_error` | `boost::sml::front::always` | `boost::sml::front::none` | `failed` |
| `release_done` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:330:11)>` | `boost::sml::front::internal` |
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
| `resetting` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:369:11)>` | `boost::sml::front::internal` |
| `reset_done` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:387:11)>` | `boost::sml::front::internal` |
| `reset_done` | `emel::buffer::chunk_allocator::events::reset_done` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_reset_done>` | `ready` |
| `failed` | `on_entry` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<(lambda at /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/buffer/chunk_allocator/sm.hpp:398:11)>` | `boost::sml::front::internal` |
| `failed` | `emel::buffer::chunk_allocator::events::configure_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_configure_error>` | `ready` |
| `failed` | `emel::buffer::chunk_allocator::events::allocate_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_allocate_error>` | `ready` |
| `failed` | `emel::buffer::chunk_allocator::events::release_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_release_error>` | `ready` |
| `failed` | `emel::buffer::chunk_allocator::events::reset_error` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_reset_error>` | `ready` |
| `ready` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>` | `failed` |
| `configuring` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>` | `failed` |
| `applying_configure` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>` | `failed` |
| `configure_done` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>` | `failed` |
| `validating_allocate` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>` | `failed` |
| `selecting_block` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>` | `failed` |
| `ensuring_chunk` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>` | `failed` |
| `committing_allocate` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>` | `failed` |
| `allocate_done` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>` | `failed` |
| `validating_release` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>` | `failed` |
| `merging_release` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>` | `failed` |
| `release_done` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>` | `failed` |
| `resetting` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>` | `failed` |
| `reset_done` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>` | `failed` |
| `failed` | `boost::sml::back::_` | `boost::sml::front::always` | `boost::sml::aux::zero_wrapper<emel::buffer::chunk_allocator::action::on_unexpected>` | `failed` |
