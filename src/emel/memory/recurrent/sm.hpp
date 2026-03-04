#pragma once
// benchmark: designed

#include <array>
#include <cstdint>
#include <memory>

#include "emel/memory/detail.hpp"
#include "emel/memory/recurrent/actions.hpp"
#include "emel/memory/recurrent/context.hpp"
#include "emel/memory/recurrent/events.hpp"
#include "emel/memory/recurrent/guards.hpp"
#include "emel/memory/view.hpp"
#include "emel/sm.hpp"

namespace emel::memory::recurrent {

struct ready {};

struct reserve_request_decision {};
struct reserve_exec {};
struct reserve_result_decision {};

struct allocate_sequence_request_decision {};
struct allocate_sequence_exec {};
struct allocate_sequence_result_decision {};

struct allocate_slots_request_decision {};
struct allocate_slots_request_shape_decision {};
struct allocate_slots_request_length_decision {};
struct allocate_slots_exec {};
struct allocate_slots_result_decision {};

struct branch_sequence_request_decision {};
struct branch_sequence_request_shape_decision {};
struct branch_sequence_request_capacity_decision {};
struct branch_sequence_exec {};
struct branch_sequence_result_decision {};
struct branch_sequence_copy_exec {};
struct branch_sequence_copy_result_decision {};
struct branch_sequence_rollback_exec {};

struct free_sequence_request_decision {};
struct free_sequence_exec {};
struct free_sequence_result_decision {};

struct rollback_slots_request_decision {};
struct rollback_slots_exec {};
struct rollback_slots_result_decision {};

struct capture_request_decision {};
struct capture_exec {};
struct capture_result_decision {};

struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<reserve_request_decision> <= *sml::state<ready> + sml::event<event::reserve_runtime>
          / action::begin_reserve
      , sml::state<reserve_exec> <= sml::state<reserve_request_decision>
          + sml::completion<event::reserve_runtime> [ guard::reserve_request_valid{} ]
      , sml::state<errored> <= sml::state<reserve_request_decision>
          + sml::completion<event::reserve_runtime> [ guard::reserve_request_invalid{} ]
          / action::mark_invalid_request
      , sml::state<reserve_result_decision> <= sml::state<reserve_exec>
          + sml::completion<event::reserve_runtime> / action::exec_reserve
      , sml::state<done> <= sml::state<reserve_result_decision>
          + sml::completion<event::reserve_runtime> [ guard::operation_succeeded{} ]
      , sml::state<errored> <= sml::state<reserve_result_decision>
          + sml::completion<event::reserve_runtime> [ guard::operation_failed_with_error{} ]
          / action::mark_error_from_operation
      , sml::state<errored> <= sml::state<reserve_result_decision>
          + sml::completion<event::reserve_runtime> [ guard::operation_failed_without_error{} ]
          / action::mark_backend_error

      //------------------------------------------------------------------------------//
      , sml::state<allocate_sequence_request_decision> <= sml::state<ready>
          + sml::event<event::allocate_sequence_runtime> / action::begin_allocate_sequence
      , sml::state<allocate_sequence_exec> <= sml::state<allocate_sequence_request_decision>
          + sml::completion<event::allocate_sequence_runtime>
          [ guard::allocate_sequence_request_inactive_with_slot{} ]
      , sml::state<done> <= sml::state<allocate_sequence_request_decision>
          + sml::completion<event::allocate_sequence_runtime>
          [ guard::allocate_sequence_request_active{} ]
          / action::mark_operation_success
      , sml::state<errored> <= sml::state<allocate_sequence_request_decision>
          + sml::completion<event::allocate_sequence_runtime>
          [ guard::allocate_sequence_request_inactive_without_slot{} ]
          / action::mark_backend_error
      , sml::state<errored> <= sml::state<allocate_sequence_request_decision>
          + sml::completion<event::allocate_sequence_runtime>
          [ guard::allocate_sequence_request_invalid{} ]
          / action::mark_invalid_request
      , sml::state<allocate_sequence_result_decision> <= sml::state<allocate_sequence_exec>
          + sml::completion<event::allocate_sequence_runtime>
          / action::exec_allocate_sequence_inactive
      , sml::state<done> <= sml::state<allocate_sequence_result_decision>
          + sml::completion<event::allocate_sequence_runtime> [ guard::operation_succeeded{} ]
      , sml::state<errored> <= sml::state<allocate_sequence_result_decision>
          + sml::completion<event::allocate_sequence_runtime> [ guard::operation_failed_with_error{} ]
          / action::mark_error_from_operation
      , sml::state<errored> <= sml::state<allocate_sequence_result_decision>
          + sml::completion<event::allocate_sequence_runtime> [ guard::operation_failed_without_error{} ]
          / action::mark_backend_error

      //------------------------------------------------------------------------------//
      , sml::state<allocate_slots_request_decision> <= sml::state<ready>
          + sml::event<event::allocate_slots_runtime> / action::begin_allocate_slots
      , sml::state<allocate_slots_request_shape_decision> <= sml::state<allocate_slots_request_decision>
          + sml::completion<event::allocate_slots_runtime>

      , sml::state<allocate_slots_request_length_decision>
          <= sml::state<allocate_slots_request_shape_decision>
          + sml::completion<event::allocate_slots_runtime>
          [ guard::allocate_slots_request_shape_valid{} ]
      , sml::state<errored> <= sml::state<allocate_slots_request_shape_decision>
          + sml::completion<event::allocate_slots_runtime>
          [ guard::allocate_slots_request_shape_invalid{} ]
          / action::mark_invalid_request

      , sml::state<allocate_slots_exec> <= sml::state<allocate_slots_request_length_decision>
          + sml::completion<event::allocate_slots_runtime>
          [ guard::allocate_slots_request_length_valid{} ]
      , sml::state<errored> <= sml::state<allocate_slots_request_length_decision>
          + sml::completion<event::allocate_slots_runtime>
          [ guard::allocate_slots_request_length_invalid{} ]
          / action::mark_invalid_request
      , sml::state<allocate_slots_result_decision> <= sml::state<allocate_slots_exec>
          + sml::completion<event::allocate_slots_runtime> / action::exec_allocate_slots
      , sml::state<done> <= sml::state<allocate_slots_result_decision>
          + sml::completion<event::allocate_slots_runtime> [ guard::operation_succeeded{} ]
      , sml::state<errored> <= sml::state<allocate_slots_result_decision>
          + sml::completion<event::allocate_slots_runtime> [ guard::operation_failed_with_error{} ]
          / action::mark_error_from_operation
      , sml::state<errored> <= sml::state<allocate_slots_result_decision>
          + sml::completion<event::allocate_slots_runtime> [ guard::operation_failed_without_error{} ]
          / action::mark_backend_error

      //------------------------------------------------------------------------------//
      , sml::state<branch_sequence_request_decision> <= sml::state<ready>
          + sml::event<event::branch_sequence_runtime> / action::begin_branch_sequence
      , sml::state<branch_sequence_request_shape_decision>
          <= sml::state<branch_sequence_request_decision>
          + sml::completion<event::branch_sequence_runtime>
      , sml::state<branch_sequence_request_capacity_decision>
          <= sml::state<branch_sequence_request_shape_decision>
          + sml::completion<event::branch_sequence_runtime>
          [ guard::branch_sequence_request_shape_valid{} ]
      , sml::state<errored> <= sml::state<branch_sequence_request_shape_decision>
          + sml::completion<event::branch_sequence_runtime>
          [ guard::branch_sequence_request_shape_invalid{} ]
          / action::mark_invalid_request
      , sml::state<branch_sequence_exec> <= sml::state<branch_sequence_request_capacity_decision>
          + sml::completion<event::branch_sequence_runtime>
          [ guard::branch_sequence_request_capacity_available{} ]
      , sml::state<errored> <= sml::state<branch_sequence_request_capacity_decision>
          + sml::completion<event::branch_sequence_runtime>
          [ guard::branch_sequence_request_capacity_exhausted{} ]
          / action::mark_backend_error
      , sml::state<branch_sequence_result_decision> <= sml::state<branch_sequence_exec>
          + sml::completion<event::branch_sequence_runtime>
          / action::exec_branch_sequence_prepare_child_slot
      , sml::state<branch_sequence_copy_exec> <= sml::state<branch_sequence_result_decision>
          + sml::completion<event::branch_sequence_runtime>
          [ guard::branch_slot_activation_succeeded{} ]
      , sml::state<errored> <= sml::state<branch_sequence_result_decision>
          + sml::completion<event::branch_sequence_runtime>
          [ guard::branch_slot_activation_failed{} ]
          / action::mark_backend_error
      , sml::state<branch_sequence_copy_result_decision> <= sml::state<branch_sequence_copy_exec>
          + sml::completion<event::branch_sequence_runtime>
          / action::exec_branch_sequence_copy_callback
      , sml::state<done> <= sml::state<branch_sequence_copy_result_decision>
          + sml::completion<event::branch_sequence_runtime>
          [ guard::operation_succeeded{} ]
          / action::finalize_branch_sequence_success
      , sml::state<branch_sequence_rollback_exec> <= sml::state<branch_sequence_copy_result_decision>
          + sml::completion<event::branch_sequence_runtime>
          [ guard::operation_failed_with_error{} ]
          / action::mark_error_from_operation
      , sml::state<branch_sequence_rollback_exec> <= sml::state<branch_sequence_copy_result_decision>
          + sml::completion<event::branch_sequence_runtime>
          [ guard::operation_failed_without_error{} ]
          / action::mark_backend_error
      , sml::state<errored> <= sml::state<branch_sequence_rollback_exec>
          + sml::completion<event::branch_sequence_runtime>
          / action::exec_branch_sequence_rollback_child_slot

      //------------------------------------------------------------------------------//
      , sml::state<free_sequence_request_decision> <= sml::state<ready>
          + sml::event<event::free_sequence_runtime> / action::begin_free_sequence
      , sml::state<free_sequence_exec> <= sml::state<free_sequence_request_decision>
          + sml::completion<event::free_sequence_runtime>
          [ guard::free_sequence_request_active{} ]
      , sml::state<done> <= sml::state<free_sequence_request_decision>
          + sml::completion<event::free_sequence_runtime>
          [ guard::free_sequence_request_inactive{} ]
          / action::mark_operation_success
      , sml::state<errored> <= sml::state<free_sequence_request_decision>
          + sml::completion<event::free_sequence_runtime> [ guard::free_sequence_request_invalid{} ]
          / action::mark_invalid_request
      , sml::state<free_sequence_result_decision> <= sml::state<free_sequence_exec>
          + sml::completion<event::free_sequence_runtime> / action::exec_free_sequence_active
      , sml::state<done> <= sml::state<free_sequence_result_decision>
          + sml::completion<event::free_sequence_runtime> [ guard::operation_succeeded{} ]
      , sml::state<errored> <= sml::state<free_sequence_result_decision>
          + sml::completion<event::free_sequence_runtime> [ guard::operation_failed_with_error{} ]
          / action::mark_error_from_operation
      , sml::state<errored> <= sml::state<free_sequence_result_decision>
          + sml::completion<event::free_sequence_runtime> [ guard::operation_failed_without_error{} ]
          / action::mark_backend_error

      //------------------------------------------------------------------------------//
      , sml::state<rollback_slots_request_decision> <= sml::state<ready>
          + sml::event<event::rollback_slots_runtime> / action::begin_rollback_slots
      , sml::state<rollback_slots_exec> <= sml::state<rollback_slots_request_decision>
          + sml::completion<event::rollback_slots_runtime>
          [ guard::rollback_slots_request_valid{} ]
      , sml::state<errored> <= sml::state<rollback_slots_request_decision>
          + sml::completion<event::rollback_slots_runtime>
          [ guard::rollback_slots_request_invalid{} ]
          / action::mark_invalid_request
      , sml::state<rollback_slots_result_decision> <= sml::state<rollback_slots_exec>
          + sml::completion<event::rollback_slots_runtime> / action::exec_rollback_slots
      , sml::state<done> <= sml::state<rollback_slots_result_decision>
          + sml::completion<event::rollback_slots_runtime> [ guard::operation_succeeded{} ]
      , sml::state<errored> <= sml::state<rollback_slots_result_decision>
          + sml::completion<event::rollback_slots_runtime> [ guard::operation_failed_with_error{} ]
          / action::mark_error_from_operation
      , sml::state<errored> <= sml::state<rollback_slots_result_decision>
          + sml::completion<event::rollback_slots_runtime> [ guard::operation_failed_without_error{} ]
          / action::mark_backend_error

      //------------------------------------------------------------------------------//
      , sml::state<capture_request_decision> <= sml::state<ready>
          + sml::event<event::capture_view_runtime> / action::begin_capture_view
      , sml::state<capture_exec> <= sml::state<capture_request_decision>
          + sml::completion<event::capture_view_runtime> [ guard::capture_request_valid{} ]
      , sml::state<errored> <= sml::state<capture_request_decision>
          + sml::completion<event::capture_view_runtime> [ guard::capture_request_invalid{} ]
          / action::mark_invalid_request
      , sml::state<capture_result_decision> <= sml::state<capture_exec>
          + sml::completion<event::capture_view_runtime> / action::exec_capture_view
      , sml::state<done> <= sml::state<capture_result_decision>
          + sml::completion<event::capture_view_runtime> [ guard::operation_succeeded{} ]
      , sml::state<errored> <= sml::state<capture_result_decision>
          + sml::completion<event::capture_view_runtime> [ guard::operation_failed_with_error{} ]
          / action::mark_error_from_operation
      , sml::state<errored> <= sml::state<capture_result_decision>
          + sml::completion<event::capture_view_runtime> [ guard::operation_failed_without_error{} ]
          / action::mark_backend_error

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<done> + sml::completion<event::reserve_runtime>
          / action::publish_done
      , sml::state<ready> <= sml::state<errored> + sml::completion<event::reserve_runtime>
          / action::publish_error
      , sml::state<ready> <= sml::state<done>
          + sml::completion<event::allocate_sequence_runtime> / action::publish_done
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<event::allocate_sequence_runtime> / action::publish_error
      , sml::state<ready> <= sml::state<done>
          + sml::completion<event::allocate_slots_runtime> / action::publish_done
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<event::allocate_slots_runtime> / action::publish_error
      , sml::state<ready> <= sml::state<done>
          + sml::completion<event::branch_sequence_runtime> / action::publish_done
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<event::branch_sequence_runtime> / action::publish_error
      , sml::state<ready> <= sml::state<done>
          + sml::completion<event::free_sequence_runtime> / action::publish_done
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<event::free_sequence_runtime> / action::publish_error
      , sml::state<ready> <= sml::state<done>
          + sml::completion<event::rollback_slots_runtime> / action::publish_done
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<event::rollback_slots_runtime> / action::publish_error
      , sml::state<ready> <= sml::state<done>
          + sml::completion<event::capture_view_runtime> / action::publish_done
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<event::capture_view_runtime> / action::publish_error

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<reserve_request_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<reserve_exec> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<reserve_result_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_sequence_request_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_sequence_exec> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_sequence_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_slots_request_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_slots_request_shape_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_slots_request_length_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_slots_exec> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<allocate_slots_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<branch_sequence_request_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<branch_sequence_request_shape_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<branch_sequence_request_capacity_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<branch_sequence_exec> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<branch_sequence_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<branch_sequence_copy_exec> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<branch_sequence_copy_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<branch_sequence_rollback_exec>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<free_sequence_request_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<free_sequence_exec> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<free_sequence_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<rollback_slots_request_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<rollback_slots_exec> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<rollback_slots_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<capture_request_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<capture_exec> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<capture_result_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<done> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<errored> + sml::unexpected_event<sml::_>
          / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  sm() : base_type(), snapshot_(std::make_unique<view::snapshot>()) {}

  bool process_event(const event::reserve & ev) {
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    event::reserve_ctx ctx{};
    event::reserve_runtime runtime{
        ev,
        ctx,
        emel::memory::detail::bind_or_sink(ev.error_out, error_sink)};
    const bool accepted = base_type::process_event(runtime);
    snapshot_dirty_ = true;
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::allocate_sequence & ev) {
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    event::allocate_sequence_ctx ctx{};
    event::allocate_sequence_runtime runtime{
        ev,
        ctx,
        emel::memory::detail::bind_or_sink(ev.error_out, error_sink)};
    const bool accepted = base_type::process_event(runtime);
    snapshot_dirty_ = true;
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::allocate_slots & ev) {
    int32_t block_count_sink = 0;
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    event::allocate_slots_ctx ctx{};
    event::allocate_slots_runtime runtime{
        ev,
        ctx,
        emel::memory::detail::bind_or_sink(ev.block_count_out, block_count_sink),
        emel::memory::detail::bind_or_sink(ev.error_out, error_sink)};
    const bool accepted = base_type::process_event(runtime);
    snapshot_dirty_ = true;
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::branch_sequence & ev) {
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    event::branch_sequence_ctx ctx{};
    event::branch_sequence_runtime runtime{
        ev,
        ctx,
        emel::memory::detail::bind_or_sink(ev.error_out, error_sink)};
    const bool accepted = base_type::process_event(runtime);
    snapshot_dirty_ = true;
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::free_sequence & ev) {
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    event::free_sequence_ctx ctx{};
    event::free_sequence_runtime runtime{
        ev,
        ctx,
        emel::memory::detail::bind_or_sink(ev.error_out, error_sink)};
    const bool accepted = base_type::process_event(runtime);
    snapshot_dirty_ = true;
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::rollback_slots & ev) {
    int32_t block_count_sink = 0;
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    event::rollback_slots_ctx ctx{};
    event::rollback_slots_runtime runtime{
        ev,
        ctx,
        emel::memory::detail::bind_or_sink(ev.block_count_out, block_count_sink),
        emel::memory::detail::bind_or_sink(ev.error_out, error_sink)};
    const bool accepted = base_type::process_event(runtime);
    snapshot_dirty_ = true;
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::capture_view & ev) {
    view::snapshot & snapshot_out =
        emel::memory::detail::bind_or_sink(ev.snapshot_out, *snapshot_);
    const bool writes_cached_snapshot = &snapshot_out == snapshot_.get();
    const bool has_snapshot_out = ev.snapshot_out != nullptr;
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    event::capture_view_ctx ctx{};
    event::capture_view_runtime runtime{
        ev,
        ctx,
        snapshot_out,
        emel::memory::detail::bind_or_sink(ev.error_out, error_sink),
        has_snapshot_out};
    const bool accepted = base_type::process_event(runtime);
    const bool ok = accepted && ctx.err == emel::error::cast(error::none);
    snapshot_dirty_ = snapshot_dirty_ && !(ok && writes_cached_snapshot);
    return ok;
  }

  bool try_view(view::snapshot & snapshot_out, emel::error::type & err_out) noexcept {
    int32_t err = static_cast<int32_t>(emel::error::cast(error::none));
    const bool accepted = process_event(event::capture_view{
      .snapshot_out = &snapshot_out,
      .error_out = &err,
    });
    err_out = static_cast<emel::error::type>(err);
    return accepted && err_out == emel::error::cast(error::none);
  }

  const view::snapshot & view() noexcept {
    const int32_t was_dirty = static_cast<int32_t>(snapshot_dirty_);
    emel::error::type err = emel::error::cast(error::none);
    const std::array<bool (*)(sm &, view::snapshot &, emel::error::type &) noexcept, 2> refreshers{
        &sm::refresh_noop,
        &sm::refresh_try_view,
    };
    const bool refreshed =
        refreshers[static_cast<size_t>(snapshot_dirty_)](*this, *snapshot_, err);
    const int32_t refreshed_int = static_cast<int32_t>(refreshed);
    const int32_t refresh_failed = was_dirty * (1 - refreshed_int);
    const std::array<void (*)(view::snapshot &) noexcept, 2> clearers{
        &emel::memory::detail::clear_snapshot_noop,
        &emel::memory::detail::clear_snapshot,
    };
    clearers[static_cast<size_t>(refresh_failed)](*snapshot_);
    snapshot_dirty_ = static_cast<bool>(refresh_failed);
    return *snapshot_;
  }

 private:
  static bool refresh_noop(sm &, view::snapshot &, emel::error::type &) noexcept { return true; }

  static bool refresh_try_view(sm & self, view::snapshot & snapshot_out,
                               emel::error::type & err_out) noexcept {
    return self.try_view(snapshot_out, err_out);
  }

  std::unique_ptr<view::snapshot> snapshot_;
  bool snapshot_dirty_ = true;
};

using Recurrent = sm;

}  // namespace emel::memory::recurrent
