#pragma once
// benchmark: designed

#include "emel/sm.hpp"
#include "emel/tensor/actions.hpp"
#include "emel/tensor/context.hpp"
#include "emel/tensor/errors.hpp"
#include "emel/tensor/events.hpp"
#include "emel/tensor/guards.hpp"

namespace emel::tensor {

struct ready {};

struct reserve_tensor_request_decision {};
struct reserve_tensor_exec {};
struct reserve_tensor_result_decision {};

struct publish_filled_tensor_request_decision {};
struct publish_filled_tensor_exec {};
struct publish_filled_tensor_result_decision {};

struct release_tensor_ref_request_decision {};
struct release_tensor_ref_exec {};
struct release_tensor_ref_result_decision {};

struct reset_tensor_epoch_request_decision {};
struct reset_tensor_epoch_exec {};
struct reset_tensor_epoch_result_decision {};

struct capture_tensor_state_request_decision {};
struct capture_tensor_state_exec {};
struct capture_tensor_state_result_decision {};

struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<reserve_tensor_request_decision> <= *sml::state<ready>
          + sml::event<detail::reserve_tensor_runtime> / action::begin_reserve_tensor
      , sml::state<reserve_tensor_exec> <= sml::state<reserve_tensor_request_decision>
          + sml::completion<detail::reserve_tensor_runtime> [ guard::reserve_tensor_request_valid{} ]
      , sml::state<errored> <= sml::state<reserve_tensor_request_decision>
          + sml::completion<detail::reserve_tensor_runtime> [ guard::reserve_tensor_request_invalid{} ]
          / action::mark_invalid_request
      , sml::state<reserve_tensor_result_decision> <= sml::state<reserve_tensor_exec>
          + sml::completion<detail::reserve_tensor_runtime> / action::exec_reserve_tensor
      , sml::state<done> <= sml::state<reserve_tensor_result_decision>
          + sml::completion<detail::reserve_tensor_runtime> [ guard::operation_succeeded{} ]
      , sml::state<errored> <= sml::state<reserve_tensor_result_decision>
          + sml::completion<detail::reserve_tensor_runtime> [ guard::operation_failed_internal{} ]
          / action::mark_internal_error
      , sml::state<errored> <= sml::state<reserve_tensor_result_decision>
          + sml::completion<detail::reserve_tensor_runtime>
          [ guard::operation_not_dispatched{} ]
          / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<publish_filled_tensor_request_decision> <= sml::state<ready>
          + sml::event<detail::publish_filled_tensor_runtime> / action::begin_publish_filled_tensor
      , sml::state<publish_filled_tensor_exec> <= sml::state<publish_filled_tensor_request_decision>
          + sml::completion<detail::publish_filled_tensor_runtime>
          [ guard::publish_filled_tensor_request_valid{} ]
      , sml::state<errored> <= sml::state<publish_filled_tensor_request_decision>
          + sml::completion<detail::publish_filled_tensor_runtime>
          [ guard::publish_filled_tensor_request_invalid{} ]
          / action::mark_invalid_request
      , sml::state<publish_filled_tensor_result_decision> <= sml::state<publish_filled_tensor_exec>
          + sml::completion<detail::publish_filled_tensor_runtime>
          / action::exec_publish_filled_tensor
      , sml::state<done> <= sml::state<publish_filled_tensor_result_decision>
          + sml::completion<detail::publish_filled_tensor_runtime> [ guard::operation_succeeded{} ]
      , sml::state<errored> <= sml::state<publish_filled_tensor_result_decision>
          + sml::completion<detail::publish_filled_tensor_runtime>
          [ guard::operation_failed_internal{} ]
          / action::mark_internal_error
      , sml::state<errored> <= sml::state<publish_filled_tensor_result_decision>
          + sml::completion<detail::publish_filled_tensor_runtime>
          [ guard::operation_not_dispatched{} ]
          / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<release_tensor_ref_request_decision> <= sml::state<ready>
          + sml::event<detail::release_tensor_ref_runtime> / action::begin_release_tensor_ref
      , sml::state<release_tensor_ref_exec> <= sml::state<release_tensor_ref_request_decision>
          + sml::completion<detail::release_tensor_ref_runtime>
          [ guard::release_tensor_ref_request_valid{} ]
      , sml::state<errored> <= sml::state<release_tensor_ref_request_decision>
          + sml::completion<detail::release_tensor_ref_runtime>
          [ guard::release_tensor_ref_request_invalid{} ]
          / action::mark_invalid_request
      , sml::state<release_tensor_ref_result_decision> <= sml::state<release_tensor_ref_exec>
          + sml::completion<detail::release_tensor_ref_runtime> / action::exec_release_tensor_ref
      , sml::state<done> <= sml::state<release_tensor_ref_result_decision>
          + sml::completion<detail::release_tensor_ref_runtime> [ guard::operation_succeeded{} ]
      , sml::state<errored> <= sml::state<release_tensor_ref_result_decision>
          + sml::completion<detail::release_tensor_ref_runtime>
          [ guard::operation_failed_internal{} ]
          / action::mark_internal_error
      , sml::state<errored> <= sml::state<release_tensor_ref_result_decision>
          + sml::completion<detail::release_tensor_ref_runtime>
          [ guard::operation_not_dispatched{} ]
          / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<reset_tensor_epoch_request_decision> <= sml::state<ready>
          + sml::event<detail::reset_tensor_epoch_runtime> / action::begin_reset_tensor_epoch
      , sml::state<reset_tensor_epoch_exec> <= sml::state<reset_tensor_epoch_request_decision>
          + sml::completion<detail::reset_tensor_epoch_runtime>
          [ guard::reset_tensor_epoch_request_valid{} ]
      , sml::state<errored> <= sml::state<reset_tensor_epoch_request_decision>
          + sml::completion<detail::reset_tensor_epoch_runtime>
          [ guard::reset_tensor_epoch_request_invalid{} ]
          / action::mark_invalid_request
      , sml::state<reset_tensor_epoch_result_decision> <= sml::state<reset_tensor_epoch_exec>
          + sml::completion<detail::reset_tensor_epoch_runtime> / action::exec_reset_tensor_epoch
      , sml::state<done> <= sml::state<reset_tensor_epoch_result_decision>
          + sml::completion<detail::reset_tensor_epoch_runtime> [ guard::operation_succeeded{} ]
      , sml::state<errored> <= sml::state<reset_tensor_epoch_result_decision>
          + sml::completion<detail::reset_tensor_epoch_runtime>
          [ guard::operation_failed_internal{} ]
          / action::mark_internal_error
      , sml::state<errored> <= sml::state<reset_tensor_epoch_result_decision>
          + sml::completion<detail::reset_tensor_epoch_runtime>
          [ guard::operation_not_dispatched{} ]
          / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<capture_tensor_state_request_decision> <= sml::state<ready>
          + sml::event<detail::capture_tensor_state_runtime> / action::begin_capture_tensor_state
      , sml::state<capture_tensor_state_exec> <= sml::state<capture_tensor_state_request_decision>
          + sml::completion<detail::capture_tensor_state_runtime>
          [ guard::capture_tensor_state_request_valid{} ]
      , sml::state<errored> <= sml::state<capture_tensor_state_request_decision>
          + sml::completion<detail::capture_tensor_state_runtime>
          [ guard::capture_tensor_state_request_invalid{} ]
          / action::mark_invalid_request
      , sml::state<capture_tensor_state_result_decision> <= sml::state<capture_tensor_state_exec>
          + sml::completion<detail::capture_tensor_state_runtime>
          / action::exec_capture_tensor_state
      , sml::state<done> <= sml::state<capture_tensor_state_result_decision>
          + sml::completion<detail::capture_tensor_state_runtime>
          [ guard::capture_operation_succeeded{} ]
      , sml::state<errored> <= sml::state<capture_tensor_state_result_decision>
          + sml::completion<detail::capture_tensor_state_runtime>
          [ guard::capture_operation_not_dispatched{} ]
          / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<done> + sml::completion<detail::reserve_tensor_runtime>
          / action::publish_done
      , sml::state<ready> <= sml::state<errored> + sml::completion<detail::reserve_tensor_runtime>
          / action::publish_error
      , sml::state<ready> <= sml::state<done>
          + sml::completion<detail::publish_filled_tensor_runtime> / action::publish_done
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<detail::publish_filled_tensor_runtime> / action::publish_error
      , sml::state<ready> <= sml::state<done>
          + sml::completion<detail::release_tensor_ref_runtime> / action::publish_done
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<detail::release_tensor_ref_runtime> / action::publish_error
      , sml::state<ready> <= sml::state<done>
          + sml::completion<detail::reset_tensor_epoch_runtime> / action::publish_done
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<detail::reset_tensor_epoch_runtime> / action::publish_error
      , sml::state<ready> <= sml::state<done>
          + sml::completion<detail::capture_tensor_state_runtime> / action::publish_done
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<detail::capture_tensor_state_runtime> / action::publish_error

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<reserve_tensor_request_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<reserve_tensor_exec> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<reserve_tensor_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<publish_filled_tensor_request_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<publish_filled_tensor_exec> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<publish_filled_tensor_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<release_tensor_ref_request_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<release_tensor_ref_exec> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<release_tensor_ref_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<reset_tensor_epoch_request_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<reset_tensor_epoch_exec> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<reset_tensor_epoch_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<capture_tensor_state_request_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<capture_tensor_state_exec> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<capture_tensor_state_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
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

  bool process_event(const event::reserve_tensor & ev) {
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    detail::runtime_status ctx{};
    detail::reserve_tensor_runtime runtime{
      ev,
      ctx,
      tensor::detail::bind_or_sink(ev.error_out, error_sink),
    };
    base_type::process_event(runtime);
    return ctx.ok;
  }

  bool process_event(const event::publish_filled_tensor & ev) {
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    detail::runtime_status ctx{};
    detail::publish_filled_tensor_runtime runtime{
      ev,
      ctx,
      tensor::detail::bind_or_sink(ev.error_out, error_sink),
    };
    base_type::process_event(runtime);
    return ctx.ok;
  }

  bool process_event(const event::release_tensor_ref & ev) {
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    detail::runtime_status ctx{};
    detail::release_tensor_ref_runtime runtime{
      ev,
      ctx,
      tensor::detail::bind_or_sink(ev.error_out, error_sink),
    };
    base_type::process_event(runtime);
    return ctx.ok;
  }

  bool process_event(const event::reset_tensor_epoch & ev) {
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    detail::runtime_status ctx{};
    detail::reset_tensor_epoch_runtime runtime{
      ev,
      ctx,
      tensor::detail::bind_or_sink(ev.error_out, error_sink),
    };
    base_type::process_event(runtime);
    return ctx.ok;
  }

  bool process_event(const event::capture_tensor_state & ev) {
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    detail::runtime_status ctx{};
    detail::capture_tensor_state_runtime runtime{
      ev,
      ctx,
      tensor::detail::bind_or_sink(ev.error_out, error_sink),
    };
    base_type::process_event(runtime);
    return ctx.ok;
  }

  bool try_capture(const int32_t tensor_id, event::tensor_state & state_out,
                   emel::error::type & err_out) noexcept {
    int32_t err = static_cast<int32_t>(emel::error::cast(error::none));
    const bool ok = process_event(event::capture_tensor_state{
      .tensor_id = tensor_id,
      .state_out = &state_out,
      .error_out = &err,
    });
    err_out = static_cast<emel::error::type>(err);
    return ok;
  }
};

using Tensor = sm;

}  // namespace emel::tensor
