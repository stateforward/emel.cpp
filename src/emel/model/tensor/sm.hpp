#pragma once

#include "emel/model/tensor/actions.hpp"
#include "emel/model/tensor/context.hpp"
#include "emel/model/tensor/detail.hpp"
#include "emel/model/tensor/errors.hpp"
#include "emel/model/tensor/events.hpp"
#include "emel/model/tensor/guards.hpp"
#include "emel/sm.hpp"

namespace emel::model::tensor {

struct ready {};
struct bind_tensor_request_decision {};
struct bind_tensor_exec {};
struct bind_tensor_result_decision {};
struct evict_tensor_request_decision {};
struct evict_tensor_exec {};
struct evict_tensor_result_decision {};
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
        sml::state<bind_tensor_request_decision> <= *sml::state<ready>
          + sml::event<detail::bind_tensor_runtime> / action::begin_bind_tensor
      , sml::state<bind_tensor_exec> <= sml::state<bind_tensor_request_decision>
          + sml::completion<detail::bind_tensor_runtime> [ guard::bind_tensor_request_valid{} ]
      , sml::state<errored> <= sml::state<bind_tensor_request_decision>
          + sml::completion<detail::bind_tensor_runtime> [ guard::bind_tensor_request_invalid{} ]
          / action::mark_invalid_request
      , sml::state<bind_tensor_result_decision> <= sml::state<bind_tensor_exec>
          + sml::completion<detail::bind_tensor_runtime> / action::exec_bind_tensor
      , sml::state<done> <= sml::state<bind_tensor_result_decision>
          + sml::completion<detail::bind_tensor_runtime> [ guard::operation_succeeded{} ]
      , sml::state<errored> <= sml::state<bind_tensor_result_decision>
          + sml::completion<detail::bind_tensor_runtime> [ guard::operation_not_dispatched{} ]
          / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      , sml::state<evict_tensor_request_decision> <= sml::state<ready>
          + sml::event<detail::evict_tensor_runtime> / action::begin_evict_tensor
      , sml::state<evict_tensor_exec> <= sml::state<evict_tensor_request_decision>
          + sml::completion<detail::evict_tensor_runtime> [ guard::evict_tensor_request_valid{} ]
      , sml::state<errored> <= sml::state<evict_tensor_request_decision>
          + sml::completion<detail::evict_tensor_runtime> [ guard::evict_tensor_request_invalid{} ]
          / action::mark_invalid_request
      , sml::state<evict_tensor_result_decision> <= sml::state<evict_tensor_exec>
          + sml::completion<detail::evict_tensor_runtime> / action::exec_evict_tensor
      , sml::state<done> <= sml::state<evict_tensor_result_decision>
          + sml::completion<detail::evict_tensor_runtime> [ guard::operation_succeeded{} ]
      , sml::state<errored> <= sml::state<evict_tensor_result_decision>
          + sml::completion<detail::evict_tensor_runtime> [ guard::operation_not_dispatched{} ]
          / action::mark_invalid_request

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
          + sml::completion<detail::capture_tensor_state_runtime> [ guard::operation_succeeded{} ]
      , sml::state<errored> <= sml::state<capture_tensor_state_result_decision>
          + sml::completion<detail::capture_tensor_state_runtime>
          [ guard::operation_not_dispatched{} ]
          / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<done> + sml::completion<detail::bind_tensor_runtime>
          / action::publish_done
      , sml::state<ready> <= sml::state<errored> + sml::completion<detail::bind_tensor_runtime>
          / action::publish_error
      , sml::state<ready> <= sml::state<done> + sml::completion<detail::evict_tensor_runtime>
          / action::publish_done
      , sml::state<ready> <= sml::state<errored> + sml::completion<detail::evict_tensor_runtime>
          / action::publish_error
      , sml::state<ready> <= sml::state<done>
          + sml::completion<detail::capture_tensor_state_runtime> / action::publish_done
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<detail::capture_tensor_state_runtime> / action::publish_error

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<bind_tensor_request_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<bind_tensor_exec> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<bind_tensor_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<evict_tensor_request_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<evict_tensor_exec> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<evict_tensor_result_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<capture_tensor_state_request_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<capture_tensor_state_exec>
          + sml::unexpected_event<sml::_> / action::on_unexpected
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

  bool process_event(const event::bind_tensor & ev) {
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    detail::runtime_status ctx{};
    detail::bind_tensor_runtime runtime{ev, ctx, tensor::detail::bind_or_sink(ev.error_out, error_sink)};
    base_type::process_event(runtime);
    return ctx.ok;
  }

  bool process_event(const event::evict_tensor & ev) {
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    detail::runtime_status ctx{};
    detail::evict_tensor_runtime runtime{ev, ctx, tensor::detail::bind_or_sink(ev.error_out, error_sink)};
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
};

}  // namespace emel::model::tensor
