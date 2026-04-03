#pragma once
// benchmark: designed

#include "emel/sm.hpp"
#include "emel/tensor/view/actions.hpp"
#include "emel/tensor/view/context.hpp"
#include "emel/tensor/view/detail.hpp"
#include "emel/tensor/view/errors.hpp"
#include "emel/tensor/view/events.hpp"
#include "emel/tensor/view/guards.hpp"

namespace emel::tensor::view {

template <class policy>
struct ready {};

template <class policy>
struct capture_tensor_view_request_decision {};

template <class policy>
struct capture_tensor_view_exec {};

template <class policy>
struct capture_tensor_view_result_decision {};

template <class policy>
struct done {};

template <class policy>
struct errored {};

template <class policy>
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
        sml::state<capture_tensor_view_request_decision<policy>> <= *sml::state<ready<policy>>
          + sml::event<detail::capture_tensor_view_runtime<policy>> / action::begin_capture_tensor_view
      , sml::state<capture_tensor_view_exec<policy>> <=
          sml::state<capture_tensor_view_request_decision<policy>>
          + sml::completion<detail::capture_tensor_view_runtime<policy>>
          [ guard::capture_tensor_view_request_valid{} ]
      , sml::state<errored<policy>> <= sml::state<capture_tensor_view_request_decision<policy>>
          + sml::completion<detail::capture_tensor_view_runtime<policy>>
          [ guard::capture_tensor_view_request_invalid{} ]
          / action::mark_invalid_request
      , sml::state<capture_tensor_view_result_decision<policy>> <=
          sml::state<capture_tensor_view_exec<policy>>
          + sml::completion<detail::capture_tensor_view_runtime<policy>>
          / action::exec_capture_tensor_view
      , sml::state<done<policy>> <= sml::state<capture_tensor_view_result_decision<policy>>
          + sml::completion<detail::capture_tensor_view_runtime<policy>>
          [ guard::operation_succeeded{} ]
      , sml::state<errored<policy>> <= sml::state<capture_tensor_view_result_decision<policy>>
          + sml::completion<detail::capture_tensor_view_runtime<policy>>
          [ guard::operation_failed_with_error{} ]
          / action::mark_error_from_operation
      , sml::state<errored<policy>> <= sml::state<capture_tensor_view_result_decision<policy>>
          + sml::completion<detail::capture_tensor_view_runtime<policy>>
          [ guard::operation_failed_without_error{} ]
          / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<ready<policy>> <= sml::state<done<policy>>
          + sml::completion<detail::capture_tensor_view_runtime<policy>> / action::publish_done
      , sml::state<ready<policy>> <= sml::state<errored<policy>>
          + sml::completion<detail::capture_tensor_view_runtime<policy>> / action::publish_error

      //------------------------------------------------------------------------------//
      , sml::state<ready<policy>> <= sml::state<ready<policy>> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready<policy>> <= sml::state<capture_tensor_view_request_decision<policy>>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready<policy>> <= sml::state<capture_tensor_view_exec<policy>>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready<policy>> <= sml::state<capture_tensor_view_result_decision<policy>>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready<policy>> <= sml::state<done<policy>> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready<policy>> <= sml::state<errored<policy>> + sml::unexpected_event<sml::_>
          / action::on_unexpected
    );
    // clang-format on
  }
};

template <class policy>
struct sm : public emel::sm<model<policy>, action::context> {
  using base_type = emel::sm<model<policy>, action::context>;
  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  bool process_event(const event::capture_tensor_view<policy> & ev) {
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    detail::runtime_status ctx{};
    detail::capture_tensor_view_runtime<policy> runtime{
      ev,
      ctx,
      detail::bind_or_sink(ev.error_out, error_sink),
    };
    base_type::process_event(runtime);
    return ctx.ok;
  }
};

}  // namespace emel::tensor::view
