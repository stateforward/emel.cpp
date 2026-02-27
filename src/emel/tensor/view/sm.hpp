#pragma once
// benchmark: scaffold

#include "emel/sm.hpp"
#include "emel/tensor/view/actions.hpp"
#include "emel/tensor/view/context.hpp"
#include "emel/tensor/view/detail.hpp"
#include "emel/tensor/view/errors.hpp"
#include "emel/tensor/view/events.hpp"
#include "emel/tensor/view/guards.hpp"

namespace emel::tensor::view {

struct ready {};
struct capture_tensor_view_request_decision {};
struct capture_tensor_view_exec {};
struct capture_tensor_view_result_decision {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
        sml::state<capture_tensor_view_request_decision> <= *sml::state<ready>
          + sml::event<detail::capture_tensor_view_runtime> / action::begin_capture_tensor_view
      , sml::state<capture_tensor_view_exec> <= sml::state<capture_tensor_view_request_decision>
          + sml::completion<detail::capture_tensor_view_runtime>
          [ guard::capture_tensor_view_request_valid{} ]
      , sml::state<errored> <= sml::state<capture_tensor_view_request_decision>
          + sml::completion<detail::capture_tensor_view_runtime>
          [ guard::capture_tensor_view_request_invalid{} ]
          / action::mark_invalid_request
      , sml::state<capture_tensor_view_result_decision> <= sml::state<capture_tensor_view_exec>
          + sml::completion<detail::capture_tensor_view_runtime> / action::exec_capture_tensor_view
      , sml::state<done> <= sml::state<capture_tensor_view_result_decision>
          + sml::completion<detail::capture_tensor_view_runtime> [ guard::operation_succeeded{} ]
      , sml::state<errored> <= sml::state<capture_tensor_view_result_decision>
          + sml::completion<detail::capture_tensor_view_runtime>
          [ guard::operation_failed_with_error{} ]
          / action::mark_error_from_operation
      , sml::state<errored> <= sml::state<capture_tensor_view_result_decision>
          + sml::completion<detail::capture_tensor_view_runtime>
          [ guard::operation_failed_without_error{} ]
          / action::mark_internal_error

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<done>
          + sml::completion<detail::capture_tensor_view_runtime> / action::publish_done
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<detail::capture_tensor_view_runtime> / action::publish_error

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<capture_tensor_view_request_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<capture_tensor_view_exec>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<capture_tensor_view_result_decision>
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

  bool process_event(const event::capture_tensor_view & ev) {
    int32_t error_sink = static_cast<int32_t>(emel::error::cast(error::none));
    detail::runtime_status ctx{};
    detail::capture_tensor_view_runtime runtime{
      ev,
      ctx,
      detail::bind_or_sink(ev.error_out, error_sink),
    };
    base_type::process_event(runtime);
    return ctx.ok;
  }
};

using View = sm;

}  // namespace emel::tensor::view
