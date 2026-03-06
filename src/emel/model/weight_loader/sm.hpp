#pragma once

#include "emel/model/weight_loader/actions.hpp"
#include "emel/model/weight_loader/context.hpp"
#include "emel/model/weight_loader/errors.hpp"
#include "emel/model/weight_loader/events.hpp"
#include "emel/model/weight_loader/guards.hpp"
#include "emel/sm.hpp"

namespace emel::model::weight_loader {

struct unbound {};
struct bound {};
struct awaiting_effects {};
struct ready {};
struct errored {};

struct bind_dispatch_decision {};
struct bind_done_decision {};
struct bind_done_callback {};
struct bind_error_decision {};
struct bind_error_callback {};

struct plan_dispatch_decision {};
struct plan_done_decision {};
struct plan_done_callback {};
struct plan_error_decision {};
struct plan_error_callback {};

struct apply_dispatch_decision {};
struct apply_request_decision {};
struct apply_error_scan_exec {};
struct apply_scan_result_decision {};
struct apply_done_decision {};
struct apply_done_callback {};
struct apply_error_decision {};
struct apply_error_callback {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Bind execution.
        sml::state<bind_dispatch_decision> <= *sml::state<unbound> + sml::event<event::bind_runtime>
          [ guard::valid_bind{} ]
          / action::exec_bind
      , sml::state<bind_dispatch_decision> <= sml::state<unbound> + sml::event<event::bind_runtime>
          [ guard::invalid_bind{} ]
          / action::mark_invalid_request
      , sml::state<bind_dispatch_decision> <= sml::state<bound> + sml::event<event::bind_runtime>
          [ guard::valid_bind{} ]
          / action::exec_bind
      , sml::state<bind_dispatch_decision> <= sml::state<bound> + sml::event<event::bind_runtime>
          [ guard::invalid_bind{} ]
          / action::mark_invalid_request
      , sml::state<bind_dispatch_decision> <= sml::state<awaiting_effects> + sml::event<event::bind_runtime>
          [ guard::valid_bind{} ]
          / action::exec_bind
      , sml::state<bind_dispatch_decision> <= sml::state<awaiting_effects> + sml::event<event::bind_runtime>
          [ guard::invalid_bind{} ]
          / action::mark_invalid_request
      , sml::state<bind_dispatch_decision> <= sml::state<ready> + sml::event<event::bind_runtime>
          [ guard::valid_bind{} ]
          / action::exec_bind
      , sml::state<bind_dispatch_decision> <= sml::state<ready> + sml::event<event::bind_runtime>
          [ guard::invalid_bind{} ]
          / action::mark_invalid_request
      , sml::state<bind_dispatch_decision> <= sml::state<errored> + sml::event<event::bind_runtime>
          [ guard::valid_bind{} ]
          / action::exec_bind
      , sml::state<bind_dispatch_decision> <= sml::state<errored> + sml::event<event::bind_runtime>
          [ guard::invalid_bind{} ]
          / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      // Bind callback dispatch.
      , sml::state<bind_done_decision> <= sml::state<bind_dispatch_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_error_none{} ]
      , sml::state<bind_error_decision> <= sml::state<bind_dispatch_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_error_invalid_request{} ]
      , sml::state<bind_error_decision> <= sml::state<bind_dispatch_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_error_capacity{} ]
      , sml::state<bind_error_decision> <= sml::state<bind_dispatch_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_error_backend_error{} ]
      , sml::state<bind_error_decision> <= sml::state<bind_dispatch_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_error_model_invalid{} ]
      , sml::state<bind_error_decision> <= sml::state<bind_dispatch_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_error_out_of_memory{} ]
      , sml::state<bind_error_decision> <= sml::state<bind_dispatch_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_error_internal_error{} ]
      , sml::state<bind_error_decision> <= sml::state<bind_dispatch_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_error_untracked{} ]
      , sml::state<bind_error_decision> <= sml::state<bind_dispatch_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_error_unknown{} ]

      , sml::state<bind_done_callback> <= sml::state<bind_done_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_done_callback_present{} ]
          / action::publish_bind_done
      , sml::state<bound> <= sml::state<bind_done_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_done_callback_absent{} ]
      , sml::state<bound> <= sml::state<bind_done_callback>
          + sml::completion<event::bind_runtime>

      , sml::state<bind_error_callback> <= sml::state<bind_error_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_error_callback_present{} ]
          / action::publish_bind_error
      , sml::state<errored> <= sml::state<bind_error_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_error_callback_absent{} ]
      , sml::state<errored> <= sml::state<bind_error_callback>
          + sml::completion<event::bind_runtime>

      //------------------------------------------------------------------------------//
      // Plan execution.
      , sml::state<plan_dispatch_decision> <= sml::state<bound> + sml::event<event::plan_runtime>
          [ guard::valid_plan{} ]
          / action::exec_plan
      , sml::state<plan_dispatch_decision> <= sml::state<bound> + sml::event<event::plan_runtime>
          [ guard::invalid_plan_request{} ]
          / action::mark_invalid_request
      , sml::state<plan_dispatch_decision> <= sml::state<bound> + sml::event<event::plan_runtime>
          [ guard::invalid_plan_capacity{} ]
          / action::mark_capacity

      , sml::state<plan_dispatch_decision> <= sml::state<ready> + sml::event<event::plan_runtime>
          [ guard::valid_plan{} ]
          / action::exec_plan
      , sml::state<plan_dispatch_decision> <= sml::state<ready> + sml::event<event::plan_runtime>
          [ guard::invalid_plan_request{} ]
          / action::mark_invalid_request
      , sml::state<plan_dispatch_decision> <= sml::state<ready> + sml::event<event::plan_runtime>
          [ guard::invalid_plan_capacity{} ]
          / action::mark_capacity

      , sml::state<plan_dispatch_decision> <= sml::state<unbound> + sml::event<event::plan_runtime>
          / action::mark_invalid_request
      , sml::state<plan_dispatch_decision> <= sml::state<awaiting_effects> + sml::event<event::plan_runtime>
          / action::mark_invalid_request
      , sml::state<plan_dispatch_decision> <= sml::state<errored> + sml::event<event::plan_runtime>
          / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      // Plan callback dispatch.
      , sml::state<plan_done_decision> <= sml::state<plan_dispatch_decision>
          + sml::completion<event::plan_runtime> [ guard::plan_error_none{} ]
      , sml::state<plan_error_decision> <= sml::state<plan_dispatch_decision>
          + sml::completion<event::plan_runtime> [ guard::plan_error_invalid_request{} ]
      , sml::state<plan_error_decision> <= sml::state<plan_dispatch_decision>
          + sml::completion<event::plan_runtime> [ guard::plan_error_capacity{} ]
      , sml::state<plan_error_decision> <= sml::state<plan_dispatch_decision>
          + sml::completion<event::plan_runtime> [ guard::plan_error_backend_error{} ]
      , sml::state<plan_error_decision> <= sml::state<plan_dispatch_decision>
          + sml::completion<event::plan_runtime> [ guard::plan_error_model_invalid{} ]
      , sml::state<plan_error_decision> <= sml::state<plan_dispatch_decision>
          + sml::completion<event::plan_runtime> [ guard::plan_error_out_of_memory{} ]
      , sml::state<plan_error_decision> <= sml::state<plan_dispatch_decision>
          + sml::completion<event::plan_runtime> [ guard::plan_error_internal_error{} ]
      , sml::state<plan_error_decision> <= sml::state<plan_dispatch_decision>
          + sml::completion<event::plan_runtime> [ guard::plan_error_untracked{} ]
      , sml::state<plan_error_decision> <= sml::state<plan_dispatch_decision>
          + sml::completion<event::plan_runtime> [ guard::plan_error_unknown{} ]

      , sml::state<plan_done_callback> <= sml::state<plan_done_decision>
          + sml::completion<event::plan_runtime> [ guard::plan_done_callback_present{} ]
          / action::publish_plan_done
      , sml::state<awaiting_effects> <= sml::state<plan_done_decision>
          + sml::completion<event::plan_runtime> [ guard::plan_done_callback_absent{} ]
      , sml::state<awaiting_effects> <= sml::state<plan_done_callback>
          + sml::completion<event::plan_runtime>

      , sml::state<plan_error_callback> <= sml::state<plan_error_decision>
          + sml::completion<event::plan_runtime> [ guard::plan_error_callback_present{} ]
          / action::publish_plan_error
      , sml::state<errored> <= sml::state<plan_error_decision>
          + sml::completion<event::plan_runtime> [ guard::plan_error_callback_absent{} ]
      , sml::state<errored> <= sml::state<plan_error_callback>
          + sml::completion<event::plan_runtime>

      //------------------------------------------------------------------------------//
      // Apply execution.
      , sml::state<apply_request_decision> <= sml::state<awaiting_effects> + sml::event<event::apply_runtime>
      , sml::state<apply_dispatch_decision> <= sml::state<apply_request_decision>
          + sml::completion<event::apply_runtime> [ guard::invalid_apply_request{} ]
          / action::mark_apply_invalid_request
      , sml::state<apply_error_scan_exec> <= sml::state<apply_request_decision>
          + sml::completion<event::apply_runtime> [ guard::valid_apply_request{} ]
          / action::scan_apply_effect_errors
      , sml::state<apply_dispatch_decision> <= sml::state<apply_request_decision>
          + sml::completion<event::apply_runtime>
          / action::mark_apply_invalid_request

      , sml::state<apply_scan_result_decision> <= sml::state<apply_error_scan_exec>
          + sml::completion<event::apply_runtime>
      , sml::state<apply_dispatch_decision> <= sml::state<apply_scan_result_decision>
          + sml::completion<event::apply_runtime> [ guard::apply_effect_errors_present{} ]
          / action::mark_apply_backend_error
      , sml::state<apply_dispatch_decision> <= sml::state<apply_scan_result_decision>
          + sml::completion<event::apply_runtime> [ guard::apply_effect_errors_absent{} ]
          / action::exec_apply
      , sml::state<apply_dispatch_decision> <= sml::state<apply_scan_result_decision>
          + sml::completion<event::apply_runtime>
          / action::mark_apply_backend_error

      , sml::state<apply_dispatch_decision> <= sml::state<unbound> + sml::event<event::apply_runtime>
          [ guard::invalid_apply_request{} ]
          / action::mark_apply_invalid_request
      , sml::state<apply_dispatch_decision> <= sml::state<bound> + sml::event<event::apply_runtime>
          / action::mark_apply_invalid_request
      , sml::state<apply_dispatch_decision> <= sml::state<ready> + sml::event<event::apply_runtime>
          / action::mark_apply_invalid_request
      , sml::state<apply_dispatch_decision> <= sml::state<errored> + sml::event<event::apply_runtime>
          / action::mark_apply_invalid_request

      //------------------------------------------------------------------------------//
      // Apply callback dispatch.
      , sml::state<apply_done_decision> <= sml::state<apply_dispatch_decision>
          + sml::completion<event::apply_runtime> [ guard::apply_error_none{} ]
      , sml::state<apply_error_decision> <= sml::state<apply_dispatch_decision>
          + sml::completion<event::apply_runtime> [ guard::apply_error_invalid_request{} ]
      , sml::state<apply_error_decision> <= sml::state<apply_dispatch_decision>
          + sml::completion<event::apply_runtime> [ guard::apply_error_capacity{} ]
      , sml::state<apply_error_decision> <= sml::state<apply_dispatch_decision>
          + sml::completion<event::apply_runtime> [ guard::apply_error_backend_error{} ]
      , sml::state<apply_error_decision> <= sml::state<apply_dispatch_decision>
          + sml::completion<event::apply_runtime> [ guard::apply_error_model_invalid{} ]
      , sml::state<apply_error_decision> <= sml::state<apply_dispatch_decision>
          + sml::completion<event::apply_runtime> [ guard::apply_error_out_of_memory{} ]
      , sml::state<apply_error_decision> <= sml::state<apply_dispatch_decision>
          + sml::completion<event::apply_runtime> [ guard::apply_error_internal_error{} ]
      , sml::state<apply_error_decision> <= sml::state<apply_dispatch_decision>
          + sml::completion<event::apply_runtime> [ guard::apply_error_untracked{} ]
      , sml::state<apply_error_decision> <= sml::state<apply_dispatch_decision>
          + sml::completion<event::apply_runtime> [ guard::apply_error_unknown{} ]

      , sml::state<apply_done_callback> <= sml::state<apply_done_decision>
          + sml::completion<event::apply_runtime> [ guard::apply_done_callback_present{} ]
          / action::publish_apply_done
      , sml::state<ready> <= sml::state<apply_done_decision>
          + sml::completion<event::apply_runtime> [ guard::apply_done_callback_absent{} ]
      , sml::state<ready> <= sml::state<apply_done_callback>
          + sml::completion<event::apply_runtime>

      , sml::state<apply_error_callback> <= sml::state<apply_error_decision>
          + sml::completion<event::apply_runtime> [ guard::apply_error_callback_present{} ]
          / action::publish_apply_error
      , sml::state<errored> <= sml::state<apply_error_decision>
          + sml::completion<event::apply_runtime> [ guard::apply_error_callback_absent{} ]
      , sml::state<errored> <= sml::state<apply_error_callback>
          + sml::completion<event::apply_runtime>

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<errored> <= sml::state<unbound> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<bound> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<awaiting_effects> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<ready> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<errored> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<bind_dispatch_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<bind_done_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<bind_done_callback> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<bind_error_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<bind_error_callback> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<plan_dispatch_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<plan_done_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<plan_done_callback> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<plan_error_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<plan_error_callback> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<apply_dispatch_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<apply_request_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<apply_error_scan_exec> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<apply_scan_result_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<apply_done_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<apply_done_callback> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<apply_error_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<errored> <= sml::state<apply_error_callback> + sml::unexpected_event<sml::_>
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

  sm() : base_type() {}

  bool process_event(const event::bind_storage & ev) {
    event::bind_ctx ctx{};
    event::bind_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::plan_load & ev) {
    event::plan_ctx ctx{};
    event::plan_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::apply_effect_results & ev) {
    event::apply_ctx ctx{};
    event::apply_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};

}  // namespace emel::model::weight_loader
