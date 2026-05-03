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
struct state_bind_storage_decision {};
struct state_bind_storage_done_decision {};
struct state_bind_storage_done_callback {};
struct state_bind_storage_error_decision {};
struct state_bind_storage_error_callback {};
struct state_plan_load_decision {};
struct state_plan_load_done_decision {};
struct state_plan_load_done_callback {};
struct state_plan_load_error_decision {};
struct state_plan_load_capacity_error_decision {};
struct state_plan_load_error_callback {};
struct state_awaiting_effects {};
struct state_apply_effect_results_request_decision {};
struct state_apply_effect_results_done_decision {};
struct state_apply_effect_results_done_callback {};
struct state_apply_effect_results_error_decision {};
struct state_apply_effect_results_backend_error_decision {};
struct state_apply_effect_results_error_callback {};
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
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Tensor-owned bulk storage binding.
        sml::state<state_bind_storage_decision> <= *sml::state<ready>
          + sml::event<event::bind_storage> [ guard::storage_bind_valid{} ]
          / action::effect_bind_storage
      , sml::state<state_bind_storage_error_decision> <= sml::state<ready>
          + sml::event<event::bind_storage> [ guard::storage_bind_invalid{} ]

      , sml::state<state_bind_storage_done_decision> <= sml::state<state_bind_storage_decision>
          + sml::completion<event::bind_storage>

      , sml::state<state_bind_storage_done_callback> <=
          sml::state<state_bind_storage_done_decision>
          + sml::completion<event::bind_storage>
          [ guard::bind_storage_done_callback_present{} ]
          / action::publish_bind_storage_done
      , sml::state<ready> <= sml::state<state_bind_storage_done_decision>
          + sml::completion<event::bind_storage>
          [ guard::bind_storage_done_callback_absent{} ]
      , sml::state<ready> <= sml::state<state_bind_storage_done_callback>
          + sml::completion<event::bind_storage>

      , sml::state<state_bind_storage_error_callback> <=
          sml::state<state_bind_storage_error_decision>
          + sml::completion<event::bind_storage>
          [ guard::bind_storage_error_callback_present{} ]
          / action::publish_bind_storage_error
      , sml::state<ready> <= sml::state<state_bind_storage_error_decision>
          + sml::completion<event::bind_storage>
          [ guard::bind_storage_error_callback_absent{} ]
      , sml::state<ready> <= sml::state<state_bind_storage_error_callback>
          + sml::completion<event::bind_storage>

      //------------------------------------------------------------------------------//
      // Tensor-owned load planning.
      , sml::state<state_plan_load_done_decision> <= sml::state<ready>
          + sml::event<detail::plan_load_runtime> [ guard::plan_load_valid{} ]
          / action::effect_plan_load
      , sml::state<state_plan_load_error_decision> <= sml::state<ready>
          + sml::event<detail::plan_load_runtime>
          [ guard::plan_load_invalid_request{} ]
      , sml::state<state_plan_load_capacity_error_decision> <= sml::state<ready>
          + sml::event<detail::plan_load_runtime>
          [ guard::plan_load_invalid_capacity{} ]

      , sml::state<state_plan_load_done_callback> <= sml::state<state_plan_load_done_decision>
          + sml::completion<detail::plan_load_runtime>
          [ guard::plan_load_done_callback_present{} ]
          / action::publish_plan_load_done
      , sml::state<state_awaiting_effects> <= sml::state<state_plan_load_done_decision>
          + sml::completion<detail::plan_load_runtime>
          [ guard::plan_load_done_callback_absent{} ]
          / action::record_plan_load_done
      , sml::state<state_awaiting_effects> <= sml::state<state_plan_load_done_callback>
          + sml::completion<detail::plan_load_runtime>

      , sml::state<state_plan_load_error_callback> <= sml::state<state_plan_load_error_decision>
          + sml::completion<detail::plan_load_runtime>
          [ guard::plan_load_error_callback_present{} ]
          / action::publish_plan_load_invalid_request
      , sml::state<ready> <= sml::state<state_plan_load_error_decision>
          + sml::completion<detail::plan_load_runtime>
          [ guard::plan_load_error_callback_absent{} ]
          / action::record_plan_load_invalid_request
      , sml::state<state_plan_load_error_callback> <=
          sml::state<state_plan_load_capacity_error_decision>
          + sml::completion<detail::plan_load_runtime>
          [ guard::plan_load_error_callback_present{} ]
          / action::publish_plan_load_capacity_error
      , sml::state<ready> <= sml::state<state_plan_load_capacity_error_decision>
          + sml::completion<detail::plan_load_runtime>
          [ guard::plan_load_error_callback_absent{} ]
          / action::record_plan_load_capacity_error
      , sml::state<ready> <= sml::state<state_plan_load_error_callback>
          + sml::completion<detail::plan_load_runtime>

      //------------------------------------------------------------------------------//
      // Tensor-owned effect application.
      , sml::state<state_apply_effect_results_request_decision> <=
          sml::state<state_awaiting_effects>
          + sml::event<detail::apply_effect_results_runtime>
      , sml::state<state_apply_effect_results_error_decision> <=
          sml::state<state_apply_effect_results_request_decision>
          + sml::completion<detail::apply_effect_results_runtime>
          [ guard::apply_results_invalid{} ]
      , sml::state<state_apply_effect_results_backend_error_decision> <=
          sml::state<state_apply_effect_results_request_decision>
          + sml::completion<detail::apply_effect_results_runtime>
          [ guard::apply_results_valid_with_effect_errors{} ]
      , sml::state<state_apply_effect_results_done_decision> <=
          sml::state<state_apply_effect_results_request_decision>
          + sml::completion<detail::apply_effect_results_runtime>
          [ guard::apply_results_valid_without_effect_errors{} ]
          / action::effect_apply_results

      , sml::state<state_apply_effect_results_error_decision> <= sml::state<ready>
          + sml::event<detail::apply_effect_results_runtime>

      , sml::state<state_apply_effect_results_done_callback> <=
          sml::state<state_apply_effect_results_done_decision>
          + sml::completion<detail::apply_effect_results_runtime>
          [ guard::apply_effect_results_done_callback_present{} ]
          / action::publish_apply_effect_results_done
      , sml::state<ready> <= sml::state<state_apply_effect_results_done_decision>
          + sml::completion<detail::apply_effect_results_runtime>
          [ guard::apply_effect_results_done_callback_absent{} ]
          / action::record_apply_effect_results_done
      , sml::state<ready> <= sml::state<state_apply_effect_results_done_callback>
          + sml::completion<detail::apply_effect_results_runtime>

      , sml::state<state_apply_effect_results_error_callback> <=
          sml::state<state_apply_effect_results_error_decision>
          + sml::completion<detail::apply_effect_results_runtime>
          [ guard::apply_effect_results_error_callback_present{} ]
          / action::publish_apply_effect_results_invalid_request
      , sml::state<ready> <= sml::state<state_apply_effect_results_error_decision>
          + sml::completion<detail::apply_effect_results_runtime>
          [ guard::apply_effect_results_error_callback_absent{} ]
          / action::record_apply_effect_results_invalid_request
      , sml::state<state_apply_effect_results_error_callback> <=
          sml::state<state_apply_effect_results_backend_error_decision>
          + sml::completion<detail::apply_effect_results_runtime>
          [ guard::apply_effect_results_error_callback_present{} ]
          / action::publish_apply_effect_results_backend_error
      , sml::state<ready> <= sml::state<state_apply_effect_results_backend_error_decision>
          + sml::completion<detail::apply_effect_results_runtime>
          [ guard::apply_effect_results_error_callback_absent{} ]
          / action::record_apply_effect_results_backend_error
      , sml::state<ready> <= sml::state<state_apply_effect_results_error_callback>
          + sml::completion<detail::apply_effect_results_runtime>

      //------------------------------------------------------------------------------//
      , sml::state<bind_tensor_request_decision> <= sml::state<ready>
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
          [ guard::error_code_output_present{} ]
          / action::publish_done_with_error_code
      , sml::state<ready> <= sml::state<done> + sml::completion<detail::bind_tensor_runtime>
          [ guard::error_code_output_absent{} ]
          / action::publish_done
      , sml::state<ready> <= sml::state<errored> + sml::completion<detail::bind_tensor_runtime>
          [ guard::error_code_output_present{} ]
          / action::publish_error_with_error_code
      , sml::state<ready> <= sml::state<errored> + sml::completion<detail::bind_tensor_runtime>
          [ guard::error_code_output_absent{} ]
          / action::publish_error
      , sml::state<ready> <= sml::state<done> + sml::completion<detail::evict_tensor_runtime>
          [ guard::error_code_output_present{} ]
          / action::publish_done_with_error_code
      , sml::state<ready> <= sml::state<done> + sml::completion<detail::evict_tensor_runtime>
          [ guard::error_code_output_absent{} ]
          / action::publish_done
      , sml::state<ready> <= sml::state<errored> + sml::completion<detail::evict_tensor_runtime>
          [ guard::error_code_output_present{} ]
          / action::publish_error_with_error_code
      , sml::state<ready> <= sml::state<errored> + sml::completion<detail::evict_tensor_runtime>
          [ guard::error_code_output_absent{} ]
          / action::publish_error
      , sml::state<ready> <= sml::state<done>
          + sml::completion<detail::capture_tensor_state_runtime>
          [ guard::error_code_output_present{} ]
          / action::publish_done_with_error_code
      , sml::state<ready> <= sml::state<done>
          + sml::completion<detail::capture_tensor_state_runtime>
          [ guard::error_code_output_absent{} ]
          / action::publish_done
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<detail::capture_tensor_state_runtime>
          [ guard::error_code_output_present{} ]
          / action::publish_error_with_error_code
      , sml::state<ready> <= sml::state<errored>
          + sml::completion<detail::capture_tensor_state_runtime>
          [ guard::error_code_output_absent{} ]
          / action::publish_error

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<state_bind_storage_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_bind_storage_done_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_bind_storage_done_callback>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_bind_storage_error_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_bind_storage_error_callback>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_plan_load_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_plan_load_done_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_plan_load_done_callback>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_plan_load_error_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_plan_load_capacity_error_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_plan_load_error_callback>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_awaiting_effects>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_apply_effect_results_request_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_apply_effect_results_done_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_apply_effect_results_done_callback>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_apply_effect_results_error_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_apply_effect_results_backend_error_decision>
          + sml::unexpected_event<sml::_> / action::on_unexpected
      , sml::state<ready> <= sml::state<state_apply_effect_results_error_callback>
          + sml::unexpected_event<sml::_> / action::on_unexpected
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

  bool process_event(const event::bind_storage &ev) {
    const bool valid = guard::storage_bind_valid{}(ev);
    const bool accepted = base_type::process_event(ev);
    return accepted && valid;
  }

  bool process_event(const event::plan_load &ev) {
    detail::runtime_status ctx{};
    detail::plan_load_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.ok;
  }

  bool process_event(const event::apply_effect_results &ev) {
    detail::runtime_status ctx{};
    detail::apply_effect_results_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.ok;
  }

  bool process_event(const event::bind_tensor &ev) {
    detail::runtime_status ctx{};
    detail::bind_tensor_runtime runtime{ev, ctx, ev.error_out};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.ok;
  }

  bool process_event(const event::evict_tensor &ev) {
    detail::runtime_status ctx{};
    detail::evict_tensor_runtime runtime{ev, ctx, ev.error_out};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.ok;
  }

  bool process_event(const event::capture_tensor_state &ev) {
    detail::runtime_status ctx{};
    detail::capture_tensor_state_runtime runtime{ev, ctx, ev.error_out};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.ok;
  }
};

} // namespace emel::model::tensor
