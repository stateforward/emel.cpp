#pragma once

// benchmark: scaffold

#include "emel/model/needle/actions.hpp"
#include "emel/model/needle/context.hpp"
#include "emel/model/needle/errors.hpp"
#include "emel/model/needle/events.hpp"
#include "emel/model/needle/guards.hpp"
#include "emel/sm.hpp"

namespace emel::model::needle {

// Binder lifecycle: unbound until a bind request maps the positional loader
// table onto the named contract; every validation failure lands in
// state_errored with a typed bind_error. Re-binding from bound or errored is
// allowed (the request carries fresh inputs and output storage).
struct state_unbound {};
struct state_bound {};
struct state_errored {};

struct state_bind_request_decision {};
struct state_bind_outcome_dispatch {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Bind op.
        sml::state<state_bind_request_decision> <= *sml::state<state_unbound>
          + sml::event<event::bind_runtime> / action::effect_begin_bind
      , sml::state<state_bind_request_decision> <= sml::state<state_bound>
          + sml::event<event::bind_runtime> / action::effect_begin_bind
      , sml::state<state_bind_request_decision> <= sml::state<state_errored>
          + sml::event<event::bind_runtime> / action::effect_begin_bind

      , sml::state<state_bind_outcome_dispatch> <= sml::state<state_bind_request_decision>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_valid_request{} ]
          / action::effect_exec_bind
      , sml::state<state_bind_outcome_dispatch> <= sml::state<state_bind_request_decision>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_invalid_request{} ]
          / action::effect_mark_bind_invalid_request

      , sml::state<state_bound> <= sml::state<state_bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_error_none{} ]
          / action::effect_publish_bind_done
      , sml::state<state_errored> <= sml::state<state_bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_error_invalid_request{} ]
          / action::effect_publish_bind_error
      , sml::state<state_errored> <= sml::state<state_bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_error_geometry_invalid{} ]
          / action::effect_publish_bind_error
      , sml::state<state_errored> <= sml::state<state_bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_error_tensor_count_mismatch{} ]
          / action::effect_publish_bind_error
      , sml::state<state_errored> <= sml::state<state_bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_error_tensor_dtype_mismatch{} ]
          / action::effect_publish_bind_error
      , sml::state<state_errored> <= sml::state<state_bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_error_tensor_shape_mismatch{} ]
          / action::effect_publish_bind_error
      , sml::state<state_errored> <= sml::state<state_bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_error_head_manifest_invalid{} ]
          / action::effect_publish_bind_error
      , sml::state<state_errored> <= sml::state<state_bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_error_internal_error{} ]
          / action::effect_publish_bind_error
      , sml::state<state_errored> <= sml::state<state_bind_outcome_dispatch>
          + sml::completion<event::bind_runtime> [ guard::guard_bind_error_unknown{} ]
          / action::effect_publish_bind_error

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<state_errored> <= sml::state<state_unbound> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_bound> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_errored> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_bind_request_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_bind_outcome_dispatch> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
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

  bool process_event(const event::bind &ev) {
    event::bind_ctx ctx{};
    event::bind_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};

using NeedleBinder = sm;

} // namespace emel::model::needle
