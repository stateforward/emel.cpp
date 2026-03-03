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

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<bound> <= *sml::state<unbound> + sml::event<event::bind_runtime>
          [ guard::valid_bind{} ]
          / action::exec_bind
      , sml::state<errored> <= *sml::state<unbound> + sml::event<event::bind_runtime>
          [ guard::invalid_bind{} ]
          / action::mark_invalid_request

      , sml::state<bound> <= sml::state<bound> + sml::event<event::bind_runtime>
          [ guard::valid_bind{} ]
          / action::exec_bind
      , sml::state<errored> <= sml::state<bound> + sml::event<event::bind_runtime>
          [ guard::invalid_bind{} ]
          / action::mark_invalid_request

      , sml::state<bound> <= sml::state<awaiting_effects> + sml::event<event::bind_runtime>
          [ guard::valid_bind{} ]
          / action::exec_bind
      , sml::state<errored> <= sml::state<awaiting_effects> + sml::event<event::bind_runtime>
          [ guard::invalid_bind{} ]
          / action::mark_invalid_request

      , sml::state<bound> <= sml::state<ready> + sml::event<event::bind_runtime>
          [ guard::valid_bind{} ]
          / action::exec_bind
      , sml::state<errored> <= sml::state<ready> + sml::event<event::bind_runtime>
          [ guard::invalid_bind{} ]
          / action::mark_invalid_request

      , sml::state<bound> <= sml::state<errored> + sml::event<event::bind_runtime>
          [ guard::valid_bind{} ]
          / action::exec_bind
      , sml::state<errored> <= sml::state<errored> + sml::event<event::bind_runtime>
          [ guard::invalid_bind{} ]
          / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      , sml::state<awaiting_effects> <= sml::state<bound> + sml::event<event::plan_runtime>
          [ guard::valid_plan{} ]
          / action::exec_plan
      , sml::state<errored> <= sml::state<bound> + sml::event<event::plan_runtime>
          [ guard::invalid_plan_request{} ]
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<bound> + sml::event<event::plan_runtime>
          [ guard::invalid_plan_capacity{} ]
          / action::mark_capacity

      , sml::state<awaiting_effects> <= sml::state<ready> + sml::event<event::plan_runtime>
          [ guard::valid_plan{} ]
          / action::exec_plan
      , sml::state<errored> <= sml::state<ready> + sml::event<event::plan_runtime>
          [ guard::invalid_plan_request{} ]
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<ready> + sml::event<event::plan_runtime>
          [ guard::invalid_plan_capacity{} ]
          / action::mark_capacity

      , sml::state<errored> <= sml::state<unbound> + sml::event<event::plan_runtime>
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<awaiting_effects> + sml::event<event::plan_runtime>
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<errored> + sml::event<event::plan_runtime>
          / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<awaiting_effects> + sml::event<event::apply_runtime>
          [ guard::valid_apply{} ]
          / action::exec_apply
      , sml::state<errored> <= sml::state<awaiting_effects> + sml::event<event::apply_runtime>
          [ guard::invalid_apply_request{} ]
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<awaiting_effects> + sml::event<event::apply_runtime>
          [ guard::apply_has_effect_errors{} ]
          / action::mark_backend_error

      , sml::state<errored> <= sml::state<unbound> + sml::event<event::apply_runtime>
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<bound> + sml::event<event::apply_runtime>
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<ready> + sml::event<event::apply_runtime>
          / action::mark_invalid_request
      , sml::state<errored> <= sml::state<errored> + sml::event<event::apply_runtime>
          / action::mark_invalid_request

      //------------------------------------------------------------------------------//
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
    const bool phase_ok = ctx.err == emel::error::cast(error::none);
    while (phase_ok && static_cast<bool>(ev.on_done)) {
      ev.on_done(events::bind_done{.request = ev});
      break;
    }
    while ((!phase_ok) && static_cast<bool>(ev.on_error)) {
      ev.on_error(events::bind_error{
          .request = ev,
          .err = ctx.err,
      });
      break;
    }
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::plan_load & ev) {
    event::plan_ctx ctx{};
    event::plan_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    const bool phase_ok = ctx.err == emel::error::cast(error::none);
    while (phase_ok && static_cast<bool>(ev.on_done)) {
      ev.on_done(events::plan_done{
          .request = ev,
          .effect_count = ctx.effect_count,
      });
      break;
    }
    while ((!phase_ok) && static_cast<bool>(ev.on_error)) {
      ev.on_error(events::plan_error{
          .request = ev,
          .err = ctx.err,
      });
      break;
    }
    return accepted && ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::apply_effect_results & ev) {
    event::apply_ctx ctx{};
    event::apply_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    const bool phase_ok = ctx.err == emel::error::cast(error::none);
    while (phase_ok && static_cast<bool>(ev.on_done)) {
      ev.on_done(events::apply_done{.request = ev});
      break;
    }
    while ((!phase_ok) && static_cast<bool>(ev.on_error)) {
      ev.on_error(events::apply_error{
          .request = ev,
          .err = ctx.err,
      });
      break;
    }
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};

}  // namespace emel::model::weight_loader
