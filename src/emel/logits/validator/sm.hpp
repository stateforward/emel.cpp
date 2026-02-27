#pragma once

#include "emel/logits/validator/actions.hpp"
#include "emel/logits/validator/context.hpp"
#include "emel/logits/validator/events.hpp"
#include "emel/logits/validator/guards.hpp"
#include "emel/sm.hpp"

namespace emel::logits::validator {

struct ready {};
struct request_decision {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<request_decision> <= *sml::state<ready> + sml::event<event::build_runtime>
          / action::begin_build
      , sml::state<done> <= sml::state<request_decision>
          + sml::completion<event::build_runtime> [ guard::valid_request{} ]
          / action::execute_build
      , sml::state<errored> <= sml::state<request_decision>
          + sml::completion<event::build_runtime> [ guard::invalid_request{} ]
          / action::mark_invalid_request

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<done> + sml::completion<event::build_runtime>
          / action::publish_done
      , sml::state<ready> <= sml::state<errored> + sml::completion<event::build_runtime>
          / action::publish_error

      //------------------------------------------------------------------------------//
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<ready> <= sml::state<request_decision> + sml::unexpected_event<sml::_>
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

  sm() : base_type() {}

  bool process_event(const event::build & ev) {
    event::build_ctx ctx{};
    event::build_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }

 private:
};

using Validator = sm;

}  // namespace emel::logits::validator
