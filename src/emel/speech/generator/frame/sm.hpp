#pragma once

// benchmark: designed

#include <concepts>
#include <type_traits>

#include "emel/sm.hpp"
#include "emel/speech/generator/frame/actions.hpp"
#include "emel/speech/generator/frame/context.hpp"
#include "emel/speech/generator/frame/events.hpp"
#include "emel/speech/generator/frame/guards.hpp"

namespace emel::speech::generator::frame {

struct state_idle {};
struct state_tokenize_result {};
struct state_plan_result {};
struct state_predict_result {};
struct state_graph_result {};
struct state_sample_result {};
struct state_detokenize_result {};
struct state_done_channel_decision {};
struct state_error_channel_decision {};

/*
synchronous speech generator middle-frame actor

state purpose
- state_idle accepts one already encoded frame.
- result states synchronously interpret tokenize, plan, predict, graph, sample,
  and detokenize child outcomes.
- channel-decision states publish exactly one immediate callback when present.

control invariants
- every child dispatch remains inside the originating RTC boundary.
- request-local status is carried only by detail::run_frame.
- action::context owns only the injected collaborators and reusable buffers.
- every accepted run returns to state_idle before process_event returns.
*/
template <action::frame_dependencies dependencies_type> struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    using event_run = detail::run_frame;
    using guard_done_present =
        guard::guard_done_callback_present<dependencies_type>;
    using guard_done_absent =
        guard::guard_done_callback_absent<dependencies_type>;
    using guard_error_present =
        guard::guard_error_callback_present<dependencies_type>;
    using guard_error_absent =
        guard::guard_error_callback_absent<dependencies_type>;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Request validation and tokenization.
        sml::state<state_tokenize_result> <= *sml::state<state_idle>
          + sml::event<event_run> [ guard::guard_request_valid<dependencies_type>{} ]
          / action::effect_tokenize<dependencies_type>{}
      , sml::state<state_error_channel_decision> <= sml::state<state_idle>
          + sml::event<event_run> [ guard::guard_request_invalid<dependencies_type>{} ]
          / action::effect_fail<dependencies_type, error::invalid_request>{}

      //------------------------------------------------------------------------------//
      // Bounded middle-frame phase chain.
      , sml::state<state_plan_result> <= sml::state<state_tokenize_result>
          + sml::completion<event_run> [ guard::guard_child_succeeded<dependencies_type>{} ]
          / action::effect_plan<dependencies_type>{}
      , sml::state<state_error_channel_decision> <= sml::state<state_tokenize_result>
          + sml::completion<event_run> [ guard::guard_child_failed<dependencies_type>{} ]
          / action::effect_fail<dependencies_type, error::tokenize_failed>{}
      , sml::state<state_predict_result> <= sml::state<state_plan_result>
          + sml::completion<event_run> [ guard::guard_child_succeeded<dependencies_type>{} ]
          / action::effect_predict<dependencies_type>{}
      , sml::state<state_error_channel_decision> <= sml::state<state_plan_result>
          + sml::completion<event_run> [ guard::guard_child_failed<dependencies_type>{} ]
          / action::effect_fail<dependencies_type, error::planning_failed>{}
      , sml::state<state_graph_result> <= sml::state<state_predict_result>
          + sml::completion<event_run> [ guard::guard_child_succeeded<dependencies_type>{} ]
          / action::effect_execute_graph<dependencies_type>{}
      , sml::state<state_error_channel_decision> <= sml::state<state_predict_result>
          + sml::completion<event_run> [ guard::guard_child_failed<dependencies_type>{} ]
          / action::effect_fail<dependencies_type, error::predict_failed>{}
      , sml::state<state_sample_result> <= sml::state<state_graph_result>
          + sml::completion<event_run> [ guard::guard_prediction_succeeded<dependencies_type>{} ]
          / action::effect_sample<dependencies_type>{}
      , sml::state<state_error_channel_decision> <= sml::state<state_graph_result>
          + sml::completion<event_run> [ guard::guard_prediction_failed<dependencies_type>{} ]
          / action::effect_fail<dependencies_type, error::graph_failed>{}
      , sml::state<state_detokenize_result> <= sml::state<state_sample_result>
          + sml::completion<event_run> [ guard::guard_prediction_succeeded<dependencies_type>{} ]
          / action::effect_detokenize<dependencies_type>{}
      , sml::state<state_error_channel_decision> <= sml::state<state_sample_result>
          + sml::completion<event_run> [ guard::guard_prediction_failed<dependencies_type>{} ]
          / action::effect_fail<dependencies_type, error::sample_failed>{}
      , sml::state<state_done_channel_decision> <= sml::state<state_detokenize_result>
          + sml::completion<event_run> [ guard::guard_frame_produced<dependencies_type>{} ]
          / action::effect_publish_done<dependencies_type>{}
      , sml::state<state_error_channel_decision> <= sml::state<state_detokenize_result>
          + sml::completion<event_run> [ guard::guard_frame_pending<dependencies_type>{} ]
          / action::effect_fail<dependencies_type, error::frame_pending>{}
      , sml::state<state_error_channel_decision> <= sml::state<state_detokenize_result>
          + sml::completion<event_run> [ guard::guard_frame_failed<dependencies_type>{} ]
          / action::effect_fail<dependencies_type, error::detokenize_failed>{}

      //------------------------------------------------------------------------------//
      // Immediate outcome channels and idle restoration.
      , sml::state<state_idle> <= sml::state<state_done_channel_decision>
          + sml::completion<event_run> [ guard_done_present{} ]
          / action::effect_emit_done<dependencies_type>{}
      , sml::state<state_idle> <= sml::state<state_done_channel_decision>
          + sml::completion<event_run> [ guard_done_absent{} ]
      , sml::state<state_idle> <= sml::state<state_error_channel_decision>
          + sml::completion<event_run> [ guard_error_present{} ]
          / action::effect_emit_error<dependencies_type>{}
      , sml::state<state_idle> <= sml::state<state_error_channel_decision>
          + sml::completion<event_run> [ guard_error_absent{} ]

      //------------------------------------------------------------------------------//
      // The frame actor owns no cross-dispatch phase data; reset still has an
      // explicit typed actor boundary so its parent can reset all three stages.
      , sml::state<state_idle> <= sml::state<state_idle>
          + sml::event<event::reset> / action::effect_reset<dependencies_type>{}

      //------------------------------------------------------------------------------//
      // RTC makes idle the only externally observable event boundary.
      , sml::state<state_idle> <= sml::state<state_idle>
          + sml::unexpected_event<sml::_> / action::effect_unexpected<dependencies_type>{}
    );
    // clang-format on
  }
};

template <class base_type>
inline bool process_run_event(base_type &base, const event::run &ev) {
  detail::run_ctx ctx{};
  const bool accepted = base.process_event(detail::run_frame{ev, ctx});
  return accepted && ctx.err == action::error_code(error::none);
}

template <class base_type>
inline bool process_reset_event(base_type &base, const event::reset &ev) {
  const bool accepted = base.process_event(ev);
  return accepted && ev.error_out == action::error_code(error::none);
}

template <action::frame_dependencies dependencies_type>
struct sm : private emel::sm<model<dependencies_type>,
                             action::context<dependencies_type>> {
  using base_type =
      emel::sm<model<dependencies_type>, action::context<dependencies_type>>;
  using base_type::is;
  using base_type::visit_current_states;

  explicit sm(const dependencies_type &deps)
      : base_type(action::context<dependencies_type>{deps}) {}

  bool process_event(const event::run &ev) {
    return process_run_event(static_cast<base_type &>(*this), ev);
  }

  bool process_event(const event::reset &ev) {
    return process_reset_event(static_cast<base_type &>(*this), ev);
  }

  template <class external_event_type>
    requires(
        !std::same_as<std::remove_cvref_t<external_event_type>,
                      detail::run_frame> &&
        !std::same_as<std::remove_cvref_t<external_event_type>, event::run> &&
        !std::same_as<std::remove_cvref_t<external_event_type>, event::reset>)
  bool process_event(const external_event_type &ev) {
    return base_type::process_event(ev);
  }
};

// Test-only traced actor. Its policy is closed to sml::logger<T>, so it cannot
// activate queue, defer, locking, or scheduler behavior.
template <action::frame_dependencies dependencies_type, class logger_type>
struct traced_sm : private emel::sm<model<dependencies_type>,
                                    action::context<dependencies_type>,
                                    stateforward::sml::logger<logger_type>> {
  using base_type =
      emel::sm<model<dependencies_type>, action::context<dependencies_type>,
               stateforward::sml::logger<logger_type>>;
  using base_type::is;
  using base_type::visit_current_states;

  traced_sm(const dependencies_type &deps, logger_type &logger)
      : base_type(action::context<dependencies_type>{deps}, logger) {}

  bool process_event(const event::run &ev) {
    return process_run_event(static_cast<base_type &>(*this), ev);
  }

  bool process_event(const event::reset &ev) {
    return process_reset_event(static_cast<base_type &>(*this), ev);
  }

  template <class external_event_type>
    requires(
        !std::same_as<std::remove_cvref_t<external_event_type>,
                      detail::run_frame> &&
        !std::same_as<std::remove_cvref_t<external_event_type>, event::run> &&
        !std::same_as<std::remove_cvref_t<external_event_type>, event::reset>)
  bool process_event(const external_event_type &ev) {
    return base_type::process_event(ev);
  }
};

template <class dependencies_type>
sm(const dependencies_type &) -> sm<dependencies_type>;

} // namespace emel::speech::generator::frame
