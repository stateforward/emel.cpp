#pragma once
// benchmark: designed

#include "emel/sm.hpp"
#include "emel/text/generator/decode_wavefront/actions.hpp"
#include "emel/text/generator/decode_wavefront/guards.hpp"

namespace emel::text::generator::decode_wavefront {

// Public alias for the lane thread pool the sm constructor requires, so callers
// (integrators, benchmarks) can name it without reaching into the action
// namespace.
using lane_pool = action::lane_pool;

struct state_idle {};
struct state_validation_decision {};
struct state_group_ready {};
struct state_lane0_decision {};
struct state_lane1_decision {};
struct state_lane2_decision {};
struct state_lane3_decision {};
struct state_lane4_decision {};
struct state_lane5_decision {};
struct state_lane6_decision {};
struct state_lane7_decision {};
struct state_parallel_decision {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Request validation and compatibility grouping.
        sml::state<state_validation_decision> <= *sml::state<state_idle>
                 + sml::event<event::run>
                 / action::effect_begin_run

      , sml::state<state_idle> <= sml::state<state_validation_decision>
                 + sml::completion<event::run>
                 [ guard::guard_invalid_request{} ]
                 / action::effect_reject_invalid_request

      , sml::state<state_idle> <= sml::state<state_validation_decision>
                 + sml::completion<event::run>
                 [ guard::guard_multi_lane_incompatible{} ]
                 / action::effect_reject_incompatible_lanes

      , sml::state<state_group_ready> <= sml::state<state_validation_decision>
                 + sml::completion<event::run>
                 [ guard::guard_single_lane{} ]
                 / action::effect_mark_single_lane

      , sml::state<state_group_ready> <= sml::state<state_validation_decision>
                 + sml::completion<event::run>
                 [ guard::guard_multi_lane_compatible{} ]
                 / action::effect_mark_grouped_lanes

      //------------------------------------------------------------------------------//
      // Bounded lane dispatch. Serial lanes use explicit transition stages;
      // pool-backed multi-lane groups fork/join once inside the RTC chain.
      , sml::state<state_parallel_decision> <= sml::state<state_group_ready>
                 + sml::completion<event::run>
                 [ guard::guard_parallel_dispatch{} ]
                 / action::effect_dispatch_parallel_lanes

      , sml::state<state_lane0_decision> <= sml::state<state_group_ready>
                 + sml::completion<event::run>
                 [ guard::guard_serial_dispatch{} ]
                 / action::effect_dispatch_lane<0>{}

      , sml::state<state_idle> <= sml::state<state_lane0_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_rejected<0>{} ]
                 / action::effect_mark_lane_rejected<0>{}

      , sml::state<state_idle> <= sml::state<state_lane0_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_accepted_and_last<0>{} ]
                 / action::effect_commit_done

      , sml::state<state_lane1_decision> <= sml::state<state_lane0_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_accepted_and_more<0>{} ]
                 / action::effect_dispatch_lane<1>{}

      , sml::state<state_idle> <= sml::state<state_lane1_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_rejected<1>{} ]
                 / action::effect_mark_lane_rejected<1>{}

      , sml::state<state_idle> <= sml::state<state_lane1_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_accepted_and_last<1>{} ]
                 / action::effect_commit_done

      , sml::state<state_lane2_decision> <= sml::state<state_lane1_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_accepted_and_more<1>{} ]
                 / action::effect_dispatch_lane<2>{}

      , sml::state<state_idle> <= sml::state<state_lane2_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_rejected<2>{} ]
                 / action::effect_mark_lane_rejected<2>{}

      , sml::state<state_idle> <= sml::state<state_lane2_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_accepted_and_last<2>{} ]
                 / action::effect_commit_done

      , sml::state<state_lane3_decision> <= sml::state<state_lane2_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_accepted_and_more<2>{} ]
                 / action::effect_dispatch_lane<3>{}

      , sml::state<state_idle> <= sml::state<state_lane3_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_rejected<3>{} ]
                 / action::effect_mark_lane_rejected<3>{}

      , sml::state<state_idle> <= sml::state<state_lane3_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_accepted_and_last<3>{} ]
                 / action::effect_commit_done

      , sml::state<state_lane4_decision> <= sml::state<state_lane3_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_accepted_and_more<3>{} ]
                 / action::effect_dispatch_lane<4>{}

      , sml::state<state_idle> <= sml::state<state_lane4_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_rejected<4>{} ]
                 / action::effect_mark_lane_rejected<4>{}

      , sml::state<state_idle> <= sml::state<state_lane4_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_accepted_and_last<4>{} ]
                 / action::effect_commit_done

      , sml::state<state_lane5_decision> <= sml::state<state_lane4_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_accepted_and_more<4>{} ]
                 / action::effect_dispatch_lane<5>{}

      , sml::state<state_idle> <= sml::state<state_lane5_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_rejected<5>{} ]
                 / action::effect_mark_lane_rejected<5>{}

      , sml::state<state_idle> <= sml::state<state_lane5_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_accepted_and_last<5>{} ]
                 / action::effect_commit_done

      , sml::state<state_lane6_decision> <= sml::state<state_lane5_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_accepted_and_more<5>{} ]
                 / action::effect_dispatch_lane<6>{}

      , sml::state<state_idle> <= sml::state<state_lane6_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_rejected<6>{} ]
                 / action::effect_mark_lane_rejected<6>{}

      , sml::state<state_idle> <= sml::state<state_lane6_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_accepted_and_last<6>{} ]
                 / action::effect_commit_done

      , sml::state<state_lane7_decision> <= sml::state<state_lane6_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_accepted_and_more<6>{} ]
                 / action::effect_dispatch_lane<7>{}

      , sml::state<state_idle> <= sml::state<state_lane7_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_rejected<7>{} ]
                 / action::effect_mark_lane_rejected<7>{}

      , sml::state<state_idle> <= sml::state<state_lane7_decision>
                 + sml::completion<event::run>
                 [ guard::guard_lane_accepted_and_last<7>{} ]
                 / action::effect_commit_done

      , sml::state<state_idle> <= sml::state<state_parallel_decision>
                 + sml::completion<event::run>
                 [ guard::guard_parallel_lane_rejected<0>{} ]
                 / action::effect_mark_lane_rejected<0>{}

      , sml::state<state_idle> <= sml::state<state_parallel_decision>
                 + sml::completion<event::run>
                 [ guard::guard_parallel_lane_rejected<1>{} ]
                 / action::effect_mark_lane_rejected<1>{}

      , sml::state<state_idle> <= sml::state<state_parallel_decision>
                 + sml::completion<event::run>
                 [ guard::guard_parallel_lane_rejected<2>{} ]
                 / action::effect_mark_lane_rejected<2>{}

      , sml::state<state_idle> <= sml::state<state_parallel_decision>
                 + sml::completion<event::run>
                 [ guard::guard_parallel_lane_rejected<3>{} ]
                 / action::effect_mark_lane_rejected<3>{}

      , sml::state<state_idle> <= sml::state<state_parallel_decision>
                 + sml::completion<event::run>
                 [ guard::guard_parallel_lane_rejected<4>{} ]
                 / action::effect_mark_lane_rejected<4>{}

      , sml::state<state_idle> <= sml::state<state_parallel_decision>
                 + sml::completion<event::run>
                 [ guard::guard_parallel_lane_rejected<5>{} ]
                 / action::effect_mark_lane_rejected<5>{}

      , sml::state<state_idle> <= sml::state<state_parallel_decision>
                 + sml::completion<event::run>
                 [ guard::guard_parallel_lane_rejected<6>{} ]
                 / action::effect_mark_lane_rejected<6>{}

      , sml::state<state_idle> <= sml::state<state_parallel_decision>
                 + sml::completion<event::run>
                 [ guard::guard_parallel_lane_rejected<7>{} ]
                 / action::effect_mark_lane_rejected<7>{}

      , sml::state<state_idle> <= sml::state<state_parallel_decision>
                 + sml::completion<event::run>
                 [ guard::guard_parallel_all_lanes_accepted{} ]
                 / action::effect_commit_done

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<state_idle> <= sml::state<state_idle> + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
      , sml::state<state_idle> <= sml::state<state_validation_decision>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
      , sml::state<state_idle> <= sml::state<state_group_ready>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
      , sml::state<state_idle> <= sml::state<state_lane0_decision>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
      , sml::state<state_idle> <= sml::state<state_lane1_decision>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
      , sml::state<state_idle> <= sml::state<state_lane2_decision>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
      , sml::state<state_idle> <= sml::state<state_lane3_decision>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
      , sml::state<state_idle> <= sml::state<state_lane4_decision>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
      , sml::state<state_idle> <= sml::state<state_lane5_decision>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
      , sml::state<state_idle> <= sml::state<state_lane6_decision>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
      , sml::state<state_idle> <= sml::state<state_lane7_decision>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
      , sml::state<state_idle> <= sml::state<state_parallel_decision>
                 + sml::unexpected_event<sml::_>
                 / action::effect_on_unexpected
    );
    // clang-format on
  }
};

using static_co_policy =
    emel::policy::coroutine_scheduler<emel::policy::fifo_scheduler<16u, 64u>>;

struct sm : public emel::co_sm<model, action::context, static_co_policy> {
  using base_type = emel::co_sm<model, action::context, static_co_policy>;
  using base_type::is;
  using base_type::visit_current_states;

  sm() = default;
  explicit sm(action::lane_pool & pool) : base_type(action::context{.pool = &pool}) {}

  bool process_event(const event::run & ev) {
    const bool accepted = process_event_async(ev).result();
    return accepted && ev.out.err == emel::error::cast(error::none);
  }

  emel::bool_task process_event_async(const event::run & ev) {
    const bool accepted = base_type::process_event_async(ev).result();
    return emel::bool_task::from_value(
        accepted && ev.out.err == emel::error::cast(error::none));
  }
};

}  // namespace emel::text::generator::decode_wavefront
