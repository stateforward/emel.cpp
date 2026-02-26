#pragma once

#include "emel/generator/actions.hpp"
#include "emel/generator/events.hpp"
#include "emel/generator/guards.hpp"
#include "emel/sm.hpp"

namespace emel::generator {

struct ready {};
struct conditioning {};
struct conditioning_decision {};
struct planning {};
struct planning_decision {};
struct prefill {};
struct prefill_decision {};
struct decode_compute {};
struct decode_compute_decision {};
struct decode_sample {};
struct decode_sample_decision {};
struct decode_render {};
struct decode_render_decision {};
struct generate_decision {};
struct unexpected_event {};

/**
 * generator orchestration scaffold.
 *
 * state purposes:
 * - `ready`: idle state awaiting generation intent.
 * - `conditioning`/`conditioning_decision`: request and classify conditioning phase completion.
 * - `planning`/`planning_decision`: request and classify planning phase completion.
 * - `prefill`/`prefill_decision`: request and classify prefill phase completion.
 * - `decode_*` and `*_decision`: bounded decode loop split into explicit compute/sample/render phases.
 * - `generate_decision`: centralized success/error outcome reducer.
 * - `unexpected_event`: catchall for unhandled external events.
 *
 * guard semantics:
 * - `valid_generate`/`invalid_generate`: validate `event::generate_run` shape and decode bound.
 * - `phase_ok`/`phase_failed`: classify each phase without action-side branching.
 * - `decode_should_continue`/`decode_complete`: enforce bounded decode progress.
 *
 * action side effects:
 * - `begin_generate`: initialize per-dispatch runtime context.
 * - `request_*`: execute one bounded phase step and report through `generate_ctx`.
 * - `dispatch_*`: publish done/error outcomes synchronously before dispatch returns.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Request validation.
        sml::state<conditioning> <= *sml::state<ready> + sml::event<event::generate_run>
                 [ guard::valid_generate{} ]
                 / action::begin_generate

      , sml::state<generate_decision> <= sml::state<ready> + sml::event<event::generate_run>
                 [ guard::invalid_generate{} ]
                 / action::reject_invalid_generate

      //------------------------------------------------------------------------------//
      // Conditioning phase.
      , sml::state<conditioning_decision> <= sml::state<conditioning> + sml::completion<event::generate_run>
                 / action::request_conditioning

      , sml::state<generate_decision> <= sml::state<conditioning_decision> + sml::completion<event::generate_run>
                 [ guard::phase_failed{} ]

      , sml::state<planning> <= sml::state<conditioning_decision> + sml::completion<event::generate_run>
                 [ guard::phase_ok{} ]

      //------------------------------------------------------------------------------//
      // Planning phase.
      , sml::state<planning_decision> <= sml::state<planning> + sml::completion<event::generate_run>
                 / action::request_planning

      , sml::state<generate_decision> <= sml::state<planning_decision> + sml::completion<event::generate_run>
                 [ guard::phase_failed{} ]

      , sml::state<prefill> <= sml::state<planning_decision> + sml::completion<event::generate_run>
                 [ guard::phase_ok{} ]

      //------------------------------------------------------------------------------//
      // Prefill phase.
      , sml::state<prefill_decision> <= sml::state<prefill> + sml::completion<event::generate_run>
                 / action::request_prefill

      , sml::state<generate_decision> <= sml::state<prefill_decision> + sml::completion<event::generate_run>
                 [ guard::phase_failed{} ]

      , sml::state<decode_compute> <= sml::state<prefill_decision> + sml::completion<event::generate_run>
                 [ guard::phase_ok{} ]

      //------------------------------------------------------------------------------//
      // Decode compute phase.
      , sml::state<decode_compute_decision> <= sml::state<decode_compute> + sml::completion<event::generate_run>
                 / action::request_decode_compute

      , sml::state<generate_decision> <= sml::state<decode_compute_decision> +
               sml::completion<event::generate_run>
                 [ guard::phase_failed{} ]

      , sml::state<decode_sample> <= sml::state<decode_compute_decision> + sml::completion<event::generate_run>
                 [ guard::phase_ok{} ]

      //------------------------------------------------------------------------------//
      // Decode sample phase.
      , sml::state<decode_sample_decision> <= sml::state<decode_sample> + sml::completion<event::generate_run>
                 / action::request_decode_sample

      , sml::state<generate_decision> <= sml::state<decode_sample_decision> +
               sml::completion<event::generate_run>
                 [ guard::phase_failed{} ]

      , sml::state<decode_render> <= sml::state<decode_sample_decision> + sml::completion<event::generate_run>
                 [ guard::phase_ok{} ]

      //------------------------------------------------------------------------------//
      // Decode render phase.
      , sml::state<decode_render_decision> <= sml::state<decode_render> + sml::completion<event::generate_run>
                 / action::request_decode_render

      , sml::state<generate_decision> <= sml::state<decode_render_decision> +
               sml::completion<event::generate_run>
                 [ guard::phase_failed{} ]

      , sml::state<decode_compute> <= sml::state<decode_render_decision> + sml::completion<event::generate_run>
                 [ guard::decode_should_continue{} ]

      , sml::state<generate_decision> <= sml::state<decode_render_decision> + sml::completion<event::generate_run>
                 [ guard::decode_complete{} ]

      //------------------------------------------------------------------------------//
      // Finalization and outcome dispatch.
      , sml::state<ready> <= sml::state<generate_decision> + sml::completion<event::generate_run>
                 [ guard::phase_ok_with_error_out{} ]
                 / action::dispatch_done_with_error_out

      , sml::state<ready> <= sml::state<generate_decision> + sml::completion<event::generate_run>
                 [ guard::phase_ok_without_error_out{} ]
                 / action::dispatch_done_without_error_out

      , sml::state<ready> <= sml::state<generate_decision> + sml::completion<event::generate_run>
                 [ guard::phase_failed_with_dispatch_and_error_out{} ]
                 / action::dispatch_error_with_dispatch_and_error_out

      , sml::state<ready> <= sml::state<generate_decision> + sml::completion<event::generate_run>
                 [ guard::phase_failed_with_dispatch_only{} ]
                 / action::dispatch_error_with_dispatch_only

      , sml::state<ready> <= sml::state<generate_decision> + sml::completion<event::generate_run>
                 [ guard::phase_failed_with_error_out_only{} ]
                 / action::dispatch_error_with_error_out_only

      , sml::state<ready> <= sml::state<generate_decision> + sml::completion<event::generate_run>
                 [ guard::phase_failed_without_error_channels{} ]
                 / action::dispatch_error_without_error_channels

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<generate_decision> <= sml::state<conditioning> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<generate_decision> <= sml::state<conditioning_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<generate_decision> <= sml::state<planning> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<generate_decision> <= sml::state<planning_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<generate_decision> <= sml::state<prefill> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<generate_decision> <= sml::state<prefill_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<generate_decision> <= sml::state<decode_compute> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<generate_decision> <= sml::state<decode_compute_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<generate_decision> <= sml::state<decode_sample> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<generate_decision> <= sml::state<decode_sample_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<generate_decision> <= sml::state<decode_render> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<generate_decision> <= sml::state<decode_render_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<ready> <= sml::state<generate_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<ready> <= sml::state<unexpected_event> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm_with_context<model, action::context> {
  using base_type = emel::sm_with_context<model, action::context>;
  using base_type::base_type;
  using base_type::process_event;

  bool process_event(const event::generate & ev) {
    event::generate_ctx ctx{};
    event::generate_run evt{ev, ctx};
    const bool accepted = base_type::process_event(evt);
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};

}  // namespace emel::generator
