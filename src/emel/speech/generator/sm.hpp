#pragma once

// benchmark: designed

#include <utility>

#include "emel/sm.hpp"
#include "emel/speech/generator/actions.hpp"
#include "emel/speech/generator/context.hpp"
#include "emel/speech/generator/events.hpp"
#include "emel/speech/generator/guards.hpp"

namespace emel::speech::generator {

struct state_uninitialized {};
struct state_initialize_done_channel_decision {};
struct state_initialize_error_ready_channel_decision {};
struct state_ready {};

struct state_generate_conditioning {};
struct state_generate_prefill {};
struct state_generate_predict {};
struct state_generate_sample {};
struct state_generate_decode {};
struct state_generate_postprocess {};
struct state_generate_error_ready_channel_decision {};
struct state_generate_error_uninitialized_channel_decision {};
struct state_generate_error_errored_channel_decision {};

struct state_stream_encode {};
struct state_stream_tokenize {};
struct state_stream_prefill {};
struct state_stream_predict {};
struct state_stream_sample {};
struct state_stream_detokenize {};
struct state_stream_decode {};
struct state_stream_postprocess {};
struct state_stream_error_ready_channel_decision {};
struct state_stream_error_uninitialized_channel_decision {};
struct state_stream_error_errored_channel_decision {};

struct state_flush_decode {};
struct state_flush_postprocess {};
struct state_flush_error_ready_channel_decision {};
struct state_flush_error_uninitialized_channel_decision {};
struct state_flush_error_errored_channel_decision {};
struct state_errored {};

/*
generic speech generator scaffold (single source of truth)

state purpose
- state_ready accepts text-to-speech generation, duplex stream frames, and
  decoder flush requests through separate event types so incompatible request
  shapes are not representable as one flag-driven payload.
- generate states reserve the text/reference conditioning, prefill,
  acoustic-representation prediction, sampling, decoding, and postprocessing
  phases used by text/reference synthesis systems.
- stream states reserve the input encode, token alignment, prediction,
  sampling, detokenization, streaming decode, and postprocessing phases used
  by duplex streaming systems.

control invariants
- the machine contains no model-family names, contracts, routes, constants, or
  runtime handler tables. Model-family binding remains outside this component.
- planner, memory, graph, sampler, and kernel actors are injected by reference
  and remain independently owned actors.
- every currently uncut phase is explicit in the transition graph. Valid
  generation requests finish with cutover_pending rather than silently
  succeeding or dispatching the maintained model-family implementation.
- request-local status remains in typed runtime events; context stores only
  persistent injected collaborators.

cutover contract
- each reserved phase will be replaced incrementally by synchronous child
  dispatch while preserving this phase order and RTC boundary.
- decoder cadence, streaming versus synthesis input, callback presence, and
  request validation remain guard-selected transitions.
*/
struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    using init_run = event::initialize_run;
    using generate_run = event::generate_run;
    using stream_run = event::stream_frame_run;
    using flush_run = event::flush_run;

    using init_error_present = guard::guard_error_callback_present<init_run>;
    using init_error_absent = guard::guard_error_callback_absent<init_run>;
    using generate_error_present =
        guard::guard_error_callback_present<generate_run>;
    using generate_error_absent =
        guard::guard_error_callback_absent<generate_run>;
    using stream_error_present =
        guard::guard_error_callback_present<stream_run>;
    using stream_error_absent = guard::guard_error_callback_absent<stream_run>;
    using flush_error_present = guard::guard_error_callback_present<flush_run>;
    using flush_error_absent = guard::guard_error_callback_absent<flush_run>;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Construction-time dependency injection makes initialization a pure
      // lifecycle transition. Duplicate initialization remains observable.
        sml::state<state_initialize_done_channel_decision> <=
          *sml::state<state_uninitialized> + sml::event<init_run>
          / action::effect_accept_initialize{}
      , sml::state<state_initialize_error_ready_channel_decision> <=
          sml::state<state_ready> + sml::event<init_run>
          / action::effect_fail_initialize<error::already_initialized>{}
      , sml::state<state_initialize_done_channel_decision> <=
          sml::state<state_errored> + sml::event<init_run>
          / action::effect_accept_initialize{}
      , sml::state<state_ready> <= sml::state<state_initialize_done_channel_decision>
          + sml::completion<init_run>
          [ guard::guard_initialize_done_callback_present{} ]
          / action::effect_emit_initialize_done{}
      , sml::state<state_ready> <= sml::state<state_initialize_done_channel_decision>
          + sml::completion<init_run>
          [ guard::guard_initialize_done_callback_absent{} ]
      , sml::state<state_ready> <=
          sml::state<state_initialize_error_ready_channel_decision>
          + sml::completion<init_run> [ init_error_present{} ]
          / action::effect_emit_initialize_error{}
      , sml::state<state_ready> <=
          sml::state<state_initialize_error_ready_channel_decision>
          + sml::completion<init_run> [ init_error_absent{} ]

      //------------------------------------------------------------------------------//
      // Text/reference synthesis phase scaffold.
      , sml::state<state_generate_conditioning> <= sml::state<state_ready>
          + sml::event<generate_run> [ guard::guard_generate_request_valid{} ]
          / action::effect_prepare_generate{}
      , sml::state<state_generate_error_ready_channel_decision> <=
          sml::state<state_ready> + sml::event<generate_run>
          [ guard::guard_generate_request_invalid{} ]
          / action::effect_fail_generate<error::invalid_request>{}
      , sml::state<state_generate_prefill> <= sml::state<state_generate_conditioning>
          + sml::completion<generate_run>
      , sml::state<state_generate_predict> <= sml::state<state_generate_prefill>
          + sml::completion<generate_run>
      , sml::state<state_generate_sample> <= sml::state<state_generate_predict>
          + sml::completion<generate_run>
      , sml::state<state_generate_decode> <= sml::state<state_generate_sample>
          + sml::completion<generate_run>
      , sml::state<state_generate_postprocess> <= sml::state<state_generate_decode>
          + sml::completion<generate_run>
      , sml::state<state_generate_error_ready_channel_decision> <=
          sml::state<state_generate_postprocess> + sml::completion<generate_run>
          / action::effect_fail_generate<error::cutover_pending>{}
      , sml::state<state_ready> <=
          sml::state<state_generate_error_ready_channel_decision>
          + sml::completion<generate_run> [ generate_error_present{} ]
          / action::effect_emit_generation_error{}
      , sml::state<state_ready> <=
          sml::state<state_generate_error_ready_channel_decision>
          + sml::completion<generate_run> [ generate_error_absent{} ]

      //------------------------------------------------------------------------------//
      // Duplex streaming phase scaffold.
      , sml::state<state_stream_encode> <= sml::state<state_ready>
          + sml::event<stream_run> [ guard::guard_stream_request_valid{} ]
          / action::effect_prepare_stream_frame{}
      , sml::state<state_stream_error_ready_channel_decision> <=
          sml::state<state_ready> + sml::event<stream_run>
          [ guard::guard_stream_request_invalid{} ]
          / action::effect_fail_stream_frame<error::invalid_request>{}
      , sml::state<state_stream_tokenize> <= sml::state<state_stream_encode>
          + sml::completion<stream_run>
      , sml::state<state_stream_prefill> <= sml::state<state_stream_tokenize>
          + sml::completion<stream_run>
      , sml::state<state_stream_predict> <= sml::state<state_stream_prefill>
          + sml::completion<stream_run>
      , sml::state<state_stream_sample> <= sml::state<state_stream_predict>
          + sml::completion<stream_run>
      , sml::state<state_stream_detokenize> <= sml::state<state_stream_sample>
          + sml::completion<stream_run>
      , sml::state<state_stream_decode> <= sml::state<state_stream_detokenize>
          + sml::completion<stream_run>
      , sml::state<state_stream_postprocess> <= sml::state<state_stream_decode>
          + sml::completion<stream_run>
      , sml::state<state_stream_error_ready_channel_decision> <=
          sml::state<state_stream_postprocess> + sml::completion<stream_run>
          / action::effect_fail_stream_frame<error::cutover_pending>{}
      , sml::state<state_ready> <=
          sml::state<state_stream_error_ready_channel_decision>
          + sml::completion<stream_run> [ stream_error_present{} ]
          / action::effect_emit_stream_frame_error{}
      , sml::state<state_ready> <=
          sml::state<state_stream_error_ready_channel_decision>
          + sml::completion<stream_run> [ stream_error_absent{} ]

      //------------------------------------------------------------------------------//
      // Buffered or streaming decoder flush scaffold.
      , sml::state<state_flush_decode> <= sml::state<state_ready>
          + sml::event<flush_run> [ guard::guard_flush_request_valid{} ]
          / action::effect_prepare_flush{}
      , sml::state<state_flush_error_ready_channel_decision> <=
          sml::state<state_ready> + sml::event<flush_run>
          [ guard::guard_flush_request_invalid{} ]
          / action::effect_fail_flush<error::invalid_request>{}
      , sml::state<state_flush_postprocess> <= sml::state<state_flush_decode>
          + sml::completion<flush_run>
      , sml::state<state_flush_error_ready_channel_decision> <=
          sml::state<state_flush_postprocess> + sml::completion<flush_run>
          / action::effect_fail_flush<error::cutover_pending>{}
      , sml::state<state_ready> <=
          sml::state<state_flush_error_ready_channel_decision>
          + sml::completion<flush_run> [ flush_error_present{} ]
          / action::effect_emit_flush_error{}
      , sml::state<state_ready> <=
          sml::state<state_flush_error_ready_channel_decision>
          + sml::completion<flush_run> [ flush_error_absent{} ]

      //------------------------------------------------------------------------------//
      // Known requests before initialization and after unexpected events.
      , sml::state<state_generate_error_uninitialized_channel_decision> <=
          sml::state<state_uninitialized> + sml::event<generate_run>
          / action::effect_fail_generate<error::uninitialized>{}
      , sml::state<state_uninitialized> <=
          sml::state<state_generate_error_uninitialized_channel_decision>
          + sml::completion<generate_run> [ generate_error_present{} ]
          / action::effect_emit_generation_error{}
      , sml::state<state_uninitialized> <=
          sml::state<state_generate_error_uninitialized_channel_decision>
          + sml::completion<generate_run> [ generate_error_absent{} ]
      , sml::state<state_stream_error_uninitialized_channel_decision> <=
          sml::state<state_uninitialized> + sml::event<stream_run>
          / action::effect_fail_stream_frame<error::uninitialized>{}
      , sml::state<state_uninitialized> <=
          sml::state<state_stream_error_uninitialized_channel_decision>
          + sml::completion<stream_run> [ stream_error_present{} ]
          / action::effect_emit_stream_frame_error{}
      , sml::state<state_uninitialized> <=
          sml::state<state_stream_error_uninitialized_channel_decision>
          + sml::completion<stream_run> [ stream_error_absent{} ]
      , sml::state<state_flush_error_uninitialized_channel_decision> <=
          sml::state<state_uninitialized> + sml::event<flush_run>
          / action::effect_fail_flush<error::uninitialized>{}
      , sml::state<state_uninitialized> <=
          sml::state<state_flush_error_uninitialized_channel_decision>
          + sml::completion<flush_run> [ flush_error_present{} ]
          / action::effect_emit_flush_error{}
      , sml::state<state_uninitialized> <=
          sml::state<state_flush_error_uninitialized_channel_decision>
          + sml::completion<flush_run> [ flush_error_absent{} ]
      , sml::state<state_uninitialized> <= sml::state<state_uninitialized>
          + sml::event<event::reset>
          / action::effect_reject<error::uninitialized>{}

      , sml::state<state_generate_error_errored_channel_decision> <=
          sml::state<state_errored> + sml::event<generate_run>
          / action::effect_fail_generate<error::internal_error>{}
      , sml::state<state_errored> <=
          sml::state<state_generate_error_errored_channel_decision>
          + sml::completion<generate_run> [ generate_error_present{} ]
          / action::effect_emit_generation_error{}
      , sml::state<state_errored> <=
          sml::state<state_generate_error_errored_channel_decision>
          + sml::completion<generate_run> [ generate_error_absent{} ]
      , sml::state<state_stream_error_errored_channel_decision> <=
          sml::state<state_errored> + sml::event<stream_run>
          / action::effect_fail_stream_frame<error::internal_error>{}
      , sml::state<state_errored> <=
          sml::state<state_stream_error_errored_channel_decision>
          + sml::completion<stream_run> [ stream_error_present{} ]
          / action::effect_emit_stream_frame_error{}
      , sml::state<state_errored> <=
          sml::state<state_stream_error_errored_channel_decision>
          + sml::completion<stream_run> [ stream_error_absent{} ]
      , sml::state<state_flush_error_errored_channel_decision> <=
          sml::state<state_errored> + sml::event<flush_run>
          / action::effect_fail_flush<error::internal_error>{}
      , sml::state<state_errored> <=
          sml::state<state_flush_error_errored_channel_decision>
          + sml::completion<flush_run> [ flush_error_present{} ]
          / action::effect_emit_flush_error{}
      , sml::state<state_errored> <=
          sml::state<state_flush_error_errored_channel_decision>
          + sml::completion<flush_run> [ flush_error_absent{} ]

      //------------------------------------------------------------------------------//
      // Reset preserves injected collaborators and returns to ready.
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::reset> / action::effect_reset{}
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::event<event::reset> / action::effect_reset{}

      //------------------------------------------------------------------------------//
      // Unknown external events are observable through state_errored.
      , sml::state<state_errored> <= sml::state<state_uninitialized>
          + sml::unexpected_event<sml::_> / action::effect_unexpected{}
      , sml::state<state_errored> <= sml::state<state_ready>
          + sml::unexpected_event<sml::_> / action::effect_unexpected{}
      , sml::state<state_errored> <= sml::state<state_errored>
          + sml::unexpected_event<sml::_> / action::effect_unexpected{}
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  explicit sm(const dependencies &deps) : base_type(std::in_place, deps) {}

  bool process_event(const event::initialize &ev) {
    event::initialize_ctx ctx{};
    const bool accepted =
        base_type::process_event(event::initialize_run{ev, ctx});
    return accepted && ctx.err == action::error_code(error::none);
  }

  bool process_event(const event::generate &ev) {
    event::generate_ctx ctx{};
    const bool accepted =
        base_type::process_event(event::generate_run{ev, ctx});
    return accepted && ctx.err == action::error_code(error::none);
  }

  bool process_event(const event::stream_frame &ev) {
    event::stream_frame_ctx ctx{};
    const bool accepted =
        base_type::process_event(event::stream_frame_run{ev, ctx});
    return accepted && ctx.err == action::error_code(error::none);
  }

  bool process_event(const event::flush &ev) {
    event::flush_ctx ctx{};
    const bool accepted = base_type::process_event(event::flush_run{ev, ctx});
    return accepted && ctx.err == action::error_code(error::none);
  }

  bool process_event(const event::reset &ev) {
    const bool accepted = base_type::process_event(ev);
    return accepted && ev.error_out == action::error_code(error::none);
  }
};

using Generator = sm;

} // namespace emel::speech::generator
