#pragma once

#include <stateforward/sml.hpp>

#include "emel/sm.hpp"
#include "emel/speech/codec/mimi/decoder/actions.hpp"
#include "emel/speech/codec/mimi/decoder/context.hpp"
#include "emel/speech/codec/mimi/decoder/events.hpp"
#include "emel/speech/codec/mimi/decoder/guards.hpp"

// Mimi decode actor: one 12.5 Hz latent column -> one 80 ms PCM frame.
// Stage order mirrors the reference: depthwise upsample, codec transformer,
// SEANet decoder; stage outcomes are routed by guards over the per-dispatch
// runtime ctx, never decided inside actions.
namespace emel::speech::codec::mimi::decoder {

struct state_ready {};
struct state_runtime_decision {};
struct state_shape_decision {};
struct state_capacity_decision {};
struct state_upsample_running {};
struct state_transformer_variant_decision {};
struct state_transformer_running {};
struct state_backend_variant_decision {};
struct state_backend_running {};
struct state_success_error_out_decision {};
struct state_success_callback_decision {};
struct state_error_error_out_decision {};
struct state_error_callback_decision {};
struct state_done {};
struct state_errored {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Request validation.
        sml::state<state_runtime_decision> <= *sml::state<state_ready>
          + sml::event<event::decode_run>
          / action::effect_begin_decode{}
      , sml::state<state_shape_decision> <= sml::state<state_runtime_decision>
          + sml::completion<event::decode_run> [ guard::guard_runtime_bound{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_runtime_decision>
          + sml::completion<event::decode_run> [ guard::guard_runtime_unbound{} ]
          / action::effect_mark_runtime_unbound{}

      , sml::state<state_capacity_decision> <= sml::state<state_shape_decision>
          + sml::completion<event::decode_run> [ guard::guard_request_shape_valid{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_shape_decision>
          + sml::completion<event::decode_run> [ guard::guard_request_shape_invalid{} ]
          / action::effect_mark_request_shape_invalid{}

      , sml::state<state_upsample_running> <= sml::state<state_capacity_decision>
          + sml::completion<event::decode_run> [ guard::guard_buffer_capacity_valid{} ]
          / action::effect_run_upsample{}
      , sml::state<state_error_error_out_decision> <= sml::state<state_capacity_decision>
          + sml::completion<event::decode_run> [ guard::guard_buffer_capacity_invalid{} ]
          / action::effect_mark_buffer_capacity_invalid{}

      //------------------------------------------------------------------------------//
      // Compute stages.
      , sml::state<state_transformer_variant_decision> <= sml::state<state_upsample_running>
          + sml::completion<event::decode_run> [ guard::guard_stage_ok{} ]
      , sml::state<state_transformer_running> <= sml::state<state_transformer_variant_decision>
          + sml::completion<event::decode_run> [ guard::guard_proj_f32{} ]
          / action::effect_run_transformer<false>{}
      , sml::state<state_transformer_running> <= sml::state<state_transformer_variant_decision>
          + sml::completion<event::decode_run> [ guard::guard_proj_q8{} ]
          / action::effect_run_transformer<true>{}
      , sml::state<state_error_error_out_decision> <= sml::state<state_upsample_running>
          + sml::completion<event::decode_run> [ guard::guard_stage_failed{} ]
          / action::effect_mark_upsample_failed{}

      , sml::state<state_backend_variant_decision> <= sml::state<state_transformer_running>
          + sml::completion<event::decode_run> [ guard::guard_stage_ok{} ]
      , sml::state<state_backend_running> <= sml::state<state_backend_variant_decision>
          + sml::completion<event::decode_run> [ guard::guard_conv_f32{} ]
          / action::effect_run_backend<false>{}
      , sml::state<state_backend_running> <= sml::state<state_backend_variant_decision>
          + sml::completion<event::decode_run> [ guard::guard_conv_f16{} ]
          / action::effect_run_backend<true>{}
      , sml::state<state_error_error_out_decision> <= sml::state<state_transformer_running>
          + sml::completion<event::decode_run> [ guard::guard_stage_failed{} ]
          / action::effect_mark_transformer_failed{}

      , sml::state<state_success_error_out_decision> <= sml::state<state_backend_running>
          + sml::completion<event::decode_run> [ guard::guard_stage_ok{} ]
      , sml::state<state_error_error_out_decision> <= sml::state<state_backend_running>
          + sml::completion<event::decode_run> [ guard::guard_stage_failed{} ]
          / action::effect_mark_backend_failed{}

      //------------------------------------------------------------------------------//
      // Publish.
      , sml::state<state_success_callback_decision> <= sml::state<state_success_error_out_decision>
          + sml::completion<event::decode_run> [ guard::guard_has_error_out{} ]
          / action::effect_store_error_out{}
      , sml::state<state_success_callback_decision> <= sml::state<state_success_error_out_decision>
          + sml::completion<event::decode_run> [ guard::guard_no_error_out{} ]
      , sml::state<state_error_callback_decision> <= sml::state<state_error_error_out_decision>
          + sml::completion<event::decode_run> [ guard::guard_has_error_out{} ]
          / action::effect_store_error_out{}
      , sml::state<state_error_callback_decision> <= sml::state<state_error_error_out_decision>
          + sml::completion<event::decode_run> [ guard::guard_no_error_out{} ]
      , sml::state<state_done> <= sml::state<state_success_callback_decision>
          + sml::completion<event::decode_run> [ guard::guard_has_done_callback{} ]
          / action::effect_emit_done{}
      , sml::state<state_done> <= sml::state<state_success_callback_decision>
          + sml::completion<event::decode_run> [ guard::guard_no_done_callback{} ]
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::decode_run> [ guard::guard_has_error_callback{} ]
          / action::effect_emit_error{}
      , sml::state<state_errored> <= sml::state<state_error_callback_decision>
          + sml::completion<event::decode_run> [ guard::guard_no_error_callback{} ]
      , sml::state<state_ready> <= sml::state<state_done>
          + sml::completion<event::decode_run>
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::completion<event::decode_run>

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<state_ready> <= sml::state<state_ready> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_runtime_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_shape_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_capacity_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_upsample_running>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_transformer_variant_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_transformer_running>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_backend_variant_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_backend_running>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_success_error_out_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_success_callback_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_error_error_out_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_error_callback_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_done>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_errored>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  sm() = default;

  bool process_event(const event::decode &ev) {
    event::decode_ctx ctx{};
    event::decode_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail::to_error(error::none);
  }
};

} // namespace emel::speech::codec::mimi::decoder
