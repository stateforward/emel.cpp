#pragma once
// benchmark: designed

#include <stateforward/sml.hpp>

#include "emel/sm.hpp"
#include "emel/speech/codec/mimi/quantizer/actions.hpp"
#include "emel/speech/codec/mimi/quantizer/context.hpp"
#include "emel/speech/codec/mimi/quantizer/events.hpp"
#include "emel/speech/codec/mimi/quantizer/guards.hpp"

// Mimi split-RVQ actor: latent column <-> n_q codebook indexes. The encode
// and decode flows are separate explicit state chains; every error route is
// decided by pure validation guards BEFORE the compute action runs, so the
// compute stages are non-failing and chain unconditionally.
namespace emel::speech::codec::mimi::quantizer {

struct state_ready {};
struct state_encode_runtime_decision {};
struct state_encode_shape_decision {};
struct state_encode_variant_decision {};
struct state_encode_running {};
struct state_encode_success_error_out_decision {};
struct state_encode_success_callback_decision {};
struct state_encode_error_error_out_decision {};
struct state_encode_error_callback_decision {};
struct state_encode_done {};
struct state_encode_errored {};
struct state_decode_runtime_decision {};
struct state_decode_shape_decision {};
struct state_decode_codes_decision {};
struct state_decode_variant_decision {};
struct state_decode_running {};
struct state_decode_success_error_out_decision {};
struct state_decode_success_callback_decision {};
struct state_decode_error_error_out_decision {};
struct state_decode_error_callback_decision {};
struct state_decode_done {};
struct state_decode_errored {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;
    using encode_run = event::encode_run;
    using decode_run = event::decode_run;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Encode flow.
        sml::state<state_encode_runtime_decision> <= *sml::state<state_ready>
          + sml::event<encode_run>
      , sml::state<state_encode_shape_decision> <= sml::state<state_encode_runtime_decision>
          + sml::completion<encode_run> [ guard::guard_runtime_bound<encode_run>{} ]
      , sml::state<state_encode_error_error_out_decision> <= sml::state<state_encode_runtime_decision>
          + sml::completion<encode_run> [ guard::guard_runtime_unbound<encode_run>{} ]
          / action::effect_mark_runtime_unbound<encode_run>{}
      , sml::state<state_encode_variant_decision> <= sml::state<state_encode_shape_decision>
          + sml::completion<encode_run> [ guard::guard_encode_shape_valid{} ]
      , sml::state<state_encode_running> <= sml::state<state_encode_variant_decision>
          + sml::completion<encode_run> [ guard::guard_class_f32<encode_run>{} ]
          / action::effect_run_quantize<false, false>{}
      , sml::state<state_encode_running> <= sml::state<state_encode_variant_decision>
          + sml::completion<encode_run> [ guard::guard_class_f16<encode_run>{} ]
          / action::effect_run_quantize<true, false>{}
      , sml::state<state_encode_running> <= sml::state<state_encode_variant_decision>
          + sml::completion<encode_run> [ guard::guard_class_q8<encode_run>{} ]
          / action::effect_run_quantize<true, true>{}
      , sml::state<state_encode_error_error_out_decision> <= sml::state<state_encode_shape_decision>
          + sml::completion<encode_run> [ guard::guard_encode_shape_invalid{} ]
          / action::effect_mark_request_shape_invalid<encode_run>{}
      , sml::state<state_encode_success_error_out_decision> <= sml::state<state_encode_running>
          + sml::completion<encode_run>
      , sml::state<state_encode_success_callback_decision> <= sml::state<state_encode_success_error_out_decision>
          + sml::completion<encode_run> [ guard::guard_has_error_out<encode_run>{} ]
          / action::effect_store_error_out<encode_run>{}
      , sml::state<state_encode_success_callback_decision> <= sml::state<state_encode_success_error_out_decision>
          + sml::completion<encode_run> [ guard::guard_no_error_out<encode_run>{} ]
      , sml::state<state_encode_error_callback_decision> <= sml::state<state_encode_error_error_out_decision>
          + sml::completion<encode_run> [ guard::guard_has_error_out<encode_run>{} ]
          / action::effect_store_error_out<encode_run>{}
      , sml::state<state_encode_error_callback_decision> <= sml::state<state_encode_error_error_out_decision>
          + sml::completion<encode_run> [ guard::guard_no_error_out<encode_run>{} ]
      , sml::state<state_encode_done> <= sml::state<state_encode_success_callback_decision>
          + sml::completion<encode_run> [ guard::guard_has_done_callback<encode_run>{} ]
          / action::effect_emit_encode_done{}
      , sml::state<state_encode_done> <= sml::state<state_encode_success_callback_decision>
          + sml::completion<encode_run> [ guard::guard_no_done_callback<encode_run>{} ]
      , sml::state<state_encode_errored> <= sml::state<state_encode_error_callback_decision>
          + sml::completion<encode_run> [ guard::guard_has_error_callback<encode_run>{} ]
          / action::effect_emit_encode_error{}
      , sml::state<state_encode_errored> <= sml::state<state_encode_error_callback_decision>
          + sml::completion<encode_run> [ guard::guard_no_error_callback<encode_run>{} ]
      , sml::state<state_ready> <= sml::state<state_encode_done>
          + sml::completion<encode_run>
      , sml::state<state_ready> <= sml::state<state_encode_errored>
          + sml::completion<encode_run>

      //------------------------------------------------------------------------------//
      // Decode flow.
      , sml::state<state_decode_runtime_decision> <= sml::state<state_ready>
          + sml::event<decode_run>
      , sml::state<state_decode_shape_decision> <= sml::state<state_decode_runtime_decision>
          + sml::completion<decode_run> [ guard::guard_runtime_bound<decode_run>{} ]
      , sml::state<state_decode_error_error_out_decision> <= sml::state<state_decode_runtime_decision>
          + sml::completion<decode_run> [ guard::guard_runtime_unbound<decode_run>{} ]
          / action::effect_mark_runtime_unbound<decode_run>{}
      , sml::state<state_decode_codes_decision> <= sml::state<state_decode_shape_decision>
          + sml::completion<decode_run> [ guard::guard_decode_shape_valid{} ]
      , sml::state<state_decode_variant_decision> <= sml::state<state_decode_codes_decision>
          + sml::completion<decode_run> [ guard::guard_decode_codes_valid{} ]
      , sml::state<state_decode_error_error_out_decision> <= sml::state<state_decode_codes_decision>
          + sml::completion<decode_run> [ guard::guard_decode_codes_invalid{} ]
          / action::effect_mark_code_range_invalid{}
      , sml::state<state_decode_running> <= sml::state<state_decode_variant_decision>
          + sml::completion<decode_run> [ guard::guard_class_f32<decode_run>{} ]
          / action::effect_run_dequantize<false, false>{}
      , sml::state<state_decode_running> <= sml::state<state_decode_variant_decision>
          + sml::completion<decode_run> [ guard::guard_class_f16<decode_run>{} ]
          / action::effect_run_dequantize<true, false>{}
      , sml::state<state_decode_running> <= sml::state<state_decode_variant_decision>
          + sml::completion<decode_run> [ guard::guard_class_q8<decode_run>{} ]
          / action::effect_run_dequantize<true, true>{}
      , sml::state<state_decode_error_error_out_decision> <= sml::state<state_decode_shape_decision>
          + sml::completion<decode_run> [ guard::guard_decode_shape_invalid{} ]
          / action::effect_mark_request_shape_invalid<decode_run>{}
      , sml::state<state_decode_success_error_out_decision> <= sml::state<state_decode_running>
          + sml::completion<decode_run>
      , sml::state<state_decode_success_callback_decision> <= sml::state<state_decode_success_error_out_decision>
          + sml::completion<decode_run> [ guard::guard_has_error_out<decode_run>{} ]
          / action::effect_store_error_out<decode_run>{}
      , sml::state<state_decode_success_callback_decision> <= sml::state<state_decode_success_error_out_decision>
          + sml::completion<decode_run> [ guard::guard_no_error_out<decode_run>{} ]
      , sml::state<state_decode_error_callback_decision> <= sml::state<state_decode_error_error_out_decision>
          + sml::completion<decode_run> [ guard::guard_has_error_out<decode_run>{} ]
          / action::effect_store_error_out<decode_run>{}
      , sml::state<state_decode_error_callback_decision> <= sml::state<state_decode_error_error_out_decision>
          + sml::completion<decode_run> [ guard::guard_no_error_out<decode_run>{} ]
      , sml::state<state_decode_done> <= sml::state<state_decode_success_callback_decision>
          + sml::completion<decode_run> [ guard::guard_has_done_callback<decode_run>{} ]
          / action::effect_emit_decode_done{}
      , sml::state<state_decode_done> <= sml::state<state_decode_success_callback_decision>
          + sml::completion<decode_run> [ guard::guard_no_done_callback<decode_run>{} ]
      , sml::state<state_decode_errored> <= sml::state<state_decode_error_callback_decision>
          + sml::completion<decode_run> [ guard::guard_has_error_callback<decode_run>{} ]
          / action::effect_emit_decode_error{}
      , sml::state<state_decode_errored> <= sml::state<state_decode_error_callback_decision>
          + sml::completion<decode_run> [ guard::guard_no_error_callback<decode_run>{} ]
      , sml::state<state_ready> <= sml::state<state_decode_done>
          + sml::completion<decode_run>
      , sml::state<state_ready> <= sml::state<state_decode_errored>
          + sml::completion<decode_run>

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<state_ready> <= sml::state<state_ready> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_encode_runtime_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_encode_shape_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_encode_variant_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_encode_running>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_encode_success_error_out_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_encode_success_callback_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_encode_error_error_out_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_encode_error_callback_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_encode_done>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_encode_errored>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_decode_runtime_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_decode_shape_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_decode_codes_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_decode_variant_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_decode_running>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_decode_success_error_out_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_decode_success_callback_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_decode_error_error_out_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_decode_error_callback_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_decode_done>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected{}
      , sml::state<state_ready> <= sml::state<state_decode_errored>
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

  bool process_event(const event::encode &ev) {
    event::encode_ctx ctx{};
    event::encode_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail::to_error(error::none);
  }

  bool process_event(const event::decode &ev) {
    event::decode_ctx ctx{};
    event::decode_run runtime_ev{ev, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail::to_error(error::none);
  }
};

} // namespace emel::speech::codec::mimi::quantizer
