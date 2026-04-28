#pragma once

#include "emel/speech/decoder/whisper/context.hpp"
#include "emel/speech/decoder/whisper/detail.hpp"
#include "emel/speech/decoder/whisper/events.hpp"

namespace emel::speech::decoder::whisper::action {

namespace kdetail = emel::speech::decoder::whisper::detail;

struct effect_begin_decode {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::none);
    runtime_ev.request.token_out = 0;
    runtime_ev.request.confidence_out = 0.0f;
    runtime_ev.request.digest_out = 0u;
    runtime_ev.request.generated_token_count_out = 0;
  }
};

struct effect_mark_model_invalid {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::model_invalid);
  }
};

struct effect_mark_encoder_state_invalid {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::encoder_state);
  }
};

struct effect_mark_decode_policy_invalid {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::decode_policy);
  }
};

struct effect_mark_generated_token_capacity_invalid {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::generated_token_capacity);
  }
};

struct effect_mark_logits_capacity_invalid {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::logits_capacity);
  }
};

struct effect_mark_workspace_capacity_invalid {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::workspace_capacity);
  }
};

struct effect_mark_unsupported_variant {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::unsupported_variant);
  }
};

template <kdetail::linear_weight_variant Variant,
          kdetail::aux_weight_variant Aux = kdetail::aux_weight_variant::q8_0>
struct effect_run_decoder_variant {
  void operator()(const event::decode_run &runtime_ev,
                  context &ctx) const noexcept {
    uint64_t generated_token_count = 0u;
    const auto &tokens = runtime_ev.request.policy.tokens;
    const kdetail::decode_policy_runtime policy{
        .eot = tokens.eot,
        .sot = tokens.sot,
        .translate = tokens.translate,
        .transcribe = tokens.transcribe,
        .no_speech = tokens.no_speech,
        .notimestamps = tokens.notimestamps,
        .timestamp_begin = tokens.timestamp_begin,
        .space = tokens.space,
    };
    const uint64_t digest = kdetail::run_decoder_sequence<Variant, Aux>(
        *runtime_ev.request.contract.model,
        runtime_ev.request.encoder_state.data(),
        static_cast<uint64_t>(runtime_ev.request.encoder_frame_count),
        policy, runtime_ev.request.policy.prompt_tokens.data(),
        static_cast<uint64_t>(runtime_ev.request.policy.prompt_tokens.size()),
        runtime_ev.request.workspace.data(), runtime_ev.request.logits.data(),
        runtime_ev.request.generated_tokens.data(),
        static_cast<uint64_t>(runtime_ev.request.generated_tokens.size()),
        generated_token_count, runtime_ev.request.token_out,
        runtime_ev.request.confidence_out);
    runtime_ev.request.generated_token_count_out =
        static_cast<int32_t>(generated_token_count);
    runtime_ev.request.digest_out = digest;
    runtime_ev.ctx.err = detail::to_error(error::none);
    if constexpr (Variant == kdetail::linear_weight_variant::q8_0) {
      ++ctx.q8_0_dispatch_count;
    } else if constexpr (Variant == kdetail::linear_weight_variant::q4_0) {
      ++ctx.q4_0_dispatch_count;
    } else if constexpr (Variant == kdetail::linear_weight_variant::q4_1) {
      ++ctx.q4_1_dispatch_count;
    }
  }
};

struct effect_store_success_error {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_store_error_error {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_emit_done {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_done(events::decode_done{
        .request = &runtime_ev.request,
        .token = runtime_ev.request.token_out,
        .confidence = runtime_ev.request.confidence_out,
        .digest = runtime_ev.request.digest_out,
    });
  }
};

struct effect_emit_error {
  void operator()(const event::decode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_error(events::decode_error{
        .request = &runtime_ev.request,
        .err = runtime_ev.ctx.err,
    });
  }
};

struct effect_on_unexpected {
  template <class unexpected_event_type>
  void operator()(const unexpected_event_type &, context &) const noexcept {}
};

using effect_run_decoder_q8_0_t =
    effect_run_decoder_variant<kdetail::linear_weight_variant::q8_0>;
using effect_run_decoder_q8_0_f32_aux_t =
    effect_run_decoder_variant<kdetail::linear_weight_variant::q8_0,
                               kdetail::aux_weight_variant::f32>;
using effect_run_decoder_q4_0_t =
    effect_run_decoder_variant<kdetail::linear_weight_variant::q4_0>;
using effect_run_decoder_q4_1_t =
    effect_run_decoder_variant<kdetail::linear_weight_variant::q4_1>;

inline constexpr effect_begin_decode effect_begin_decode{};
inline constexpr effect_mark_model_invalid effect_mark_model_invalid{};
inline constexpr effect_mark_encoder_state_invalid
    effect_mark_encoder_state_invalid{};
inline constexpr effect_mark_decode_policy_invalid
    effect_mark_decode_policy_invalid{};
inline constexpr effect_mark_generated_token_capacity_invalid
    effect_mark_generated_token_capacity_invalid{};
inline constexpr effect_mark_logits_capacity_invalid
    effect_mark_logits_capacity_invalid{};
inline constexpr effect_mark_workspace_capacity_invalid
    effect_mark_workspace_capacity_invalid{};
inline constexpr effect_mark_unsupported_variant
    effect_mark_unsupported_variant{};
inline constexpr effect_run_decoder_q8_0_t effect_run_decoder_q8_0{};
inline constexpr effect_run_decoder_q8_0_f32_aux_t
    effect_run_decoder_q8_0_f32_aux{};
inline constexpr effect_run_decoder_q4_0_t effect_run_decoder_q4_0{};
inline constexpr effect_run_decoder_q4_1_t effect_run_decoder_q4_1{};
inline constexpr effect_store_success_error effect_store_success_error{};
inline constexpr effect_store_error_error effect_store_error_error{};
inline constexpr effect_emit_done effect_emit_done{};
inline constexpr effect_emit_error effect_emit_error{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

} // namespace emel::speech::decoder::whisper::action
