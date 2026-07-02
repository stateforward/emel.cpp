#pragma once

#include <algorithm>

#include "emel/speech/encoder/whisper/context.hpp"
#include "emel/speech/encoder/whisper/detail.hpp"
#include "emel/speech/encoder/whisper/events.hpp"

namespace emel::speech::encoder::whisper::action {

namespace kdetail = emel::speech::encoder::whisper::detail;

struct effect_begin_encode {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::none);
    runtime_ev.request.frame_count_out = 0;
    runtime_ev.request.width_out = 0;
    runtime_ev.request.digest_out = 0u;
  }
};

struct effect_mark_model_invalid {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::model_invalid);
  }
};

struct effect_mark_sample_rate_invalid {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::sample_rate);
  }
};

struct effect_mark_channel_count_invalid {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::channel_count);
  }
};

struct effect_mark_pcm_shape_invalid {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::pcm_shape);
  }
};

struct effect_mark_output_capacity_invalid {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::output_capacity);
  }
};

struct effect_mark_workspace_capacity_invalid {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::workspace_capacity);
  }
};

struct effect_mark_unsupported_variant {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::unsupported_variant);
  }
};

template <kdetail::linear_weight_variant Variant,
          kdetail::aux_weight_variant Aux = kdetail::aux_weight_variant::q8_0>
struct effect_run_encoder_variant {
  void operator()(const event::encode_run &runtime_ev,
                  context &ctx) const noexcept {
    uint64_t frame_count = 0u;
    const uint64_t digest = kdetail::run_encoder<Variant, Aux>(
        *runtime_ev.request.contract.model, runtime_ev.request.pcm.data(),
        static_cast<uint64_t>(runtime_ev.request.pcm.size()),
        runtime_ev.request.workspace.data(),
        runtime_ev.request.encoder_state.data(), frame_count);
    runtime_ev.request.frame_count_out = static_cast<int32_t>(frame_count);
    runtime_ev.request.width_out = kdetail::k_embedding_length;
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
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_store_error_error {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_emit_done {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_done(events::encode_done{
        .request = &runtime_ev.request,
        .frame_count = runtime_ev.request.frame_count_out,
        .width = runtime_ev.request.width_out,
        .digest = runtime_ev.request.digest_out,
    });
  }
};

struct effect_emit_error {
  void operator()(const event::encode_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_error(events::encode_error{
        .request = &runtime_ev.request,
        .err = runtime_ev.ctx.err,
    });
  }
};

struct effect_on_unexpected {
  template <class unexpected_event_type>
  void operator()(const unexpected_event_type &, context &) const noexcept {}
};

using effect_run_encoder_q8_0_t =
    effect_run_encoder_variant<kdetail::linear_weight_variant::q8_0>;
using effect_run_encoder_q8_0_f32_aux_t =
    effect_run_encoder_variant<kdetail::linear_weight_variant::q8_0,
                               kdetail::aux_weight_variant::f32>;
using effect_run_encoder_q4_0_t =
    effect_run_encoder_variant<kdetail::linear_weight_variant::q4_0>;
using effect_run_encoder_q4_1_t =
    effect_run_encoder_variant<kdetail::linear_weight_variant::q4_1>;

inline constexpr effect_begin_encode effect_begin_encode{};
inline constexpr effect_mark_model_invalid effect_mark_model_invalid{};
inline constexpr effect_mark_sample_rate_invalid
    effect_mark_sample_rate_invalid{};
inline constexpr effect_mark_channel_count_invalid
    effect_mark_channel_count_invalid{};
inline constexpr effect_mark_pcm_shape_invalid effect_mark_pcm_shape_invalid{};
inline constexpr effect_mark_output_capacity_invalid
    effect_mark_output_capacity_invalid{};
inline constexpr effect_mark_workspace_capacity_invalid
    effect_mark_workspace_capacity_invalid{};
inline constexpr effect_mark_unsupported_variant
    effect_mark_unsupported_variant{};
inline constexpr effect_run_encoder_q8_0_t effect_run_encoder_q8_0{};
inline constexpr effect_run_encoder_q8_0_f32_aux_t
    effect_run_encoder_q8_0_f32_aux{};
inline constexpr effect_run_encoder_q4_0_t effect_run_encoder_q4_0{};
inline constexpr effect_run_encoder_q4_1_t effect_run_encoder_q4_1{};
inline constexpr effect_store_success_error effect_store_success_error{};
inline constexpr effect_store_error_error effect_store_error_error{};
inline constexpr effect_emit_done effect_emit_done{};
inline constexpr effect_emit_error effect_emit_error{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

} // namespace emel::speech::encoder::whisper::action
