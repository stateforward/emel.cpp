#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string_view>

#include "emel/kernel/detail.hpp"
#include "emel/speech/encoder/whisper/detail.hpp"
#include "emel/speech/encoder/whisper/context.hpp"
#include "emel/speech/encoder/whisper/events.hpp"

namespace emel::speech::encoder::whisper::guard {

namespace kdetail = emel::speech::encoder::whisper::detail;

inline bool has_tensor(const emel::model::data &model,
                       const std::string_view name, const int32_t n_dims,
                       const std::array<int64_t, 4> &dims,
                       const uint8_t dtype) noexcept {
  const auto *tensor = kdetail::find_tensor(model, name);
  return tensor != nullptr &&
         kdetail::tensor_has_shape(*tensor, n_dims, dims) &&
         static_cast<uint8_t>(tensor->type) == dtype;
}

inline bool has_q8_vector(const emel::model::data &model,
                          const std::string_view name,
                          const int64_t length) noexcept {
  return has_tensor(model, name, 1, {length, 0, 0, 0},
                    ::emel::kernel::detail::dtype_q8_0);
}

inline bool has_aux_vector(const emel::model::data &model,
                           const std::string_view name, const int64_t length,
                           const uint8_t aux_dtype) noexcept {
  return has_tensor(model, name, 1, {length, 0, 0, 0}, aux_dtype);
}

inline bool has_aux_position_matrix(const emel::model::data &model,
                                    const std::string_view name,
                                    const uint8_t aux_dtype) noexcept {
  return has_tensor(model, name, 2, {384, 1500, 0, 0}, aux_dtype);
}

inline bool has_any_aux_vector(const emel::model::data &model,
                               const std::string_view name,
                               const int64_t length) noexcept {
  return has_aux_vector(model, name, length,
                        ::emel::kernel::detail::dtype_q8_0) ||
         has_aux_vector(model, name, length, ::emel::kernel::detail::dtype_f32);
}

inline bool has_any_aux_position_matrix(const emel::model::data &model,
                                        const std::string_view name) noexcept {
  return has_aux_position_matrix(model, name,
                                 ::emel::kernel::detail::dtype_q8_0) ||
         has_aux_position_matrix(model, name,
                                 ::emel::kernel::detail::dtype_f32);
}

inline bool has_encoder_block(const emel::model::data &model,
                              const int32_t block, const uint8_t linear_dtype,
                              const uint8_t aux_dtype) noexcept {
  char name[96] = {};
  const auto has = [&](const char *suffix, const int32_t dims,
                       const std::array<int64_t, 4> &shape,
                       const uint8_t dtype) noexcept {
    std::snprintf(name, sizeof(name), "model.encoder.layers.%d.%s", block,
                  suffix);
    return has_tensor(model, std::string_view{name}, dims, shape, dtype);
  };
  return has("self_attn.k_proj.weight", 2, {384, 384, 0, 0}, linear_dtype) &&
         has("self_attn.v_proj.weight", 2, {384, 384, 0, 0}, linear_dtype) &&
         has("self_attn.v_proj.bias", 1, {384, 0, 0, 0}, aux_dtype) &&
         has("self_attn.q_proj.weight", 2, {384, 384, 0, 0}, linear_dtype) &&
         has("self_attn.q_proj.bias", 1, {384, 0, 0, 0}, aux_dtype) &&
         has("self_attn.out_proj.weight", 2, {384, 384, 0, 0}, linear_dtype) &&
         has("self_attn.out_proj.bias", 1, {384, 0, 0, 0}, aux_dtype) &&
         has("self_attn_layer_norm.weight", 1, {384, 0, 0, 0}, aux_dtype) &&
         has("self_attn_layer_norm.bias", 1, {384, 0, 0, 0}, aux_dtype) &&
         has("fc1.weight", 2, {384, 1536, 0, 0}, linear_dtype) &&
         has("fc1.bias", 1, {1536, 0, 0, 0}, aux_dtype) &&
         has("fc2.weight", 2, {1536, 384, 0, 0}, linear_dtype) &&
         has("fc2.bias", 1, {384, 0, 0, 0}, aux_dtype) &&
         has("final_layer_norm.weight", 1, {384, 0, 0, 0}, aux_dtype) &&
         has("final_layer_norm.bias", 1, {384, 0, 0, 0}, aux_dtype);
}

inline bool has_encoder_linear_variant(const emel::model::data &model,
                                       const uint8_t linear_dtype,
                                       const uint8_t aux_dtype) noexcept {
  return has_encoder_block(model, 0, linear_dtype, aux_dtype) &&
         has_encoder_block(model, 1, linear_dtype, aux_dtype) &&
         has_encoder_block(model, 2, linear_dtype, aux_dtype) &&
         has_encoder_block(model, 3, linear_dtype, aux_dtype);
}

struct guard_model_contract_valid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &contract = runtime_ev.request.contract;
    const auto *model = contract.model;
    return model != nullptr && contract.sample_rate == kdetail::k_sample_rate &&
           contract.mel_bin_count == kdetail::k_mel_bin_count &&
           contract.embedding_length == kdetail::k_embedding_length &&
           contract.feed_forward_length == kdetail::k_feed_forward_length &&
           contract.attention_head_count == kdetail::k_attention_head_count &&
           contract.encoder_block_count == kdetail::k_encoder_block_count &&
           has_tensor(*model, "mel_filters", 2, {201, 80, 0, 0},
                      ::emel::kernel::detail::dtype_f32) &&
           has_tensor(*model, "model.encoder.conv1.weight", 3, {3, 80, 384, 0},
                      ::emel::kernel::detail::dtype_f16) &&
           has_any_aux_vector(*model, "model.encoder.conv1.bias", 384) &&
           has_tensor(*model, "model.encoder.conv2.weight", 3, {3, 384, 384, 0},
                      ::emel::kernel::detail::dtype_f16) &&
           has_any_aux_vector(*model, "model.encoder.conv2.bias", 384) &&
           has_any_aux_position_matrix(
               *model, "model.encoder.embed_positions.weight") &&
           has_any_aux_vector(*model, "model.encoder.layer_norm.weight", 384) &&
           has_any_aux_vector(*model, "model.encoder.layer_norm.bias", 384);
  }
};

struct guard_model_contract_invalid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_model_contract_valid{}(runtime_ev, ctx);
  }
};

struct guard_sample_rate_valid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.sample_rate == kdetail::k_sample_rate;
  }
};

struct guard_sample_rate_invalid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_sample_rate_valid{}(runtime_ev, ctx);
  }
};

struct guard_channel_count_valid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.channel_count == kdetail::k_channel_count;
  }
};

struct guard_channel_count_invalid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_channel_count_valid{}(runtime_ev, ctx);
  }
};

struct guard_pcm_shape_valid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    bool finite = true;
    for (const float value : runtime_ev.request.pcm) {
      finite = finite && std::isfinite(value);
    }
    return runtime_ev.request.pcm.data() != nullptr &&
           !runtime_ev.request.pcm.empty() &&
           runtime_ev.request.pcm.size() <=
               static_cast<size_t>(kdetail::k_max_mel_frame_count *
                                   kdetail::k_hop_length) &&
           finite;
  }
};

struct guard_pcm_shape_invalid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_pcm_shape_valid{}(runtime_ev, ctx);
  }
};

struct guard_output_capacity_valid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.encoder_state.data() != nullptr &&
           runtime_ev.request.encoder_state.size() >=
               static_cast<size_t>(kdetail::required_encoder_output_floats(
                   runtime_ev.request.pcm.size()));
  }
};

struct guard_output_capacity_invalid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_output_capacity_valid{}(runtime_ev, ctx);
  }
};

struct guard_workspace_capacity_valid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.workspace.data() != nullptr &&
           runtime_ev.request.workspace.size() >=
               static_cast<size_t>(kdetail::required_workspace_floats(
                   runtime_ev.request.pcm.size()));
  }
};

struct guard_workspace_capacity_invalid {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_workspace_capacity_valid{}(runtime_ev, ctx);
  }
};

struct guard_q8_0_variant {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return has_encoder_linear_variant(*runtime_ev.request.contract.model,
                                      ::emel::kernel::detail::dtype_q8_0,
                                      ::emel::kernel::detail::dtype_q8_0);
  }
};

struct guard_q8_0_f32_aux_variant {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return has_encoder_linear_variant(*runtime_ev.request.contract.model,
                                      ::emel::kernel::detail::dtype_q8_0,
                                      ::emel::kernel::detail::dtype_f32);
  }
};

struct guard_q4_0_variant {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return has_encoder_linear_variant(*runtime_ev.request.contract.model,
                                      ::emel::kernel::detail::dtype_q4_0,
                                      ::emel::kernel::detail::dtype_q8_0);
  }
};

struct guard_q4_1_variant {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &) const noexcept {
    return has_encoder_linear_variant(*runtime_ev.request.contract.model,
                                      ::emel::kernel::detail::dtype_q4_1,
                                      ::emel::kernel::detail::dtype_q8_0);
  }
};

struct guard_unsupported_variant {
  bool operator()(const event::encode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_q8_0_variant{}(runtime_ev, ctx) &&
           !guard_q8_0_f32_aux_variant{}(runtime_ev, ctx) &&
           !guard_q4_0_variant{}(runtime_ev, ctx) &&
           !guard_q4_1_variant{}(runtime_ev, ctx);
  }
};

struct guard_has_done_callback {
  bool operator()(const event::encode_run &runtime_ev) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_done);
  }
};

struct guard_no_done_callback {
  bool operator()(const event::encode_run &runtime_ev) const noexcept {
    return !guard_has_done_callback{}(runtime_ev);
  }
};

struct guard_has_error_out {
  bool operator()(const event::encode_run &runtime_ev) const noexcept {
    return runtime_ev.request.error_out != nullptr;
  }
};

struct guard_no_error_out {
  bool operator()(const event::encode_run &runtime_ev) const noexcept {
    return !guard_has_error_out{}(runtime_ev);
  }
};

struct guard_has_error_callback {
  bool operator()(const event::encode_run &runtime_ev) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_error);
  }
};

struct guard_no_error_callback {
  bool operator()(const event::encode_run &runtime_ev) const noexcept {
    return !guard_has_error_callback{}(runtime_ev);
  }
};

} // namespace emel::speech::encoder::whisper::guard
