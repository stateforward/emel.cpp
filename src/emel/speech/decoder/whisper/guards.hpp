#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>

#include "emel/kernel/detail.hpp"
#include "emel/speech/decoder/whisper/context.hpp"
#include "emel/speech/decoder/whisper/detail.hpp"
#include "emel/speech/decoder/whisper/events.hpp"
#include "emel/speech/tokenizer/whisper/any.hpp"

namespace emel::speech::decoder::whisper::guard {

namespace kdetail = emel::speech::decoder::whisper::detail;

inline bool has_tensor(const emel::model::data &model,
                       const std::string_view name, const int32_t n_dims,
                       const std::array<int64_t, 4> &dims,
                       const uint8_t dtype) noexcept {
  const auto *tensor = kdetail::find_tensor(model, name);
  return tensor != nullptr &&
         kdetail::tensor_has_shape(*tensor, n_dims, dims) &&
         static_cast<uint8_t>(tensor->type) == dtype;
}

inline bool has_aux_vector(const emel::model::data &model,
                           const std::string_view name, const int64_t length,
                           const uint8_t aux_dtype) noexcept {
  return has_tensor(model, name, 1, {length, 0, 0, 0}, aux_dtype);
}

inline bool has_aux_position_matrix(const emel::model::data &model,
                                    const std::string_view name,
                                    const uint8_t aux_dtype) noexcept {
  return has_tensor(model, name, 2, {384, 448, 0, 0}, aux_dtype);
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

inline bool has_decoder_block(const emel::model::data &model,
                              const int32_t block, const uint8_t linear_dtype,
                              const uint8_t aux_dtype) noexcept {
  char name[96] = {};
  const auto has = [&](const char *suffix, const int32_t dims,
                       const std::array<int64_t, 4> &shape,
                       const uint8_t dtype) noexcept {
    std::snprintf(name, sizeof(name), "model.decoder.layers.%d.%s", block,
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
         has("encoder_attn.k_proj.weight", 2, {384, 384, 0, 0}, linear_dtype) &&
         has("encoder_attn.v_proj.weight", 2, {384, 384, 0, 0}, linear_dtype) &&
         has("encoder_attn.v_proj.bias", 1, {384, 0, 0, 0}, aux_dtype) &&
         has("encoder_attn.q_proj.weight", 2, {384, 384, 0, 0}, linear_dtype) &&
         has("encoder_attn.q_proj.bias", 1, {384, 0, 0, 0}, aux_dtype) &&
         has("encoder_attn.out_proj.weight", 2, {384, 384, 0, 0},
             linear_dtype) &&
         has("encoder_attn.out_proj.bias", 1, {384, 0, 0, 0}, aux_dtype) &&
         has("encoder_attn_layer_norm.weight", 1, {384, 0, 0, 0}, aux_dtype) &&
         has("encoder_attn_layer_norm.bias", 1, {384, 0, 0, 0}, aux_dtype) &&
         has("fc1.weight", 2, {384, 1536, 0, 0}, linear_dtype) &&
         has("fc1.bias", 1, {1536, 0, 0, 0}, aux_dtype) &&
         has("fc2.weight", 2, {1536, 384, 0, 0}, linear_dtype) &&
         has("fc2.bias", 1, {384, 0, 0, 0}, aux_dtype) &&
         has("final_layer_norm.weight", 1, {384, 0, 0, 0}, aux_dtype) &&
         has("final_layer_norm.bias", 1, {384, 0, 0, 0}, aux_dtype);
}

inline bool has_decoder_linear_variant(const emel::model::data &model,
                                       const uint8_t linear_dtype,
                                       const uint8_t aux_dtype) noexcept {
  return has_decoder_block(model, 0, linear_dtype, aux_dtype) &&
         has_decoder_block(model, 1, linear_dtype, aux_dtype) &&
         has_decoder_block(model, 2, linear_dtype, aux_dtype) &&
         has_decoder_block(model, 3, linear_dtype, aux_dtype);
}

struct guard_model_contract_valid {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &contract = runtime_ev.request.contract;
    const auto *model = contract.model;
    return model != nullptr && contract.vocab_size == detail::k_vocab_size &&
           contract.embedding_length == detail::k_embedding_length &&
           contract.decoder_block_count == detail::k_decoder_block_count &&
           has_tensor(*model, "model.decoder.embed_tokens.weight", 2,
                      {384, 51865, 0, 0}, ::emel::kernel::detail::dtype_q8_0) &&
           has_any_aux_position_matrix(
               *model, "model.decoder.embed_positions.weight") &&
           has_any_aux_vector(*model, "model.decoder.layer_norm.weight", 384) &&
           has_any_aux_vector(*model, "model.decoder.layer_norm.bias", 384);
  }
};

struct guard_model_contract_invalid {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_model_contract_valid{}(runtime_ev, ctx);
  }
};

struct guard_encoder_state_valid {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.encoder_state.data() != nullptr &&
           runtime_ev.request.encoder_frame_count > 0 &&
           runtime_ev.request.encoder_frame_count <=
               detail::k_max_encoder_frame_count &&
           runtime_ev.request.encoder_state.size() >=
               static_cast<size_t>(runtime_ev.request.encoder_frame_count) *
                   static_cast<size_t>(detail::k_embedding_length);
  }
};

struct guard_encoder_state_invalid {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_encoder_state_valid{}(runtime_ev, ctx);
  }
};

struct guard_decode_policy_supported {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &) const noexcept {
    return emel::speech::tokenizer::whisper::
        is_tiny_asr_decode_policy_supported(runtime_ev.request.policy);
  }
};

struct guard_decode_policy_unsupported {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_decode_policy_supported{}(runtime_ev, ctx);
  }
};

struct guard_generated_token_capacity_valid {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.generated_tokens.data() != nullptr &&
           !runtime_ev.request.generated_tokens.empty();
  }
};

struct guard_generated_token_capacity_invalid {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_generated_token_capacity_valid{}(runtime_ev, ctx);
  }
};

struct guard_logits_capacity_valid {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.logits.data() != nullptr &&
           runtime_ev.request.logits.size() >=
               static_cast<size_t>(detail::k_vocab_size);
  }
};

struct guard_logits_capacity_invalid {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_logits_capacity_valid{}(runtime_ev, ctx);
  }
};

struct guard_workspace_capacity_valid {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.workspace.data() != nullptr &&
           runtime_ev.request.workspace.size() >=
               static_cast<size_t>(detail::required_decoder_workspace_floats(
                   static_cast<uint64_t>(
                       runtime_ev.request.encoder_frame_count)));
  }
};

struct guard_workspace_capacity_invalid {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_workspace_capacity_valid{}(runtime_ev, ctx);
  }
};

struct guard_q8_0_variant {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &) const noexcept {
    return has_decoder_linear_variant(*runtime_ev.request.contract.model,
                                      ::emel::kernel::detail::dtype_q8_0,
                                      ::emel::kernel::detail::dtype_q8_0);
  }
};

struct guard_q8_0_f32_aux_variant {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &) const noexcept {
    return has_decoder_linear_variant(*runtime_ev.request.contract.model,
                                      ::emel::kernel::detail::dtype_q8_0,
                                      ::emel::kernel::detail::dtype_f32);
  }
};

struct guard_q4_0_variant {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &) const noexcept {
    return has_decoder_linear_variant(*runtime_ev.request.contract.model,
                                      ::emel::kernel::detail::dtype_q4_0,
                                      ::emel::kernel::detail::dtype_q8_0);
  }
};

struct guard_q4_1_variant {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &) const noexcept {
    return has_decoder_linear_variant(*runtime_ev.request.contract.model,
                                      ::emel::kernel::detail::dtype_q4_1,
                                      ::emel::kernel::detail::dtype_q8_0);
  }
};

struct guard_unsupported_variant {
  bool operator()(const event::decode_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_q8_0_variant{}(runtime_ev, ctx) &&
           !guard_q8_0_f32_aux_variant{}(runtime_ev, ctx) &&
           !guard_q4_0_variant{}(runtime_ev, ctx) &&
           !guard_q4_1_variant{}(runtime_ev, ctx);
  }
};

struct guard_has_done_callback {
  bool operator()(const event::decode_run &runtime_ev) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_done);
  }
};

struct guard_no_done_callback {
  bool operator()(const event::decode_run &runtime_ev) const noexcept {
    return !guard_has_done_callback{}(runtime_ev);
  }
};

struct guard_has_error_out {
  bool operator()(const event::decode_run &runtime_ev) const noexcept {
    return runtime_ev.request.error_out != nullptr;
  }
};

struct guard_no_error_out {
  bool operator()(const event::decode_run &runtime_ev) const noexcept {
    return !guard_has_error_out{}(runtime_ev);
  }
};

struct guard_has_error_callback {
  bool operator()(const event::decode_run &runtime_ev) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_error);
  }
};

struct guard_no_error_callback {
  bool operator()(const event::decode_run &runtime_ev) const noexcept {
    return !guard_has_error_callback{}(runtime_ev);
  }
};

} // namespace emel::speech::decoder::whisper::guard
