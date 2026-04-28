#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "emel/kernel/aarch64/actions.hpp"
#include "emel/model/data.hpp"
#include "emel/speech/decoder/whisper/errors.hpp"

namespace emel::speech::decoder::whisper::detail {

inline emel::error::type to_error(const error value) noexcept {
  return emel::error::cast(value);
}

inline constexpr int32_t k_embedding_length = 384;
inline constexpr int32_t k_feed_forward_length = 1536;
inline constexpr int32_t k_attention_head_count = 6;
inline constexpr int32_t k_attention_head_dim = 64;
inline constexpr int32_t k_decoder_block_count = 4;
inline constexpr int32_t k_vocab_size = 51865;
inline constexpr int32_t k_max_encoder_frame_count = 1500;
inline constexpr int32_t k_decoder_prompt_token_count = 4;
inline constexpr int32_t k_decoder_context_token_count = 448;
inline constexpr int32_t k_decoder_sequence_token_count =
    k_decoder_context_token_count;
inline constexpr int32_t k_max_generated_token_count =
    k_decoder_sequence_token_count - k_decoder_prompt_token_count;
inline constexpr int32_t k_token_eot = 50257;
inline constexpr int32_t k_token_sot = 50258;
inline constexpr int32_t k_token_translate = 50358;
inline constexpr int32_t k_token_transcribe = 50359;
inline constexpr int32_t k_token_no_speech = 50362;
inline constexpr int32_t k_token_no_timestamps = 50363;
inline constexpr int32_t k_token_timestamp_begin = 50364;
inline constexpr int32_t k_token_space = 220;
inline constexpr float k_layer_norm_epsilon = 1.0e-5f;

struct execution_contract {
  const emel::model::data *model = nullptr;
  int32_t vocab_size = 0;
  int32_t embedding_length = 0;
  int32_t decoder_block_count = 0;
};

inline execution_contract
bind_execution_contract(const emel::model::data &model) noexcept {
  return execution_contract{
      .model = &model,
      .vocab_size = k_vocab_size,
      .embedding_length = k_embedding_length,
      .decoder_block_count = k_decoder_block_count,
  };
}

enum class linear_weight_variant : uint8_t {
  q8_0,
  q4_0,
  q4_1,
};

enum class aux_weight_variant : uint8_t {
  q8_0,
  f32,
};

struct decode_policy_runtime {
  int32_t eot = k_token_eot;
  int32_t sot = k_token_sot;
  int32_t translate = k_token_translate;
  int32_t transcribe = k_token_transcribe;
  int32_t no_speech = k_token_no_speech;
  int32_t notimestamps = k_token_no_timestamps;
  int32_t timestamp_begin = k_token_timestamp_begin;
  int32_t space = k_token_space;
};

inline uint64_t
required_decoder_workspace_floats(const uint64_t encoder_frames) noexcept {
  const uint64_t tokens = static_cast<uint64_t>(k_decoder_sequence_token_count);
  return (static_cast<uint64_t>(k_embedding_length) * encoder_frames * 2u *
          static_cast<uint64_t>(k_decoder_block_count)) +
         (static_cast<uint64_t>(k_embedding_length) * tokens * 6u) +
         static_cast<uint64_t>(k_feed_forward_length) +
         std::max<uint64_t>(encoder_frames, tokens);
}

inline const emel::model::data::tensor_record *
find_tensor(const emel::model::data &model,
            const std::string_view name) noexcept {
  for (uint32_t index = 0u; index < model.n_tensors; ++index) {
    const auto &tensor = model.tensors[index];
    if (emel::model::tensor_name_view(model, tensor) == name) {
      return &tensor;
    }
  }
  return nullptr;
}

inline bool tensor_has_shape(const emel::model::data::tensor_record &tensor,
                             const int32_t n_dims,
                             const std::array<int64_t, 4> &dims) noexcept {
  bool ok = tensor.data != nullptr && tensor.data_size != 0u &&
            tensor.n_dims == n_dims;
  for (int32_t dim = 0; dim < n_dims; ++dim) {
    ok = ok && tensor.dims[static_cast<size_t>(dim)] ==
                   dims[static_cast<size_t>(dim)];
  }
  return ok;
}

inline float read_f32_value(const void *data, const uint64_t index) noexcept {
  float out = 0.0f;
  std::memcpy(&out, static_cast<const float *>(data) + index, sizeof(out));
  return out;
}

inline float read_q8_0_value(const emel::model::data::tensor_record &tensor,
                             const uint64_t index) noexcept {
  const uint64_t cols = static_cast<uint64_t>(tensor.dims[0]);
  const uint64_t row = index / cols;
  const uint64_t col = index - row * cols;
  const uint64_t row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q8_0, cols);
  const auto *row_base =
      static_cast<const uint8_t *>(tensor.data) + row * row_bytes;
  const auto *blocks =
      reinterpret_cast<const ::emel::kernel::detail::quant::block_q8_0 *>(
          row_base);
  const uint64_t block = col / ::emel::kernel::detail::quant::QK8_0;
  const uint64_t lane = col - block * ::emel::kernel::detail::quant::QK8_0;
  const float d = ::emel::kernel::detail::quant::fp16_to_fp32(blocks[block].d);
  return d * static_cast<float>(blocks[block].qs[lane]);
}

inline float read_vector_q8_0(const emel::model::data::tensor_record &tensor,
                              const uint64_t index) noexcept {
  const auto *blocks =
      static_cast<const ::emel::kernel::detail::quant::block_q8_0 *>(
          tensor.data);
  const uint64_t block = index / ::emel::kernel::detail::quant::QK8_0;
  const uint64_t lane = index - block * ::emel::kernel::detail::quant::QK8_0;
  const float d = ::emel::kernel::detail::quant::fp16_to_fp32(blocks[block].d);
  return d * static_cast<float>(blocks[block].qs[lane]);
}

template <aux_weight_variant Aux>
inline float read_aux_vector(const emel::model::data::tensor_record &tensor,
                             const uint64_t index) noexcept {
  if constexpr (Aux == aux_weight_variant::f32) {
    return read_f32_value(tensor.data, index);
  } else {
    return read_vector_q8_0(tensor, index);
  }
}

template <aux_weight_variant Aux>
inline float read_aux_matrix(const emel::model::data::tensor_record &tensor,
                             const uint64_t index) noexcept {
  if constexpr (Aux == aux_weight_variant::f32) {
    return read_f32_value(tensor.data, index);
  } else {
    return read_q8_0_value(tensor, index);
  }
}

inline float
read_matrix_q4_0_value(const emel::model::data::tensor_record &tensor,
                       const uint64_t row, const uint64_t col) noexcept {
  const uint64_t cols = static_cast<uint64_t>(tensor.dims[0]);
  const uint64_t row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q4_0, cols);
  const auto *row_base =
      static_cast<const uint8_t *>(tensor.data) + row * row_bytes;
  const auto *blocks =
      reinterpret_cast<const ::emel::kernel::detail::quant::block_q4_0 *>(
          row_base);
  const uint64_t block = col / ::emel::kernel::detail::quant::QK4_0;
  const uint64_t lane = col - block * ::emel::kernel::detail::quant::QK4_0;
  const uint8_t packed = blocks[block].qs[lane % 16u];
  const uint8_t nibble = lane < 16u ? static_cast<uint8_t>(packed & 0x0fu)
                                    : static_cast<uint8_t>(packed >> 4u);
  const int32_t quant = static_cast<int32_t>(nibble) - 8;
  return ::emel::kernel::detail::quant::fp16_to_fp32(blocks[block].d) *
         static_cast<float>(quant);
}

inline float
read_matrix_q4_1_value(const emel::model::data::tensor_record &tensor,
                       const uint64_t row, const uint64_t col) noexcept {
  const uint64_t cols = static_cast<uint64_t>(tensor.dims[0]);
  const uint64_t row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q4_1, cols);
  const auto *row_base =
      static_cast<const uint8_t *>(tensor.data) + row * row_bytes;
  const auto *blocks =
      reinterpret_cast<const ::emel::kernel::detail::quant::block_q4_1 *>(
          row_base);
  const uint64_t block = col / ::emel::kernel::detail::quant::QK4_1;
  const uint64_t lane = col - block * ::emel::kernel::detail::quant::QK4_1;
  const uint8_t packed = blocks[block].qs[lane % 16u];
  const uint8_t nibble = lane < 16u ? static_cast<uint8_t>(packed & 0x0fu)
                                    : static_cast<uint8_t>(packed >> 4u);
  return ::emel::kernel::detail::quant::fp16_to_fp32(blocks[block].d) *
             static_cast<float>(nibble) +
         ::emel::kernel::detail::quant::fp16_to_fp32(blocks[block].m);
}

#if defined(__ARM_NEON) && defined(__aarch64__)
inline float
dot_q8_0_row_neon(const ::emel::kernel::detail::quant::block_q8_0 *blocks,
                  const float *input, const uint64_t block_count) noexcept {
  float32x4_t acc = vdupq_n_f32(0.0f);
  for (uint64_t block = 0; block < block_count; ++block) {
    const float d =
        ::emel::kernel::detail::quant::fp16_to_fp32(blocks[block].d);
    const uint64_t input_base = block * ::emel::kernel::detail::quant::QK8_0;
    for (uint64_t lane = 0; lane < ::emel::kernel::detail::quant::QK8_0;
         lane += 8u) {
      const int8x8_t q8 = vld1_s8(blocks[block].qs.data() + lane);
      const int16x8_t q16 = vmovl_s8(q8);
      const float32x4_t q_lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
      const float32x4_t q_hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
      acc = vmlaq_f32(acc, vld1q_f32(input + input_base + lane),
                      vmulq_n_f32(q_lo, d));
      acc = vmlaq_f32(acc, vld1q_f32(input + input_base + lane + 4u),
                      vmulq_n_f32(q_hi, d));
    }
  }
  return vaddvq_f32(acc);
}
#endif

template <linear_weight_variant Variant>
inline float dot_linear_row(const emel::model::data::tensor_record &weight,
                            const uint64_t row, const float *input,
                            const uint64_t cols) noexcept {
  float acc = 0.0f;
  if constexpr (Variant == linear_weight_variant::q8_0) {
    const uint64_t row_bytes =
        ::emel::kernel::detail::quantized_row_storage_bytes(
            ::emel::kernel::detail::dtype_q8_0, cols);
    const auto *row_base =
        static_cast<const uint8_t *>(weight.data) + row * row_bytes;
    const auto *blocks =
        reinterpret_cast<const ::emel::kernel::detail::quant::block_q8_0 *>(
            row_base);
    const uint64_t block_count = cols / ::emel::kernel::detail::quant::QK8_0;
#if defined(__ARM_NEON) && defined(__aarch64__)
    acc = dot_q8_0_row_neon(blocks, input, block_count);
#else
    for (uint64_t block = 0; block < block_count; ++block) {
      const float d =
          ::emel::kernel::detail::quant::fp16_to_fp32(blocks[block].d);
      for (uint64_t lane = 0; lane < ::emel::kernel::detail::quant::QK8_0;
           ++lane) {
        acc += d * static_cast<float>(blocks[block].qs[lane]) *
               input[block * ::emel::kernel::detail::quant::QK8_0 + lane];
      }
    }
#endif
  } else if constexpr (Variant == linear_weight_variant::q4_0) {
    for (uint64_t col = 0; col < cols; ++col) {
      acc += read_matrix_q4_0_value(weight, row, col) * input[col];
    }
  } else if constexpr (Variant == linear_weight_variant::q4_1) {
    for (uint64_t col = 0; col < cols; ++col) {
      acc += read_matrix_q4_1_value(weight, row, col) * input[col];
    }
  }
  return acc;
}

template <uint64_t In, uint64_t Out, bool HasBias,
          aux_weight_variant Aux = aux_weight_variant::q8_0>
inline void
linear_q8_0_quantized_input(const emel::model::data::tensor_record &weight,
                            const emel::model::data::tensor_record *bias,
                            const float *input, float *output) noexcept {
  const uint64_t row_bytes =
      ::emel::kernel::detail::quantized_row_storage_bytes(
          ::emel::kernel::detail::dtype_q8_0, In);
#if defined(__aarch64__) || defined(__ARM_NEON)
  ::emel::kernel::event::op_mul_mat request{
      .src0 =
          {
              .data = weight.data,
              .type = ::emel::kernel::event::dtype::q8_0,
              .ne = {In, Out, 1u, 1u},
              .nb = {1u, row_bytes, row_bytes * Out, row_bytes * Out},
          },
      .src1 =
          {
              .data = input,
              .type = ::emel::kernel::event::dtype::f32,
              .ne = {1u, In, 1u, 1u},
              .nb =
                  {
                      sizeof(float),
                      sizeof(float),
                      sizeof(float) * In,
                      sizeof(float) * In,
                  },
          },
      .dst =
          {
              .data = output,
              .type = ::emel::kernel::event::dtype::f32,
              .ne = {1u, Out, 1u, 1u},
              .nb =
                  {
                      sizeof(float),
                      sizeof(float),
                      sizeof(float) * Out,
                      sizeof(float) * Out,
                  },
          },
      .nth = 1u,
  };
  ::emel::kernel::aarch64::detail::execute_neon_mul_mat_q8_0_vector_unchecked(
      request);
#else
  constexpr uint64_t block_count = In / ::emel::kernel::detail::quant::QK8_0;
  std::array<::emel::kernel::detail::quant::block_q8_0,
             static_cast<size_t>(block_count)>
      input_blocks = {};
  ::emel::kernel::detail::quant::quantize_row_q8_0_strided(
      input, 1u, input_blocks.data(), static_cast<int64_t>(In));
  for (uint64_t row = 0; row < Out; ++row) {
    const auto *row_base =
        static_cast<const uint8_t *>(weight.data) + row * row_bytes;
    const auto *weight_blocks =
        reinterpret_cast<const ::emel::kernel::detail::quant::block_q8_0 *>(
            row_base);
    output[row] = ::emel::kernel::detail::dot_q8_0_q8_0_row_scalar(
        weight_blocks, input_blocks.data(), block_count);
  }
#endif
  if constexpr (HasBias) {
    for (uint64_t row = 0; row < Out; ++row) {
      output[row] += read_aux_vector<Aux>(*bias, row);
    }
  }
}

inline float gelu(const float value) noexcept {
  return 0.5f * value *
         (1.0f + std::tanh(0.7978845608028654f *
                           (value + 0.044715f * value * value * value)));
}

template <aux_weight_variant Aux>
inline void layer_norm_frame(const float *input,
                             const emel::model::data::tensor_record &weight,
                             const emel::model::data::tensor_record &bias,
                             float *output) noexcept {
  float mean = 0.0f;
  for (uint64_t dim = 0; dim < static_cast<uint64_t>(k_embedding_length);
       ++dim) {
    mean += input[dim];
  }
  mean /= static_cast<float>(k_embedding_length);

  float var = 0.0f;
  for (uint64_t dim = 0; dim < static_cast<uint64_t>(k_embedding_length);
       ++dim) {
    const float diff = input[dim] - mean;
    var += diff * diff;
  }
  const float inv_std =
      1.0f / std::sqrt(var / static_cast<float>(k_embedding_length) +
                       k_layer_norm_epsilon);
  for (uint64_t dim = 0; dim < static_cast<uint64_t>(k_embedding_length);
       ++dim) {
    output[dim] =
        (input[dim] - mean) * inv_std * read_aux_vector<Aux>(weight, dim) +
        read_aux_vector<Aux>(bias, dim);
  }
}

template <linear_weight_variant Variant, uint64_t In, uint64_t Out,
          aux_weight_variant Aux = aux_weight_variant::q8_0>
inline void linear(const emel::model::data::tensor_record &weight,
                   const emel::model::data::tensor_record &bias,
                   const float *input, float *output) noexcept {
  if constexpr (Variant == linear_weight_variant::q8_0) {
    linear_q8_0_quantized_input<In, Out, true, Aux>(weight, &bias, input,
                                                    output);
  } else {
    for (uint64_t row = 0; row < Out; ++row) {
      output[row] = dot_linear_row<Variant>(weight, row, input, In) +
                    read_aux_vector<Aux>(bias, row);
    }
  }
}

template <linear_weight_variant Variant, uint64_t In, uint64_t Out>
inline void linear_no_bias(const emel::model::data::tensor_record &weight,
                           const float *input, float *output) noexcept {
  if constexpr (Variant == linear_weight_variant::q8_0) {
    linear_q8_0_quantized_input<In, Out, false>(weight, nullptr, input, output);
  } else {
    for (uint64_t row = 0; row < Out; ++row) {
      output[row] = dot_linear_row<Variant>(weight, row, input, In);
    }
  }
}

inline void softmax(float *values, const uint64_t count) noexcept {
  float max_value = values[0];
  for (uint64_t index = 1; index < count; ++index) {
    max_value = std::max(max_value, values[index]);
  }
  float sum = 0.0f;
  for (uint64_t index = 0; index < count; ++index) {
    values[index] = std::exp(values[index] - max_value);
    sum += values[index];
  }
  const float inv_sum = 1.0f / sum;
  for (uint64_t index = 0; index < count; ++index) {
    values[index] *= inv_sum;
  }
}

inline uint64_t append_literal(char *output, uint64_t offset,
                               const char *text) noexcept {
  for (uint64_t index = 0; text[index] != '\0'; ++index) {
    output[offset] = text[index];
    ++offset;
  }
  return offset;
}

inline uint64_t write_layer_tensor_name(char *output, const char *prefix,
                                        const uint64_t layer,
                                        const char *suffix) noexcept {
  uint64_t offset = append_literal(output, 0u, prefix);
  output[offset] = static_cast<char>('0' + layer);
  ++offset;
  output[offset] = '.';
  ++offset;
  return append_literal(output, offset, suffix);
}

template <linear_weight_variant Variant, aux_weight_variant Aux>
inline void compute_decoder_cross_cache(const emel::model::data &model,
                                        const float *encoder_state,
                                        const uint64_t encoder_frames,
                                        float *cross_k_cache,
                                        float *cross_v_cache) noexcept {
  char name[96] = {};
  const uint64_t width = static_cast<uint64_t>(k_embedding_length);
  const uint64_t layer_stride = encoder_frames * width;
  for (uint64_t layer = 0; layer < static_cast<uint64_t>(k_decoder_block_count);
       ++layer) {
    const auto layer_tensor = [&](const char *suffix) noexcept {
      const uint64_t name_size =
          write_layer_tensor_name(name, "model.decoder.layers.", layer, suffix);
      return *find_tensor(
          model, std::string_view{name, static_cast<size_t>(name_size)});
    };
    const auto &cross_k_w = layer_tensor("encoder_attn.k_proj.weight");
    const auto &cross_v_w = layer_tensor("encoder_attn.v_proj.weight");
    const auto &cross_v_b = layer_tensor("encoder_attn.v_proj.bias");
    float *layer_cross_k = cross_k_cache + layer * layer_stride;
    float *layer_cross_v = cross_v_cache + layer * layer_stride;
    for (uint64_t frame = 0; frame < encoder_frames; ++frame) {
      const float *frame_in = encoder_state + frame * width;
      linear_no_bias<Variant, k_embedding_length, k_embedding_length>(
          cross_k_w, frame_in, layer_cross_k + frame * width);
      linear<Variant, k_embedding_length, k_embedding_length, Aux>(
          cross_v_w, cross_v_b, frame_in, layer_cross_v + frame * width);
    }
  }
}

template <linear_weight_variant Variant, aux_weight_variant Aux>
inline void
run_decoder_layer_sequence(const emel::model::data &model, const uint64_t layer,
                           const uint64_t encoder_frames,
                           const uint64_t token_count, const float *cross_k,
                           const float *cross_v, float *hidden, float *next,
                           float *q, float *k, float *v, float *attn,
                           float *norm, float *ff, float *scores) noexcept {
  char name[96] = {};
  const auto layer_tensor = [&](const char *suffix) noexcept {
    const uint64_t name_size =
        write_layer_tensor_name(name, "model.decoder.layers.", layer, suffix);
    return *find_tensor(model,
                        std::string_view{name, static_cast<size_t>(name_size)});
  };

  const auto &self_ln_w = layer_tensor("self_attn_layer_norm.weight");
  const auto &self_ln_b = layer_tensor("self_attn_layer_norm.bias");
  const auto &self_q_w = layer_tensor("self_attn.q_proj.weight");
  const auto &self_q_b = layer_tensor("self_attn.q_proj.bias");
  const auto &self_k_w = layer_tensor("self_attn.k_proj.weight");
  const auto &self_v_w = layer_tensor("self_attn.v_proj.weight");
  const auto &self_v_b = layer_tensor("self_attn.v_proj.bias");
  const auto &self_o_w = layer_tensor("self_attn.out_proj.weight");
  const auto &self_o_b = layer_tensor("self_attn.out_proj.bias");
  const auto &cross_ln_w = layer_tensor("encoder_attn_layer_norm.weight");
  const auto &cross_ln_b = layer_tensor("encoder_attn_layer_norm.bias");
  const auto &cross_q_w = layer_tensor("encoder_attn.q_proj.weight");
  const auto &cross_q_b = layer_tensor("encoder_attn.q_proj.bias");
  const auto &cross_o_w = layer_tensor("encoder_attn.out_proj.weight");
  const auto &cross_o_b = layer_tensor("encoder_attn.out_proj.bias");
  const auto &final_ln_w = layer_tensor("final_layer_norm.weight");
  const auto &final_ln_b = layer_tensor("final_layer_norm.bias");
  const auto &fc1_w = layer_tensor("fc1.weight");
  const auto &fc1_b = layer_tensor("fc1.bias");
  const auto &fc2_w = layer_tensor("fc2.weight");
  const auto &fc2_b = layer_tensor("fc2.bias");

  const uint64_t width = static_cast<uint64_t>(k_embedding_length);
  for (uint64_t token = 0; token < token_count; ++token) {
    layer_norm_frame<Aux>(hidden + token * width, self_ln_w, self_ln_b, norm);
    linear<Variant, k_embedding_length, k_embedding_length, Aux>(
        self_q_w, self_q_b, norm, q + token * width);
    linear_no_bias<Variant, k_embedding_length, k_embedding_length>(
        self_k_w, norm, k + token * width);
    linear<Variant, k_embedding_length, k_embedding_length, Aux>(
        self_v_w, self_v_b, norm, v + token * width);
  }

  const float scale =
      1.0f / std::sqrt(static_cast<float>(k_attention_head_dim));
  for (uint64_t token = 0; token < token_count; ++token) {
    std::fill_n(attn + token * width, static_cast<size_t>(width), 0.0f);
    for (uint64_t head = 0;
         head < static_cast<uint64_t>(k_attention_head_count); ++head) {
      for (uint64_t key_token = 0; key_token <= token; ++key_token) {
        float score = 0.0f;
        for (uint64_t dim = 0;
             dim < static_cast<uint64_t>(k_attention_head_dim); ++dim) {
          const uint64_t offset =
              head * static_cast<uint64_t>(k_attention_head_dim) + dim;
          score += q[token * width + offset] * k[key_token * width + offset];
        }
        scores[key_token] = score * scale;
      }
      softmax(scores, token + 1u);
      for (uint64_t key_token = 0; key_token <= token; ++key_token) {
        for (uint64_t dim = 0;
             dim < static_cast<uint64_t>(k_attention_head_dim); ++dim) {
          const uint64_t offset =
              head * static_cast<uint64_t>(k_attention_head_dim) + dim;
          attn[token * width + offset] +=
              scores[key_token] * v[key_token * width + offset];
        }
      }
    }
    linear<Variant, k_embedding_length, k_embedding_length, Aux>(
        self_o_w, self_o_b, attn + token * width, norm);
    for (uint64_t dim = 0; dim < width; ++dim) {
      next[token * width + dim] = hidden[token * width + dim] + norm[dim];
    }
  }

  for (uint64_t token = 0; token < token_count; ++token) {
    layer_norm_frame<Aux>(next + token * width, cross_ln_w, cross_ln_b, norm);
    linear<Variant, k_embedding_length, k_embedding_length, Aux>(
        cross_q_w, cross_q_b, norm, q + token * width);
    std::fill_n(attn + token * width, static_cast<size_t>(width), 0.0f);
    for (uint64_t head = 0;
         head < static_cast<uint64_t>(k_attention_head_count); ++head) {
      for (uint64_t frame = 0; frame < encoder_frames; ++frame) {
        float score = 0.0f;
        for (uint64_t dim = 0;
             dim < static_cast<uint64_t>(k_attention_head_dim); ++dim) {
          const uint64_t offset =
              head * static_cast<uint64_t>(k_attention_head_dim) + dim;
          score += q[token * width + offset] *
                   cross_k[frame * static_cast<uint64_t>(k_embedding_length) +
                           offset];
        }
        scores[frame] = score * scale;
      }
      softmax(scores, encoder_frames);
      for (uint64_t frame = 0; frame < encoder_frames; ++frame) {
        for (uint64_t dim = 0;
             dim < static_cast<uint64_t>(k_attention_head_dim); ++dim) {
          const uint64_t offset =
              head * static_cast<uint64_t>(k_attention_head_dim) + dim;
          attn[token * width + offset] +=
              scores[frame] *
              cross_v[frame * static_cast<uint64_t>(k_embedding_length) +
                      offset];
        }
      }
    }
    linear<Variant, k_embedding_length, k_embedding_length, Aux>(
        cross_o_w, cross_o_b, attn + token * width, norm);
    for (uint64_t dim = 0; dim < width; ++dim) {
      hidden[token * width + dim] = next[token * width + dim] + norm[dim];
    }
  }

  for (uint64_t token = 0; token < token_count; ++token) {
    layer_norm_frame<Aux>(hidden + token * width, final_ln_w, final_ln_b, norm);
    linear<Variant, k_embedding_length, k_feed_forward_length, Aux>(
        fc1_w, fc1_b, norm, ff);
    for (uint64_t dim = 0; dim < static_cast<uint64_t>(k_feed_forward_length);
         ++dim) {
      ff[dim] = gelu(ff[dim]);
    }
    linear<Variant, k_feed_forward_length, k_embedding_length, Aux>(
        fc2_w, fc2_b, ff, norm);
    for (uint64_t dim = 0; dim < width; ++dim) {
      hidden[token * width + dim] += norm[dim];
    }
  }
}

inline uint64_t digest_f32(const float *values, const uint64_t count) noexcept {
  uint64_t hash = 1469598103934665603ull;
  for (uint64_t index = 0; index < count; ++index) {
    uint32_t bits = 0u;
    std::memcpy(&bits, values + index, sizeof(bits));
    hash ^= static_cast<uint64_t>(bits);
    hash *= 1099511628211ull;
  }
  return hash;
}

template <linear_weight_variant Variant,
          aux_weight_variant Aux = aux_weight_variant::q8_0>
inline void compute_decoder_logits_for_tokens(
    const emel::model::data &model, const uint64_t encoder_frames,
    const float *cross_k_cache, const float *cross_v_cache,
    const int32_t *tokens, const uint64_t token_count, float *workspace,
    float *logits, float &confidence_out, uint64_t &digest_out) noexcept {
  float *hidden = workspace;
  const uint64_t hidden_count =
      static_cast<uint64_t>(k_embedding_length) * token_count;
  float *next = hidden + hidden_count;
  float *q = next + hidden_count;
  float *k = q + hidden_count;
  float *v = k + hidden_count;
  float *attn = v + hidden_count;
  float *norm = attn + hidden_count;
  float *ff = norm + static_cast<uint64_t>(k_embedding_length);
  float *scores = ff + static_cast<uint64_t>(k_feed_forward_length);

  const auto &token_embedding = *find_tensor(
      model, "model.decoder.embed_tokens.weight"); // GCOVR_EXCL_BR_LINE
  const auto &position_embedding = *find_tensor(
      model, "model.decoder.embed_positions.weight"); // GCOVR_EXCL_BR_LINE
  for (uint64_t token = 0; token < token_count; ++token) {
    for (uint64_t dim = 0; dim < static_cast<uint64_t>(k_embedding_length);
         ++dim) {
      const uint64_t token_index =
          static_cast<uint64_t>(tokens[token]) *
              static_cast<uint64_t>(k_embedding_length) +
          dim;
      const uint64_t position_index =
          token * static_cast<uint64_t>(k_embedding_length) + dim;
      hidden[token * static_cast<uint64_t>(k_embedding_length) + dim] =
          read_q8_0_value(token_embedding, token_index) +
          read_aux_matrix<Aux>(position_embedding, position_index);
    }
  }

  for (uint64_t layer = 0; layer < static_cast<uint64_t>(k_decoder_block_count);
       ++layer) {
    const uint64_t layer_offset =
        layer * encoder_frames * static_cast<uint64_t>(k_embedding_length);
    run_decoder_layer_sequence<Variant, Aux>(
        model, layer, encoder_frames, token_count, cross_k_cache + layer_offset,
        cross_v_cache + layer_offset, hidden, next, q, k, v, attn, norm, ff,
        scores);
  }

  const auto &final_w = *find_tensor(
      model, "model.decoder.layer_norm.weight"); // GCOVR_EXCL_BR_LINE
  const auto &final_b = *find_tensor(
      model, "model.decoder.layer_norm.bias"); // GCOVR_EXCL_BR_LINE
  const uint64_t last_token = token_count - 1u;
  layer_norm_frame<Aux>(hidden + last_token *
                                     static_cast<uint64_t>(k_embedding_length),
                        final_w, final_b, norm);

  if constexpr (Variant == linear_weight_variant::q8_0) {
    linear_no_bias<Variant, k_embedding_length, k_vocab_size>(token_embedding,
                                                              norm, logits);
  } else {
    for (uint64_t token = 0; token < static_cast<uint64_t>(k_vocab_size);
         ++token) {
      logits[token] = dot_linear_row<Variant>(token_embedding, token, norm,
                                              k_embedding_length);
    }
  }
  float best_score = logits[0];
  for (uint64_t token = 1; token < static_cast<uint64_t>(k_vocab_size);
       ++token) {
    best_score = std::max(best_score, logits[token]);
  }
  confidence_out = best_score;
  digest_out = digest_f32(norm, static_cast<uint64_t>(k_embedding_length));
}

inline int32_t select_greedy_timestamp_aware_token(
    const decode_policy_runtime &policy, const float *logits,
    const int32_t *generated_tokens, const uint64_t generated_token_count,
    const bool initial_token, float &confidence_out) noexcept {
  const bool last_was_timestamp =
      generated_token_count > 0u &&
      generated_tokens[generated_token_count - 1u] >= policy.timestamp_begin;
  const bool penultimate_was_timestamp =
      generated_token_count < 2u ||
      generated_tokens[generated_token_count - 2u] >= policy.timestamp_begin;

  float timestamp_max = -INFINITY;
  for (int32_t token = policy.timestamp_begin; token < k_vocab_size; ++token) {
    const bool blocked_by_pair_rule =
        last_was_timestamp && penultimate_was_timestamp;
    const bool blocked_by_initial_limit =
        initial_token && token > policy.timestamp_begin + 50;
    const float score = blocked_by_pair_rule || blocked_by_initial_limit
                            ? -INFINITY
                            : logits[token];
    timestamp_max = std::max(timestamp_max, score);
  }
  float text_max = -INFINITY;
  for (int32_t token = 0; token < policy.timestamp_begin; ++token) {
    const bool initial_suppressed =
        initial_token && (token == policy.eot || token == policy.space);
    const bool control_suppressed =
        token == policy.sot || token == policy.no_speech ||
        token == policy.notimestamps || token == policy.translate ||
        token == policy.transcribe;
    const bool blocked_by_pair_rule =
        last_was_timestamp && !penultimate_was_timestamp && token < policy.eot;
    const float score =
        initial_suppressed || control_suppressed || blocked_by_pair_rule
            ? -INFINITY
            : logits[token];
    text_max = std::max(text_max, score);
  }

  float timestamp_sum = -INFINITY;
  if (timestamp_max > -INFINITY) {
    float sum = 0.0f;
    for (int32_t token = policy.timestamp_begin; token < k_vocab_size;
         ++token) {
      const bool blocked_by_pair_rule =
          last_was_timestamp && penultimate_was_timestamp;
      const bool blocked_by_initial_limit =
          initial_token && token > policy.timestamp_begin + 50;
      if (!blocked_by_pair_rule && !blocked_by_initial_limit) {
        sum += std::exp(logits[token] - timestamp_max);
      }
    }
    timestamp_sum = std::log(sum) + timestamp_max;
  }
  const bool force_timestamp = timestamp_sum > text_max;

  int32_t best_token = 0;
  float best_score = -INFINITY;
  for (int32_t token = 0; token < k_vocab_size; ++token) {
    const bool initial_suppressed =
        initial_token && (token == policy.eot || token == policy.space);
    const bool control_suppressed =
        token == policy.sot || token == policy.no_speech ||
        token == policy.notimestamps || token == policy.translate ||
        token == policy.transcribe;
    const bool timestamp_token = token >= policy.timestamp_begin;
    const bool blocked_by_pair_rule =
        (last_was_timestamp && penultimate_was_timestamp && timestamp_token) ||
        (last_was_timestamp && !penultimate_was_timestamp &&
         token < policy.eot);
    const bool blocked_by_initial_limit = initial_token && timestamp_token &&
                                          token > policy.timestamp_begin + 50;
    const bool blocked_by_timestamp_mass = force_timestamp && !timestamp_token;
    const float score =
        initial_suppressed || control_suppressed || blocked_by_pair_rule ||
                blocked_by_initial_limit || blocked_by_timestamp_mass
            ? -INFINITY
            : logits[token];
    if (score > best_score) {
      best_score = score;
      best_token = token;
    }
  }
  confidence_out = best_score;
  return best_token;
}

template <linear_weight_variant Variant,
          aux_weight_variant Aux = aux_weight_variant::q8_0>
inline uint64_t run_decoder_sequence(
    const emel::model::data &model, const float *encoder_state,
    const uint64_t encoder_frames, const decode_policy_runtime &policy,
    const int32_t *prompt_tokens, const uint64_t prompt_token_count,
    float *workspace, float *logits, int32_t *generated_tokens,
    const uint64_t generated_token_capacity,
    uint64_t &generated_token_count_out, int32_t &token_out,
    float &confidence_out) noexcept {
  std::array<int32_t, static_cast<size_t>(k_decoder_sequence_token_count)>
      tokens = {};
  for (uint64_t index = 0; index < prompt_token_count; ++index) {
    tokens[index] = prompt_tokens[index];
  }
  const uint64_t generation_limit =
      std::min<uint64_t>(generated_token_capacity,
                         static_cast<uint64_t>(k_decoder_sequence_token_count) -
                             prompt_token_count);
  const uint64_t cross_cache_count =
      static_cast<uint64_t>(k_decoder_block_count) * encoder_frames *
      static_cast<uint64_t>(k_embedding_length);
  float *cross_k_cache = workspace;
  float *cross_v_cache = cross_k_cache + cross_cache_count;
  float *step_workspace = cross_v_cache + cross_cache_count;
  compute_decoder_cross_cache<Variant, Aux>(
      model, encoder_state, encoder_frames, cross_k_cache, cross_v_cache);
  uint64_t token_count = prompt_token_count;
  uint64_t digest = 0u;
  generated_token_count_out = 0u;
  for (uint64_t step = 0; step < generation_limit; ++step) {
    float raw_confidence = 0.0f;
    compute_decoder_logits_for_tokens<Variant, Aux>(
        model, encoder_frames, cross_k_cache, cross_v_cache, tokens.data(),
        token_count, step_workspace, logits, raw_confidence, digest);
    const int32_t next_token = select_greedy_timestamp_aware_token(
        policy, logits, generated_tokens, step, step == 0u, confidence_out);
    token_out = next_token;
    generated_tokens[step] = next_token;
    generated_token_count_out = step + 1u;
    tokens[token_count] = next_token;
    ++token_count;
    if (next_token == policy.eot ||
        (next_token >= policy.timestamp_begin && step > 0u)) {
      step = generation_limit;
    }
  }
  return digest;
}

}  // namespace emel::speech::decoder::whisper::detail
