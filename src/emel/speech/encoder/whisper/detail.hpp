#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "emel/error/error.hpp"
#include "emel/kernel/aarch64/actions.hpp"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/model/data.hpp"
#include "emel/speech/encoder/whisper/errors.hpp"

namespace emel::speech::encoder::whisper::detail {

inline emel::error::type to_error(const error value) noexcept {
  return emel::error::cast(value);
}

inline constexpr int32_t k_sample_rate = 16000;
inline constexpr int32_t k_channel_count = 1;
inline constexpr int32_t k_mel_bin_count = 80;
inline constexpr int32_t k_fft_size = 400;
inline constexpr int32_t k_fft_bin_count = 201;
inline constexpr int32_t k_hop_length = 160;
inline constexpr int32_t k_embedding_length = 384;
inline constexpr int32_t k_feed_forward_length = 1536;
inline constexpr int32_t k_attention_head_count = 6;
inline constexpr int32_t k_attention_head_dim = 64;
inline constexpr int32_t k_encoder_block_count = 4;
inline constexpr int32_t k_max_mel_frame_count = 3000;
inline constexpr int32_t k_max_encoder_frame_count = 1500;
inline constexpr int32_t k_bluestein_fft_size = 1024;
inline constexpr float k_pi = 3.14159265358979323846f;
inline constexpr float k_layer_norm_epsilon = 1.0e-5f;
inline constexpr float k_log_zero_guard = 1.0e-10f;

struct execution_contract {
  const emel::model::data *model = nullptr;
  int32_t sample_rate = 0;
  int32_t mel_bin_count = 0;
  int32_t embedding_length = 0;
  int32_t feed_forward_length = 0;
  int32_t attention_head_count = 0;
  int32_t encoder_block_count = 0;
};

inline execution_contract
bind_execution_contract(const emel::model::data &model) noexcept {
  return execution_contract{
      .model = &model,
      .sample_rate = k_sample_rate,
      .mel_bin_count = k_mel_bin_count,
      .embedding_length = k_embedding_length,
      .feed_forward_length = k_feed_forward_length,
      .attention_head_count = k_attention_head_count,
      .encoder_block_count = k_encoder_block_count,
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

inline uint64_t
mel_frame_count_for_samples(const uint64_t sample_count) noexcept {
  const uint64_t frames =
      (sample_count + static_cast<uint64_t>(k_hop_length - 1)) /
      static_cast<uint64_t>(k_hop_length);
  return std::min<uint64_t>(frames,
                            static_cast<uint64_t>(k_max_mel_frame_count));
}

inline uint64_t
encoder_frame_count_for_mel_frames(const uint64_t mel_frames) noexcept {
  return (mel_frames + 1u) / 2u;
}

inline uint64_t
required_encoder_output_floats(const uint64_t sample_count) noexcept {
  return encoder_frame_count_for_mel_frames(
             mel_frame_count_for_samples(sample_count)) *
         static_cast<uint64_t>(k_embedding_length);
}

inline uint64_t
required_workspace_floats(const uint64_t sample_count) noexcept {
  const uint64_t mel_frames = mel_frame_count_for_samples(sample_count);
  const uint64_t encoder_frames =
      encoder_frame_count_for_mel_frames(mel_frames);
  return (static_cast<uint64_t>(k_mel_bin_count) * mel_frames) +
         (static_cast<uint64_t>(k_embedding_length) * mel_frames) +
         (static_cast<uint64_t>(k_embedding_length) * encoder_frames * 6u) +
         static_cast<uint64_t>(k_embedding_length) +
         static_cast<uint64_t>(k_feed_forward_length) + encoder_frames +
         static_cast<uint64_t>(k_fft_size) * 3u +
         static_cast<uint64_t>(k_bluestein_fft_size) * 4u;
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

inline float read_f16_value(const void *data, const uint64_t index) noexcept {
  const auto *words = static_cast<const uint16_t *>(data);
  return ::emel::kernel::detail::quant::fp16_to_fp32(words[index]);
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

inline void fft_radix2(float *real, float *imag, const uint64_t count,
                       const bool inverse) noexcept {
  uint64_t j = 0u;
  for (uint64_t i = 1u; i < count; ++i) {
    uint64_t bit = count >> 1u;
    while ((j & bit) != 0u) {
      j ^= bit;
      bit >>= 1u;
    }
    j ^= bit;
    if (i < j) {
      std::swap(real[i], real[j]);
      std::swap(imag[i], imag[j]);
    }
  }

  for (uint64_t len = 2u; len <= count; len <<= 1u) {
    const float sign = inverse ? 1.0f : -1.0f;
    const float angle = sign * 2.0f * k_pi / static_cast<float>(len);
    const float w_len_real = std::cos(angle);
    const float w_len_imag = std::sin(angle);
    for (uint64_t offset = 0u; offset < count; offset += len) {
      float w_real = 1.0f;
      float w_imag = 0.0f;
      const uint64_t half = len >> 1u;
      for (uint64_t index = 0u; index < half; ++index) {
        const uint64_t even = offset + index;
        const uint64_t odd = even + half;
        const float odd_real = real[odd] * w_real - imag[odd] * w_imag;
        const float odd_imag = real[odd] * w_imag + imag[odd] * w_real;
        real[odd] = real[even] - odd_real;
        imag[odd] = imag[even] - odd_imag;
        real[even] += odd_real;
        imag[even] += odd_imag;
        const float next_real = w_real * w_len_real - w_imag * w_len_imag;
        const float next_imag = w_real * w_len_imag + w_imag * w_len_real;
        w_real = next_real;
        w_imag = next_imag;
      }
    }
  }

  const float scale = inverse ? 1.0f / static_cast<float>(count) : 1.0f;
  for (uint64_t idx = 0; idx < count; ++idx) {
    real[idx] *= scale;
    imag[idx] *= scale;
  }
}

inline float hann_window_value(const uint64_t sample) noexcept {
  const float angle =
      2.0f * k_pi * static_cast<float>(sample) / static_cast<float>(k_fft_size);
  return 0.5f - 0.5f * std::cos(angle);
}

inline void prepare_bluestein_kernel(float *kernel_real,
                                     float *kernel_imag) noexcept {
  std::fill_n(kernel_real, static_cast<size_t>(k_bluestein_fft_size), 0.0f);
  std::fill_n(kernel_imag, static_cast<size_t>(k_bluestein_fft_size), 0.0f);
  for (uint64_t n = 0; n < static_cast<uint64_t>(k_fft_size); ++n) {
    const float angle =
        k_pi * static_cast<float>(n * n) / static_cast<float>(k_fft_size);
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    kernel_real[n] = c;
    kernel_imag[n] = s;
    const uint64_t mirror = static_cast<uint64_t>(k_bluestein_fft_size) - n;
    const uint64_t active = static_cast<uint64_t>(n != 0u);
    kernel_real[mirror * active] += c * static_cast<float>(active);
    kernel_imag[mirror * active] += s * static_cast<float>(active);
  }
  fft_radix2(kernel_real, kernel_imag, k_bluestein_fft_size, false);
}

inline void prepare_frame_tables(float *window, float *chirp_real,
                                 float *chirp_imag) noexcept {
  for (uint64_t n = 0; n < static_cast<uint64_t>(k_fft_size); ++n) {
    window[n] = hann_window_value(n);
    const float angle =
        -k_pi * static_cast<float>(n * n) / static_cast<float>(k_fft_size);
    chirp_real[n] = std::cos(angle);
    chirp_imag[n] = std::sin(angle);
  }
}

inline void compute_bluestein_power(const float *frame,
                                    const float *kernel_real,
                                    const float *kernel_imag,
                                    const float *chirp_real,
                                    const float *chirp_imag, float *work_real,
                                    float *work_imag, float *power) noexcept {
  std::fill_n(work_real, static_cast<size_t>(k_bluestein_fft_size), 0.0f);
  std::fill_n(work_imag, static_cast<size_t>(k_bluestein_fft_size), 0.0f);
  for (uint64_t n = 0; n < static_cast<uint64_t>(k_fft_size); ++n) {
    work_real[n] = frame[n] * chirp_real[n];
    work_imag[n] = frame[n] * chirp_imag[n];
  }
  fft_radix2(work_real, work_imag, k_bluestein_fft_size, false);
  for (uint64_t idx = 0; idx < static_cast<uint64_t>(k_bluestein_fft_size);
       ++idx) {
    const float real =
        work_real[idx] * kernel_real[idx] - work_imag[idx] * kernel_imag[idx];
    const float imag =
        work_real[idx] * kernel_imag[idx] + work_imag[idx] * kernel_real[idx];
    work_real[idx] = real;
    work_imag[idx] = imag;
  }
  fft_radix2(work_real, work_imag, k_bluestein_fft_size, true);
  for (uint64_t k = 0; k < static_cast<uint64_t>(k_fft_bin_count); ++k) {
    const float real =
        work_real[k] * chirp_real[k] - work_imag[k] * chirp_imag[k];
    const float imag =
        work_real[k] * chirp_imag[k] + work_imag[k] * chirp_real[k];
    power[k] = real * real + imag * imag;
  }
}

inline void
compute_mel_features(const float *pcm, const uint64_t sample_count,
                     const emel::model::data::tensor_record &mel_filters,
                     float *mel, float *fft_real, float *fft_imag,
                     float *kernel_real, float *kernel_imag, float *window,
                     float *chirp_real, float *chirp_imag) noexcept {
  const uint64_t frames = mel_frame_count_for_samples(sample_count);
  prepare_bluestein_kernel(kernel_real, kernel_imag);
  prepare_frame_tables(window, chirp_real, chirp_imag);
  std::array<float, k_fft_size> frame = {};
  std::array<float, k_fft_bin_count> power = {};
  for (uint64_t frame_index = 0; frame_index < frames; ++frame_index) {
    const uint64_t offset = frame_index * static_cast<uint64_t>(k_hop_length);
    const uint64_t active_padded_samples =
        sample_count + static_cast<uint64_t>(k_fft_size / 2);
    for (uint64_t sample = 0; sample < static_cast<uint64_t>(k_fft_size);
         ++sample) {
      const uint64_t padded_index = offset + sample;
      float value = 0.0f;
      if (padded_index < active_padded_samples) {
        if (padded_index < static_cast<uint64_t>(k_fft_size / 2)) {
          const uint64_t source =
              static_cast<uint64_t>(k_fft_size / 2) - padded_index;
          value = source < sample_count ? pcm[source] : 0.0f;
        } else {
          const uint64_t source =
              padded_index - static_cast<uint64_t>(k_fft_size / 2);
          value = source < sample_count ? pcm[source] : 0.0f;
        }
      }
      frame[sample] = value * window[sample];
    }
    compute_bluestein_power(frame.data(), kernel_real, kernel_imag, chirp_real,
                            chirp_imag, fft_real, fft_imag, power.data());

    for (uint64_t mel_bin = 0; mel_bin < static_cast<uint64_t>(k_mel_bin_count);
         ++mel_bin) {
      float energy = 0.0f;
      for (uint64_t bin = 0; bin < static_cast<uint64_t>(k_fft_bin_count);
           ++bin) {
        const uint64_t filter_index =
            bin + static_cast<uint64_t>(k_fft_bin_count) * mel_bin;
        energy += read_f32_value(mel_filters.data, filter_index) * power[bin];
      }
      mel[mel_bin * frames + frame_index] =
          std::log10(std::max(energy, k_log_zero_guard));
    }
  }

  float max_value = mel[0];
  for (uint64_t index = 1u;
       index < static_cast<uint64_t>(k_mel_bin_count) * frames; ++index) {
    max_value = std::max(max_value, mel[index]);
  }
  const float floor_value = max_value - 8.0f;
  for (uint64_t index = 0u;
       index < static_cast<uint64_t>(k_mel_bin_count) * frames; ++index) {
    mel[index] = (std::max(mel[index], floor_value) + 4.0f) * 0.25f;
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

inline float read_conv_weight(const emel::model::data::tensor_record &tensor,
                              const uint64_t kernel,
                              const uint64_t input_channel,
                              const uint64_t output_channel) noexcept {
  const uint64_t offset =
      kernel + static_cast<uint64_t>(k_fft_size == 400 ? 3u : 3u) *
                   (input_channel +
                    static_cast<uint64_t>(tensor.dims[1]) * output_channel);
  return read_f16_value(tensor.data, offset);
}

template <aux_weight_variant Aux>
inline void run_conv1(const float *mel, const uint64_t mel_frames,
                      const emel::model::data::tensor_record &weight,
                      const emel::model::data::tensor_record &bias,
                      float *output) noexcept {
  const uint64_t width = static_cast<uint64_t>(k_embedding_length);
  for (uint64_t out = 0; out < width; ++out) {
    const float bias_value = read_aux_vector<Aux>(bias, out);
    for (uint64_t frame = 0; frame < mel_frames; ++frame) {
      output[frame * width + out] = bias_value;
    }

    for (uint64_t in = 0; in < static_cast<uint64_t>(k_mel_bin_count); ++in) {
      const float *mel_in = mel + in * mel_frames;
      const float weight0 = read_conv_weight(weight, 0u, in, out);
      const float weight1 = read_conv_weight(weight, 1u, in, out);
      const float weight2 = read_conv_weight(weight, 2u, in, out);
      if (mel_frames > 0u) {
        float acc = output[out];
        acc += mel_in[0] * weight1;
        if (mel_frames > 1u) {
          acc += mel_in[1] * weight2;
        }
        output[out] = acc;
      }
      for (uint64_t frame = 1u; frame + 1u < mel_frames; ++frame) {
        float acc = output[frame * width + out];
        acc += mel_in[frame - 1u] * weight0;
        acc += mel_in[frame] * weight1;
        acc += mel_in[frame + 1u] * weight2;
        output[frame * width + out] = acc;
      }
      if (mel_frames > 1u) {
        const uint64_t frame = mel_frames - 1u;
        float acc = output[frame * width + out];
        acc += mel_in[frame - 1u] * weight0;
        acc += mel_in[frame] * weight1;
        output[frame * width + out] = acc;
      }
    }

    for (uint64_t frame = 0; frame < mel_frames; ++frame) {
      float *value = output + frame * width + out;
      *value = gelu(*value);
    }
  }
}

template <aux_weight_variant Aux>
inline void run_conv2(const float *input, const uint64_t mel_frames,
                      const emel::model::data::tensor_record &weight,
                      const emel::model::data::tensor_record &bias,
                      float *output) noexcept {
  const uint64_t encoder_frames =
      encoder_frame_count_for_mel_frames(mel_frames);
  const uint64_t width = static_cast<uint64_t>(k_embedding_length);
  for (uint64_t out = 0; out < width; ++out) {
    const float bias_value = read_aux_vector<Aux>(bias, out);
    for (uint64_t frame = 0; frame < encoder_frames; ++frame) {
      output[frame * width + out] = bias_value;
    }

    for (uint64_t in = 0; in < width; ++in) {
      const float weight0 = read_conv_weight(weight, 0u, in, out);
      const float weight1 = read_conv_weight(weight, 1u, in, out);
      const float weight2 = read_conv_weight(weight, 2u, in, out);
      for (uint64_t frame = 0; frame < encoder_frames; ++frame) {
        const uint64_t center = frame * 2u;
        float acc = output[frame * width + out];
        if (center > 0u) {
          acc += input[(center - 1u) * width + in] * weight0;
        }
        if (center < mel_frames) {
          acc += input[center * width + in] * weight1;
        }
        if (center + 1u < mel_frames) {
          acc += input[(center + 1u) * width + in] * weight2;
        }
        output[frame * width + out] = acc;
      }
    }

    for (uint64_t frame = 0; frame < encoder_frames; ++frame) {
      float *value = output + frame * width + out;
      *value = gelu(*value);
    }
  }
}

template <aux_weight_variant Aux>
inline void add_positional_embedding(
    float *hidden, const uint64_t encoder_frames,
    const emel::model::data::tensor_record &positions) noexcept {
  for (uint64_t frame = 0; frame < encoder_frames; ++frame) {
    for (uint64_t dim = 0; dim < static_cast<uint64_t>(k_embedding_length);
         ++dim) {
      hidden[frame * static_cast<uint64_t>(k_embedding_length) + dim] +=
          read_aux_matrix<Aux>(
              positions,
              frame * static_cast<uint64_t>(k_embedding_length) + dim);
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
inline void
run_encoder_layer(const emel::model::data &model, const uint64_t layer,
                  const uint64_t encoder_frames, float *hidden, float *next,
                  float *q, float *k, float *v, float *attn, float *norm,
                  float *ff, float *scores) noexcept {
  char name[96] = {};
  const auto layer_tensor = [&](const char *suffix) noexcept {
    const uint64_t name_size =
        write_layer_tensor_name(name, "model.encoder.layers.", layer, suffix);
    return *find_tensor(model,
                        std::string_view{name, static_cast<size_t>(name_size)});
  };

  const auto &ln1_w = layer_tensor("self_attn_layer_norm.weight");
  const auto &ln1_b = layer_tensor("self_attn_layer_norm.bias");
  const auto &q_w = layer_tensor("self_attn.q_proj.weight");
  const auto &q_b = layer_tensor("self_attn.q_proj.bias");
  const auto &k_w = layer_tensor("self_attn.k_proj.weight");
  const auto &v_w = layer_tensor("self_attn.v_proj.weight");
  const auto &v_b = layer_tensor("self_attn.v_proj.bias");
  const auto &o_w = layer_tensor("self_attn.out_proj.weight");
  const auto &o_b = layer_tensor("self_attn.out_proj.bias");
  const auto &ln2_w = layer_tensor("final_layer_norm.weight");
  const auto &ln2_b = layer_tensor("final_layer_norm.bias");
  const auto &fc1_w = layer_tensor("fc1.weight");
  const auto &fc1_b = layer_tensor("fc1.bias");
  const auto &fc2_w = layer_tensor("fc2.weight");
  const auto &fc2_b = layer_tensor("fc2.bias");

  for (uint64_t frame = 0; frame < encoder_frames; ++frame) {
    const float *frame_in =
        hidden + frame * static_cast<uint64_t>(k_embedding_length);
    layer_norm_frame<Aux>(frame_in, ln1_w, ln1_b, norm);
    linear<Variant, k_embedding_length, k_embedding_length, Aux>(
        q_w, q_b, norm, q + frame * static_cast<uint64_t>(k_embedding_length));
    linear_no_bias<Variant, k_embedding_length, k_embedding_length>(
        k_w, norm, k + frame * static_cast<uint64_t>(k_embedding_length));
    linear<Variant, k_embedding_length, k_embedding_length, Aux>(
        v_w, v_b, norm, v + frame * static_cast<uint64_t>(k_embedding_length));
  }

  const float scale =
      1.0f / std::sqrt(static_cast<float>(k_attention_head_dim));
  for (uint64_t frame = 0; frame < encoder_frames; ++frame) {
    std::fill_n(attn + frame * static_cast<uint64_t>(k_embedding_length),
                static_cast<size_t>(k_embedding_length), 0.0f);
    for (uint64_t head = 0;
         head < static_cast<uint64_t>(k_attention_head_count); ++head) {
      for (uint64_t key_frame = 0; key_frame < encoder_frames; ++key_frame) {
        float score = 0.0f;
        for (uint64_t dim = 0;
             dim < static_cast<uint64_t>(k_attention_head_dim); ++dim) {
          const uint64_t offset =
              head * static_cast<uint64_t>(k_attention_head_dim) + dim;
          score +=
              q[frame * static_cast<uint64_t>(k_embedding_length) + offset] *
              k[key_frame * static_cast<uint64_t>(k_embedding_length) + offset];
        }
        scores[key_frame] = score * scale;
      }
      softmax(scores, encoder_frames);
      for (uint64_t key_frame = 0; key_frame < encoder_frames; ++key_frame) {
        for (uint64_t dim = 0;
             dim < static_cast<uint64_t>(k_attention_head_dim); ++dim) {
          const uint64_t offset =
              head * static_cast<uint64_t>(k_attention_head_dim) + dim;
          attn[frame * static_cast<uint64_t>(k_embedding_length) + offset] +=
              scores[key_frame] *
              v[key_frame * static_cast<uint64_t>(k_embedding_length) + offset];
        }
      }
    }
    linear<Variant, k_embedding_length, k_embedding_length, Aux>(
        o_w, o_b, attn + frame * static_cast<uint64_t>(k_embedding_length),
        norm);
    for (uint64_t dim = 0; dim < static_cast<uint64_t>(k_embedding_length);
         ++dim) {
      next[frame * static_cast<uint64_t>(k_embedding_length) + dim] =
          hidden[frame * static_cast<uint64_t>(k_embedding_length) + dim] +
          norm[dim];
    }
  }

  for (uint64_t frame = 0; frame < encoder_frames; ++frame) {
    float *frame_data =
        next + frame * static_cast<uint64_t>(k_embedding_length);
    layer_norm_frame<Aux>(frame_data, ln2_w, ln2_b, norm);
    linear<Variant, k_embedding_length, k_feed_forward_length, Aux>(
        fc1_w, fc1_b, norm, ff);
    for (uint64_t dim = 0; dim < static_cast<uint64_t>(k_feed_forward_length);
         ++dim) {
      ff[dim] = gelu(ff[dim]);
    }
    linear<Variant, k_feed_forward_length, k_embedding_length, Aux>(
        fc2_w, fc2_b, ff, norm);
    for (uint64_t dim = 0; dim < static_cast<uint64_t>(k_embedding_length);
         ++dim) {
      hidden[frame * static_cast<uint64_t>(k_embedding_length) + dim] =
          frame_data[dim] + norm[dim];
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
inline uint64_t run_encoder(const emel::model::data &model, const float *pcm,
                            const uint64_t sample_count, float *workspace,
                            float *output,
                            uint64_t &encoder_frames_out) noexcept {
  const uint64_t mel_frames = mel_frame_count_for_samples(sample_count);
  const uint64_t encoder_frames =
      encoder_frame_count_for_mel_frames(mel_frames);
  encoder_frames_out = encoder_frames;

  float *mel = workspace;
  float *conv1 = mel + static_cast<uint64_t>(k_mel_bin_count) * mel_frames;
  float *hidden =
      conv1 + static_cast<uint64_t>(k_embedding_length) * mel_frames;
  float *next =
      hidden + static_cast<uint64_t>(k_embedding_length) * encoder_frames;
  float *q = next + static_cast<uint64_t>(k_embedding_length) * encoder_frames;
  float *k = q + static_cast<uint64_t>(k_embedding_length) * encoder_frames;
  float *v = k + static_cast<uint64_t>(k_embedding_length) * encoder_frames;
  float *attn = v + static_cast<uint64_t>(k_embedding_length) * encoder_frames;
  float *norm =
      attn + static_cast<uint64_t>(k_embedding_length) * encoder_frames;
  float *ff = norm + static_cast<uint64_t>(k_embedding_length);
  float *scores = ff + static_cast<uint64_t>(k_feed_forward_length);
  float *fft_real = scores + encoder_frames;
  float *fft_imag = fft_real + static_cast<uint64_t>(k_bluestein_fft_size);
  float *kernel_real = fft_imag + static_cast<uint64_t>(k_bluestein_fft_size);
  float *kernel_imag =
      kernel_real + static_cast<uint64_t>(k_bluestein_fft_size);
  float *window = kernel_imag + static_cast<uint64_t>(k_bluestein_fft_size);
  float *chirp_real = window + static_cast<uint64_t>(k_fft_size);
  float *chirp_imag = chirp_real + static_cast<uint64_t>(k_fft_size);

  const auto &mel_filters =
      *find_tensor(model, "mel_filters"); // GCOVR_EXCL_BR_LINE
  const auto &conv1_w =
      *find_tensor(model, "model.encoder.conv1.weight"); // GCOVR_EXCL_BR_LINE
  const auto &conv1_b =
      *find_tensor(model, "model.encoder.conv1.bias"); // GCOVR_EXCL_BR_LINE
  const auto &conv2_w =
      *find_tensor(model, "model.encoder.conv2.weight"); // GCOVR_EXCL_BR_LINE
  const auto &conv2_b =
      *find_tensor(model, "model.encoder.conv2.bias"); // GCOVR_EXCL_BR_LINE
  const auto &positions = *find_tensor(
      model, "model.encoder.embed_positions.weight"); // GCOVR_EXCL_BR_LINE
  compute_mel_features(pcm, sample_count, mel_filters, mel, fft_real, fft_imag,
                       kernel_real, kernel_imag, window, chirp_real,
                       chirp_imag);
  run_conv1<Aux>(mel, mel_frames, conv1_w, conv1_b, conv1);
  run_conv2<Aux>(conv1, mel_frames, conv2_w, conv2_b, hidden);
  add_positional_embedding<Aux>(hidden, encoder_frames, positions);

  for (uint64_t layer = 0; layer < static_cast<uint64_t>(k_encoder_block_count);
       ++layer) {
    run_encoder_layer<Variant, Aux>(model, layer, encoder_frames, hidden, next,
                                    q, k, v, attn, norm, ff, scores);
  }

  const auto &final_w = *find_tensor(
      model, "model.encoder.layer_norm.weight"); // GCOVR_EXCL_BR_LINE
  const auto &final_b = *find_tensor(
      model, "model.encoder.layer_norm.bias"); // GCOVR_EXCL_BR_LINE
  for (uint64_t frame = 0; frame < encoder_frames; ++frame) {
    layer_norm_frame<Aux>(
        hidden + frame * static_cast<uint64_t>(k_embedding_length), final_w,
        final_b, output + frame * static_cast<uint64_t>(k_embedding_length));
  }
  return digest_f32(output,
                    encoder_frames * static_cast<uint64_t>(k_embedding_length));
}

} // namespace emel::speech::encoder::whisper::detail
