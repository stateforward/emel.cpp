#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"

namespace emel::kernel::test {

using dtype = emel::kernel::event::dtype;
using tensor_view = emel::kernel::event::tensor_view;
using tensor_view_mut = emel::kernel::event::tensor_view_mut;

template <class tensor_type>
inline void fill_default_nb(tensor_type & tensor) {
  const auto elem_size =
      static_cast<uint64_t>(emel::kernel::detail::dtype_size_bytes(
          emel::kernel::detail::dtype_code(tensor.type)));
  tensor.nb[0] = elem_size;
  tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
  tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
  tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
}

inline tensor_view make_src(const void * data, const dtype type, const uint64_t ne0,
                            const uint64_t ne1 = 1, const uint64_t ne2 = 1,
                            const uint64_t ne3 = 1) {
  tensor_view out{};
  out.data = data;
  out.type = type;
  out.ne = {ne0, ne1, ne2, ne3};
  fill_default_nb(out);
  return out;
}

inline tensor_view make_quantized_src(const void * data,
                                      const dtype type,
                                      const uint64_t ne0,
                                      const uint64_t ne1 = 1) {
  tensor_view out{};
  const size_t row_bytes =
      emel::kernel::detail::quantized_row_storage_bytes(
          emel::kernel::detail::dtype_code(type), ne0);
  out.data = data;
  out.type = type;
  out.ne = {ne0, ne1, 1, 1};
  out.nb[0] = 1;
  out.nb[1] = row_bytes;
  out.nb[2] = row_bytes * ne1;
  out.nb[3] = out.nb[2];
  return out;
}

inline tensor_view_mut make_dst(void * data, const dtype type, const uint64_t ne0,
                                const uint64_t ne1 = 1, const uint64_t ne2 = 1,
                                const uint64_t ne3 = 1) {
  tensor_view_mut out{};
  out.data = data;
  out.type = type;
  out.ne = {ne0, ne1, ne2, ne3};
  fill_default_nb(out);
  return out;
}

struct flash_attn_ext_fixture {
  float q[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  float k[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
  float v[8] = {2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f};
  float dst[4] = {};
};

inline emel::kernel::event::op_flash_attn_ext make_flash_attn_ext_event(
    flash_attn_ext_fixture & fixture) {
  emel::kernel::event::op_flash_attn_ext ev{};
  ev.src0 = make_src(fixture.q, dtype::f32, 4, 1, 1, 1);
  ev.src1 = make_src(fixture.k, dtype::f32, 4, 2, 1, 1);
  ev.src2 = make_src(fixture.v, dtype::f32, 4, 2, 1, 1);
  ev.dst = make_dst(fixture.dst, dtype::f32, 4, 1, 1, 1);
  ev.nth = 1;

  const float scale = 1.0f;
  std::memcpy(ev.op_params.data(), &scale, sizeof(scale));
  ev.op_params_size = sizeof(scale);
  return ev;
}

inline std::vector<float> flash_attn_reference_f16_scores(std::span<const float> q,
                                                          std::span<const float> k,
                                                          std::span<const float> v,
                                                          const uint64_t head_dim,
                                                          const uint64_t kv_tokens,
                                                          const float scale) {
  std::vector<float> out(static_cast<size_t>(head_dim), 0.0f);
  std::vector<float> rounded_weights(static_cast<size_t>(kv_tokens), 0.0f);
  std::vector<float> value_column(static_cast<size_t>(kv_tokens), 0.0f);

  float max_score = -INFINITY;
  std::vector<float> scores(static_cast<size_t>(kv_tokens), 0.0f);
  for (uint64_t token = 0; token < kv_tokens; ++token) {
    const size_t offset = static_cast<size_t>(token) * static_cast<size_t>(head_dim);
    const float score = emel::kernel::detail::dot_product_ggml_f16_scores(
                            q.data(),
                            k.data() + static_cast<std::ptrdiff_t>(offset),
                            head_dim) *
        scale;
    scores[static_cast<size_t>(token)] = score;
    max_score = std::max(max_score, score);
  }

  double score_sum = 0.0;
  for (uint64_t token = 0; token < kv_tokens; ++token) {
    score_sum += std::exp(scores[static_cast<size_t>(token)] - max_score);
  }

  const float inv_score_sum = score_sum == 0.0 ? 0.0f : static_cast<float>(1.0 / score_sum);
  for (uint64_t token = 0; token < kv_tokens; ++token) {
    const float normalized =
        std::exp(scores[static_cast<size_t>(token)] - max_score) * inv_score_sum;
    rounded_weights[static_cast<size_t>(token)] =
        emel::kernel::detail::round_fp16_weight(normalized);
  }

  for (uint64_t dim = 0; dim < head_dim; ++dim) {
    for (uint64_t token = 0; token < kv_tokens; ++token) {
      const size_t offset = static_cast<size_t>(token) * static_cast<size_t>(head_dim);
      value_column[static_cast<size_t>(token)] = v[offset + static_cast<size_t>(dim)];
    }
    out[static_cast<size_t>(dim)] = emel::kernel::detail::dot_product_ggml_f16_scores(
        value_column.data(), rounded_weights.data(), kv_tokens);
  }

  return out;
}

inline std::vector<float> flash_attn_reference_masked_total_tokens(
    std::span<const float> q,
    std::span<const float> k,
    std::span<const float> v,
    const uint64_t head_dim,
    const uint64_t kv_tokens,
    const uint64_t total_tokens,
    const float scale) {
  std::vector<float> out(static_cast<size_t>(head_dim), 0.0f);
  std::vector<float> scores(static_cast<size_t>(total_tokens), -INFINITY);
  std::vector<float> probs(static_cast<size_t>(total_tokens), 0.0f);
  std::vector<float> rounded_weights(static_cast<size_t>(total_tokens), 0.0f);
  std::vector<float> value_column(static_cast<size_t>(total_tokens), 0.0f);

  float max_score = -INFINITY;
  for (uint64_t token = 0; token < kv_tokens; ++token) {
    const size_t offset = static_cast<size_t>(token) * static_cast<size_t>(head_dim);
    const float score = emel::kernel::detail::dot_product_ggml_f16_scores(
                            q.data(),
                            k.data() + static_cast<std::ptrdiff_t>(offset),
                            head_dim) *
        scale;
    scores[static_cast<size_t>(token)] = score;
    max_score = std::max(max_score, score);
  }

  const double score_sum = emel::kernel::detail::exp_and_sum_ggml_f32(
      scores.data(), probs.data(), total_tokens, max_score);
  const float inv_score_sum = score_sum == 0.0 ? 0.0f : static_cast<float>(1.0 / score_sum);
  for (uint64_t token = 0; token < total_tokens; ++token) {
    const float weight = probs[static_cast<size_t>(token)] * inv_score_sum;
    rounded_weights[static_cast<size_t>(token)] =
        emel::kernel::detail::round_fp16_weight(weight);
  }

  for (uint64_t dim = 0; dim < head_dim; ++dim) {
    for (uint64_t token = 0; token < kv_tokens; ++token) {
      const size_t offset = static_cast<size_t>(token) * static_cast<size_t>(head_dim);
      value_column[static_cast<size_t>(token)] = v[offset + static_cast<size_t>(dim)];
    }
    out[static_cast<size_t>(dim)] = emel::kernel::detail::dot_product_ggml_f16_scores(
        value_column.data(), rounded_weights.data(), total_tokens);
  }

  return out;
}

inline std::vector<float> flash_attn_reference_rounded_weight_float_values(
    std::span<const float> q,
    std::span<const float> k,
    std::span<const float> v,
    const uint64_t head_dim,
    const uint64_t kv_tokens,
    const float scale) {
  std::vector<float> out(static_cast<size_t>(head_dim), 0.0f);
  std::vector<float> rounded_weights(static_cast<size_t>(kv_tokens), 0.0f);

  float max_score = -INFINITY;
  std::vector<float> scores(static_cast<size_t>(kv_tokens), 0.0f);
  for (uint64_t token = 0; token < kv_tokens; ++token) {
    const size_t offset = static_cast<size_t>(token) * static_cast<size_t>(head_dim);
    const float score = emel::kernel::detail::dot_product_ggml_f16_scores(
                            q.data(),
                            k.data() + static_cast<std::ptrdiff_t>(offset),
                            head_dim) *
        scale;
    scores[static_cast<size_t>(token)] = score;
    max_score = std::max(max_score, score);
  }

  double score_sum = 0.0;
  for (uint64_t token = 0; token < kv_tokens; ++token) {
    score_sum += std::exp(scores[static_cast<size_t>(token)] - max_score);
  }

  const float inv_score_sum = score_sum == 0.0 ? 0.0f : static_cast<float>(1.0 / score_sum);
  for (uint64_t token = 0; token < kv_tokens; ++token) {
    const float normalized =
        std::exp(scores[static_cast<size_t>(token)] - max_score) * inv_score_sum;
    rounded_weights[static_cast<size_t>(token)] =
        emel::kernel::detail::round_fp16_weight(normalized);
  }

  for (uint64_t dim = 0; dim < head_dim; ++dim) {
    float value_sum = 0.0f;
    for (uint64_t token = 0; token < kv_tokens; ++token) {
      const size_t offset = static_cast<size_t>(token) * static_cast<size_t>(head_dim);
      value_sum += rounded_weights[static_cast<size_t>(token)] *
          v[offset + static_cast<size_t>(dim)];
    }
    out[static_cast<size_t>(dim)] = value_sum;
  }

  return out;
}

inline std::vector<float> flash_attn_reference_online_softmax_float_values(
    std::span<const float> q,
    std::span<const float> k,
    std::span<const float> v,
    const uint64_t head_dim,
    const uint64_t kv_tokens,
    const float scale) {
  std::vector<float> out(static_cast<size_t>(head_dim), 0.0f);
  float sum = 0.0f;
  float max_score = -INFINITY;

  for (uint64_t token = 0; token < kv_tokens; ++token) {
    const size_t offset = static_cast<size_t>(token) * static_cast<size_t>(head_dim);
    const float score = emel::kernel::detail::dot_product_ggml_f16_scores(
                            q.data(),
                            k.data() + static_cast<std::ptrdiff_t>(offset),
                            head_dim) *
        scale;

    const float old_max = max_score;
    float max_scale = 1.0f;
    float value_scale = 1.0f;
    if (score > max_score) {
      max_score = score;
      max_scale = std::exp(old_max - max_score);
      for (uint64_t dim = 0; dim < head_dim; ++dim) {
        out[static_cast<size_t>(dim)] *= max_scale;
      }
    } else {
      value_scale = std::exp(score - max_score);
    }

    for (uint64_t dim = 0; dim < head_dim; ++dim) {
      out[static_cast<size_t>(dim)] +=
          v[offset + static_cast<size_t>(dim)] * value_scale;
    }
    sum = sum * max_scale + value_scale;
  }

  const float inv_sum = sum == 0.0f ? 0.0f : (1.0f / sum);
  for (uint64_t dim = 0; dim < head_dim; ++dim) {
    out[static_cast<size_t>(dim)] *= inv_sum;
  }

  return out;
}

template <class event_type>
inline event_type make_smoke_op_event() {
  event_type ev{};
  static float src0[16] = {
      0.0f, 1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f, 7.0f,
      8.0f, 9.0f, 10.0f, 11.0f,
      12.0f, 13.0f, 14.0f, 15.0f,
  };
  static float src1[16] = {
      15.0f, 14.0f, 13.0f, 12.0f,
      11.0f, 10.0f, 9.0f, 8.0f,
      7.0f, 6.0f, 5.0f, 4.0f,
      3.0f, 2.0f, 1.0f, 0.0f,
  };
  static float src2[16] = {
      1.0f, 1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f, 1.0f,
      1.0f, 1.0f, 1.0f, 1.0f,
  };
  static float dst[16] = {};

  ev.src0 = make_src(src0, dtype::f32, 4);
  ev.src1 = make_src(src1, dtype::f32, 4);
  ev.src2 = make_src(src2, dtype::f32, 4);
  ev.dst = make_dst(dst, dtype::f32, 4);
  ev.ith = 0;
  ev.nth = 1;

  if constexpr (std::is_same_v<event_type, emel::kernel::event::op_mul_mat>) {
    ev.src0 = make_src(src0, dtype::f32, 2, 2);
    ev.src1 = make_src(src1, dtype::f32, 2, 2);
    ev.dst = make_dst(dst, dtype::f32, 2, 2);
  }

  if constexpr (std::is_same_v<event_type, emel::kernel::event::op_soft_max>) {
    ev.src0 = make_src(src0, dtype::f32, 4, 2);
    ev.dst = make_dst(dst, dtype::f32, 4, 2);
  }

  if constexpr (std::is_same_v<event_type, emel::kernel::event::op_unary>) {
    ev.subop = emel::kernel::event::unary_subop::abs;
  }

  return ev;
}

}  // namespace emel::kernel::test
