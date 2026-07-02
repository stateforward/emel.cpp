#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "emel/diarization/sortformer/detail.hpp"
#include "emel/model/data.hpp"

namespace emel::diarization::sortformer::transformer::detail {

inline constexpr int32_t k_layer_count = 18;
inline constexpr int32_t k_layer_tensor_count = 16;
inline constexpr int32_t k_hidden_dim = 192;
inline constexpr int32_t k_inner_dim = 768;
inline constexpr int32_t k_attention_head_count = 8;
inline constexpr int32_t k_attention_head_dim = 24;
inline constexpr int32_t k_max_frame_count = 188;
inline constexpr int32_t k_attention_weight_cache_count = k_layer_count * 4;
inline constexpr int32_t k_feed_forward_weight_cache_count = k_layer_count * 2;
inline constexpr size_t k_hidden_projection_prepared_weight_value_count =
    static_cast<size_t>(k_hidden_dim) * static_cast<size_t>(k_hidden_dim);
inline constexpr size_t k_feed_forward_prepared_weight_value_count =
    static_cast<size_t>(k_inner_dim) * static_cast<size_t>(k_hidden_dim);

struct tensor_view {
  const emel::model::data::tensor_record * tensor = nullptr;
  std::string_view name = {};
};

struct layer_view {
  tensor_view key_bias = {};
  tensor_view key_weight = {};
  tensor_view output_bias = {};
  tensor_view output_weight = {};
  tensor_view query_bias = {};
  tensor_view query_weight = {};
  tensor_view value_bias = {};
  tensor_view value_weight = {};
  tensor_view layer_norm_1_bias = {};
  tensor_view layer_norm_1_weight = {};
  tensor_view layer_norm_2_bias = {};
  tensor_view layer_norm_2_weight = {};
  tensor_view feed_forward_in_bias = {};
  tensor_view feed_forward_in_weight = {};
  tensor_view feed_forward_out_bias = {};
  tensor_view feed_forward_out_weight = {};
};

struct contract {
  std::array<layer_view, k_layer_count> layers = {};
  uint32_t tensor_count = 0;
};

struct layer_workspace {
  layer_workspace();

  std::array<float, k_max_frame_count * k_hidden_dim> query = {};
  std::array<float, k_max_frame_count * k_hidden_dim> key = {};
  std::array<float, k_max_frame_count * k_hidden_dim> value = {};
  std::array<float, k_max_frame_count * k_hidden_dim> first_norm = {};
  std::array<float, k_max_frame_count * k_inner_dim> feed_forward_rows = {};
  std::array<float, k_max_frame_count * k_inner_dim> dense_transposed_input = {};
  std::array<float, k_max_frame_count * k_inner_dim> dense_transposed_output = {};
  std::array<float, k_hidden_dim> frame = {};
  std::array<float, k_hidden_dim> attended = {};
  std::array<float, k_inner_dim> feed_forward = {};
  std::array<float, k_max_frame_count> scores = {};
  std::array<emel::diarization::sortformer::detail::dense_weight_cache,
             k_attention_weight_cache_count>
      attention_weight_caches = {};
  std::array<emel::diarization::sortformer::detail::dense_weight_cache,
             k_feed_forward_weight_cache_count>
      feed_forward_weight_caches = {};
};

bool bind_contract(const emel::model::data & model_data,
                   contract & contract_out) noexcept;

bool prepare_weight_caches(const contract & transformer_contract,
                           layer_workspace & workspace) noexcept;

bool compute_layer_norm(std::span<const float, k_hidden_dim> input,
                        std::span<const float, k_hidden_dim> scale,
                        std::span<const float, k_hidden_dim> bias,
                        std::span<float, k_hidden_dim> output) noexcept;

bool compute_transformer_layer(
    std::span<const float> input_frames,
    uint32_t frame_count,
    std::span<const float, k_hidden_dim * k_hidden_dim> query_weight,
    std::span<const float, k_hidden_dim> query_bias,
    std::span<const float, k_hidden_dim * k_hidden_dim> key_weight,
    std::span<const float, k_hidden_dim> key_bias,
    std::span<const float, k_hidden_dim * k_hidden_dim> value_weight,
    std::span<const float, k_hidden_dim> value_bias,
    std::span<const float, k_hidden_dim * k_hidden_dim> output_weight,
    std::span<const float, k_hidden_dim> output_bias,
    std::span<const float, k_hidden_dim> layer_norm_1_weight,
    std::span<const float, k_hidden_dim> layer_norm_1_bias,
    std::span<const float, k_inner_dim * k_hidden_dim> feed_forward_in_weight,
    std::span<const float, k_inner_dim> feed_forward_in_bias,
    std::span<const float, k_hidden_dim * k_inner_dim> feed_forward_out_weight,
    std::span<const float, k_hidden_dim> feed_forward_out_bias,
    const emel::diarization::sortformer::detail::dense_weight_cache & query_cache,
    const emel::diarization::sortformer::detail::dense_weight_cache & key_cache,
    const emel::diarization::sortformer::detail::dense_weight_cache & value_cache,
    const emel::diarization::sortformer::detail::dense_weight_cache & output_cache,
    const emel::diarization::sortformer::detail::dense_weight_cache & feed_forward_in_cache,
    const emel::diarization::sortformer::detail::dense_weight_cache & feed_forward_out_cache,
    std::span<const float, k_hidden_dim> layer_norm_2_weight,
    std::span<const float, k_hidden_dim> layer_norm_2_bias,
    layer_workspace & workspace,
    std::span<float> output_frames) noexcept;

}  // namespace emel::diarization::sortformer::transformer::detail
