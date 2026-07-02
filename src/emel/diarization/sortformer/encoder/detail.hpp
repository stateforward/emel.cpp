#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "emel/diarization/sortformer/detail.hpp"
#include "emel/diarization/sortformer/encoder/feature_extractor/detail.hpp"
#include "emel/model/data.hpp"

namespace emel::diarization::sortformer::encoder::detail {

inline constexpr int32_t k_pre_tensor_count = 12;
inline constexpr int32_t k_layer_count = 17;
inline constexpr int32_t k_layer_tensor_count = 41;
inline constexpr int32_t k_model_dim = 512;
inline constexpr int32_t k_feed_forward_dim = 2048;
inline constexpr int32_t k_attention_head_count = 8;
inline constexpr int32_t k_attention_head_dim = 64;
inline constexpr int32_t k_depthwise_kernel = 9;
inline constexpr int32_t k_frame_count = 188;
inline constexpr int32_t k_relative_position_count = (k_frame_count * 2) - 1;
inline constexpr int32_t k_feature_bin_count = 128;
inline constexpr int32_t k_feature_frame_count =
    emel::diarization::sortformer::encoder::feature_extractor::detail::k_feature_frame_count;
inline constexpr int32_t k_pre_channel_count = 256;
inline constexpr int32_t k_pre_expand_lanes = 16;
inline constexpr int32_t k_pre_expanded_dim = k_pre_channel_count * k_pre_expand_lanes;
inline constexpr int32_t k_conv0_time_count = 752;
inline constexpr int32_t k_conv0_freq_count = 64;
inline constexpr int32_t k_stage1_time_count = 376;
inline constexpr int32_t k_stage1_freq_count = 32;
inline constexpr int32_t k_stage2_freq_count = 16;
inline constexpr int32_t k_required_feature_value_count =
    k_feature_frame_count * k_feature_bin_count;
inline constexpr int32_t k_required_encoder_value_count = k_frame_count * k_model_dim;
inline constexpr int32_t k_attention_weight_cache_count = k_layer_count * 5;
inline constexpr int32_t k_convolution_weight_cache_count = k_layer_count * 2;
inline constexpr int32_t k_feed_forward_weight_cache_count = k_layer_count * 4;
inline constexpr size_t k_conv0_row_value_count =
    static_cast<size_t>(k_conv0_freq_count) * static_cast<size_t>(k_pre_channel_count);
inline constexpr size_t k_stage1_row_value_count =
    static_cast<size_t>(k_stage1_freq_count) * static_cast<size_t>(k_pre_channel_count);
inline constexpr size_t k_stage2_row_value_count =
    static_cast<size_t>(k_stage2_freq_count) * static_cast<size_t>(k_pre_channel_count);
inline constexpr size_t k_model_projection_prepared_weight_value_count =
    static_cast<size_t>(k_model_dim) * static_cast<size_t>(k_model_dim);
inline constexpr size_t k_convolution_pointwise_1_prepared_weight_value_count =
    static_cast<size_t>(k_model_dim) * static_cast<size_t>(2 * k_model_dim);
inline constexpr size_t k_feed_forward_prepared_weight_value_count =
    static_cast<size_t>(k_model_dim) * static_cast<size_t>(k_feed_forward_dim);
inline constexpr size_t k_pre_output_prepared_weight_value_count =
    static_cast<size_t>(k_pre_expanded_dim) * static_cast<size_t>(k_model_dim);

struct tensor_view {
  const emel::model::data::tensor_record * tensor = nullptr;
  std::string_view name = {};
};

struct contract {
  std::array<tensor_view, k_pre_tensor_count> pre = {};
  std::array<std::array<tensor_view, k_layer_tensor_count>, k_layer_count> layers = {};
  uint32_t tensor_count = 0;
};

struct pre_encoder_workspace {
  pre_encoder_workspace();

  std::array<std::array<float, k_conv0_row_value_count>, 3> conv0_rows = {};
  std::array<std::array<float, k_stage1_row_value_count>, 3> stage1_rows = {};
  std::array<float, k_stage1_row_value_count> stage1_depthwise = {};
  std::array<float, k_stage2_row_value_count> stage2_depthwise = {};
  std::array<float, k_stage2_row_value_count> stage2_row = {};
  std::array<float, k_pre_expanded_dim> flattened = {};
  std::vector<float> pre_encoder_rows = {};
  std::vector<float> layer_input = {};
  std::vector<float> layer_output = {};
  std::vector<float> layer_norm = {};
  std::vector<float> layer_result = {};
  std::vector<float> query = {};
  std::vector<float> key = {};
  std::vector<float> value = {};
  std::vector<float> position = {};
  std::vector<float> position_projection = {};
  std::vector<float> feed_forward_rows = {};
  std::vector<float> dense_transposed_input = {};
  std::vector<float> dense_transposed_output = {};
  emel::diarization::sortformer::detail::dense_weight_cache pre_output_weight_cache = {};
  std::array<emel::diarization::sortformer::detail::dense_weight_cache,
             k_attention_weight_cache_count>
      attention_weight_caches = {};
  std::array<emel::diarization::sortformer::detail::dense_weight_cache,
             k_convolution_weight_cache_count>
      convolution_weight_caches = {};
  std::array<emel::diarization::sortformer::detail::dense_weight_cache,
             k_feed_forward_weight_cache_count>
      feed_forward_weight_caches = {};
  std::array<float, k_model_dim> frame = {};
  std::array<float, k_model_dim> attended = {};
  std::array<float, k_feed_forward_dim> feed_forward = {};
  std::array<float, k_frame_count> scores = {};
  std::array<float, k_model_dim * 2> gated = {};
};

bool bind_contract(const emel::model::data & model_data,
                   contract & contract_out) noexcept;

bool prepare_encoder_weight_caches(const contract & encoder_contract,
                                   pre_encoder_workspace & workspace) noexcept;

bool compute_affine_512(std::span<const float, k_model_dim> input,
                        std::span<const float, k_model_dim> scale,
                        std::span<const float, k_model_dim> bias,
                        std::span<float, k_model_dim> output) noexcept;

bool compute_encoder_frames_from_features(
    std::span<const float> features,
    const contract & encoder_contract,
    pre_encoder_workspace & workspace,
    std::span<float> encoder_frames) noexcept;

template <size_t Size>
std::span<const float, Size> tensor_data(
    const emel::model::data::tensor_record & tensor) noexcept {
  return std::span<const float, Size>{static_cast<const float *>(tensor.data), Size};
}

}  // namespace emel::diarization::sortformer::encoder::detail
