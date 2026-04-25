#include "emel/diarization/sortformer/encoder/detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

#include "emel/diarization/sortformer/detail.hpp"

namespace emel::diarization::sortformer::encoder::detail {

namespace {

struct tensor_spec {
  std::string_view name = {};
  int32_t n_dims = 0;
  std::array<int64_t, 4> dims = {};
};

constexpr std::array<tensor_spec, k_pre_tensor_count> k_pre_specs{{
    {"enc.pre.conv.0.b", 1, {256, 0, 0, 0}},
    {"enc.pre.conv.0.w", 4, {3, 3, 1, 256}},
    {"enc.pre.conv.2.b", 1, {256, 0, 0, 0}},
    {"enc.pre.conv.2.w", 4, {3, 3, 1, 256}},
    {"enc.pre.conv.3.b", 1, {256, 0, 0, 0}},
    {"enc.pre.conv.3.w", 4, {1, 1, 256, 256}},
    {"enc.pre.conv.5.b", 1, {256, 0, 0, 0}},
    {"enc.pre.conv.5.w", 4, {3, 3, 1, 256}},
    {"enc.pre.conv.6.b", 1, {256, 0, 0, 0}},
    {"enc.pre.conv.6.w", 4, {1, 1, 256, 256}},
    {"enc.pre.out.b", 1, {k_model_dim, 0, 0, 0}},
    {"enc.pre.out.w", 2, {4096, k_model_dim, 0, 0}},
}};

constexpr std::array<tensor_spec, k_layer_tensor_count> k_layer_specs{{
    {"conv.bn.b", 1, {k_model_dim, 0, 0, 0}},
    {"conv.bn.rm", 1, {k_model_dim, 0, 0, 0}},
    {"conv.bn.rv", 1, {k_model_dim, 0, 0, 0}},
    {"conv.bn.sc", 1, {k_model_dim, 0, 0, 0}},
    {"conv.bn.sh", 1, {k_model_dim, 0, 0, 0}},
    {"conv.bn.w", 1, {k_model_dim, 0, 0, 0}},
    {"conv.dw.b", 1, {k_model_dim, 0, 0, 0}},
    {"conv.dw.w", 3, {k_depthwise_kernel, 1, k_model_dim, 0}},
    {"conv.pw1.b", 1, {1024, 0, 0, 0}},
    {"conv.pw1.w", 3, {1, k_model_dim, 1024, 0}},
    {"conv.pw2.b", 1, {k_model_dim, 0, 0, 0}},
    {"conv.pw2.w", 3, {1, k_model_dim, k_model_dim, 0}},
    {"ff1.l1.b", 1, {k_feed_forward_dim, 0, 0, 0}},
    {"ff1.l1.w", 2, {k_model_dim, k_feed_forward_dim, 0, 0}},
    {"ff1.l2.b", 1, {k_model_dim, 0, 0, 0}},
    {"ff1.l2.w", 2, {k_feed_forward_dim, k_model_dim, 0, 0}},
    {"ff2.l1.b", 1, {k_feed_forward_dim, 0, 0, 0}},
    {"ff2.l1.w", 2, {k_model_dim, k_feed_forward_dim, 0, 0}},
    {"ff2.l2.b", 1, {k_model_dim, 0, 0, 0}},
    {"ff2.l2.w", 2, {k_feed_forward_dim, k_model_dim, 0, 0}},
    {"nc.b", 1, {k_model_dim, 0, 0, 0}},
    {"nc.w", 1, {k_model_dim, 0, 0, 0}},
    {"nff1.b", 1, {k_model_dim, 0, 0, 0}},
    {"nff1.w", 1, {k_model_dim, 0, 0, 0}},
    {"nff2.b", 1, {k_model_dim, 0, 0, 0}},
    {"nff2.w", 1, {k_model_dim, 0, 0, 0}},
    {"no.b", 1, {k_model_dim, 0, 0, 0}},
    {"no.w", 1, {k_model_dim, 0, 0, 0}},
    {"nsa.b", 1, {k_model_dim, 0, 0, 0}},
    {"nsa.w", 1, {k_model_dim, 0, 0, 0}},
    {"att.k.b", 1, {k_model_dim, 0, 0, 0}},
    {"att.k.w", 2, {k_model_dim, k_model_dim, 0, 0}},
    {"att.o.b", 1, {k_model_dim, 0, 0, 0}},
    {"att.o.w", 2, {k_model_dim, k_model_dim, 0, 0}},
    {"att.p.w", 2, {k_model_dim, k_model_dim, 0, 0}},
    {"att.q.b", 1, {k_model_dim, 0, 0, 0}},
    {"att.q.w", 2, {k_model_dim, k_model_dim, 0, 0}},
    {"att.v.b", 1, {k_model_dim, 0, 0, 0}},
    {"att.v.w", 2, {k_model_dim, k_model_dim, 0, 0}},
    {"att.pbu", 2, {k_attention_head_dim, k_attention_head_count, 0, 0}},
    {"att.pbv", 2, {k_attention_head_dim, k_attention_head_count, 0, 0}},
}};

enum layer_tensor_index : size_t {
  k_conv_bn_bias_index = 0u,
  k_conv_bn_scale_index = 3u,
  k_conv_bn_shift_index = 4u,
  k_conv_depthwise_bias_index = 6u,
  k_conv_depthwise_weight_index = 7u,
  k_conv_pointwise_1_bias_index = 8u,
  k_conv_pointwise_1_weight_index = 9u,
  k_conv_pointwise_2_bias_index = 10u,
  k_conv_pointwise_2_weight_index = 11u,
  k_feed_forward_1_linear_1_bias_index = 12u,
  k_feed_forward_1_linear_1_weight_index = 13u,
  k_feed_forward_1_linear_2_bias_index = 14u,
  k_feed_forward_1_linear_2_weight_index = 15u,
  k_feed_forward_2_linear_1_bias_index = 16u,
  k_feed_forward_2_linear_1_weight_index = 17u,
  k_feed_forward_2_linear_2_bias_index = 18u,
  k_feed_forward_2_linear_2_weight_index = 19u,
  k_norm_conv_bias_index = 20u,
  k_norm_conv_weight_index = 21u,
  k_norm_feed_forward_1_bias_index = 22u,
  k_norm_feed_forward_1_weight_index = 23u,
  k_norm_feed_forward_2_bias_index = 24u,
  k_norm_feed_forward_2_weight_index = 25u,
  k_norm_out_bias_index = 26u,
  k_norm_out_weight_index = 27u,
  k_norm_self_attention_bias_index = 28u,
  k_norm_self_attention_weight_index = 29u,
  k_attention_key_bias_index = 30u,
  k_attention_key_weight_index = 31u,
  k_attention_output_bias_index = 32u,
  k_attention_output_weight_index = 33u,
  k_attention_position_weight_index = 34u,
  k_attention_query_bias_index = 35u,
  k_attention_query_weight_index = 36u,
  k_attention_value_bias_index = 37u,
  k_attention_value_weight_index = 38u,
  k_attention_position_bias_u_index = 39u,
  k_attention_position_bias_v_index = 40u,
};

enum attention_weight_cache_index : size_t {
  k_attention_query_cache_offset = 0u,
  k_attention_key_cache_offset = 1u,
  k_attention_value_cache_offset = 2u,
  k_attention_position_cache_offset = 3u,
  k_attention_output_cache_offset = 4u,
};

enum convolution_weight_cache_index : size_t {
  k_convolution_pointwise_1_cache_offset = 0u,
  k_convolution_pointwise_2_cache_offset = 1u,
};

bool tensor_has_expected_shape(const emel::model::data::tensor_record & tensor,
                               const tensor_spec & spec) noexcept {
  if (tensor.data == nullptr || tensor.data_size == 0u || tensor.n_dims != spec.n_dims) {
    return false;
  }

  for (int32_t dim = 0; dim < spec.n_dims; ++dim) {
    if (tensor.dims[static_cast<size_t>(dim)] != spec.dims[static_cast<size_t>(dim)]) {
      return false;
    }
  }

  return true;
}

const emel::model::data::tensor_record * find_tensor(
    const emel::model::data & model_data,
    const std::string_view name) noexcept {
  for (uint32_t index = 0u; index < model_data.n_tensors; ++index) {
    const auto & tensor = model_data.tensors[index];
    if (emel::model::tensor_name_view(model_data, tensor) == name) {
      return &tensor;
    }
  }

  return nullptr;
}

bool bind_tensor(const emel::model::data & model_data,
                 const tensor_spec & spec,
                 tensor_view & view_out) noexcept {
  const auto * tensor = find_tensor(model_data, spec.name);
  if (tensor == nullptr || !tensor_has_expected_shape(*tensor, spec)) {
    return false;
  }

  view_out.tensor = tensor;
  view_out.name = spec.name;
  return true;
}

bool bind_layer_tensor(const emel::model::data & model_data,
                       const int32_t layer,
                       const tensor_spec & spec,
                       tensor_view & view_out) noexcept {
  std::array<char, 64> name = {};
  const int written = std::snprintf(name.data(),
                                    name.size(),
                                    "enc.l%d.%.*s",
                                    layer,
                                    static_cast<int>(spec.name.size()),
                                    spec.name.data());
  if (written <= 0 || static_cast<size_t>(written) >= name.size()) {
    return false;
  }

  const auto * tensor = find_tensor(model_data, std::string_view{name.data(),
                                                                 static_cast<size_t>(written)});
  if (tensor == nullptr || !tensor_has_expected_shape(*tensor, spec)) {
    return false;
  }

  view_out.tensor = tensor;
  view_out.name = emel::model::tensor_name_view(model_data, *tensor);
  return true;
}

float relu(const float value) noexcept {
  return std::max(value, 0.0f);
}

size_t row_channel_offset(const int32_t freq, const int32_t channel) noexcept {
  return (static_cast<size_t>(freq) * static_cast<size_t>(k_pre_channel_count)) +
      static_cast<size_t>(channel);
}

float feature_value(std::span<const float> features,
                    const int32_t frame,
                    const int32_t bin) noexcept {
  if (frame < 0 || frame >= k_feature_frame_count ||
      bin < 0 || bin >= k_feature_bin_count) {
    return 0.0f;
  }

  const size_t offset =
      (static_cast<size_t>(frame) * static_cast<size_t>(k_feature_bin_count)) +
      static_cast<size_t>(bin);
  return features[offset];
}

void compute_conv0_row(std::span<const float> features,
                       const int32_t time_index,
                       std::span<const float, k_pre_channel_count * 9> weights,
                       std::span<const float, k_pre_channel_count> bias,
                       std::span<float, k_conv0_row_value_count> output) noexcept {
  for (int32_t freq = 0; freq < k_conv0_freq_count; ++freq) {
    const int32_t input_freq_base = (freq * 2) - 1;

    for (int32_t channel = 0; channel < k_pre_channel_count; ++channel) {
      const size_t weight_base = static_cast<size_t>(channel) * 9u;
      float acc = bias[static_cast<size_t>(channel)];

      for (int32_t time_tap = 0; time_tap < 3; ++time_tap) {
        const int32_t input_time = (time_index * 2) + time_tap - 1;
        for (int32_t freq_tap = 0; freq_tap < 3; ++freq_tap) {
          const size_t weight_offset =
              weight_base + static_cast<size_t>((time_tap * 3) + freq_tap);
          acc += weights[weight_offset] *
              feature_value(features, input_time, input_freq_base + freq_tap);
        }
      }

      output[row_channel_offset(freq, channel)] = relu(acc);
    }
  }
}

void compute_depthwise_row(
    const std::array<const float *, 3> & input_rows,
    const int32_t input_freq_count,
    const int32_t output_freq_count,
    std::span<const float, k_pre_channel_count * 9> weights,
    std::span<const float, k_pre_channel_count> bias,
    std::span<float> output) noexcept {
  for (int32_t freq = 0; freq < output_freq_count; ++freq) {
    const int32_t input_freq_base = (freq * 2) - 1;

    for (int32_t channel = 0; channel < k_pre_channel_count; ++channel) {
      const size_t weight_base = static_cast<size_t>(channel) * 9u;
      float acc = bias[static_cast<size_t>(channel)];

      for (int32_t time_tap = 0; time_tap < 3; ++time_tap) {
        const float * row = input_rows[static_cast<size_t>(time_tap)];
        if (row == nullptr) {
          continue;
        }

        for (int32_t freq_tap = 0; freq_tap < 3; ++freq_tap) {
          const int32_t input_freq = input_freq_base + freq_tap;
          if (input_freq < 0 || input_freq >= input_freq_count) {
            continue;
          }

          const size_t weight_offset =
              weight_base + static_cast<size_t>((time_tap * 3) + freq_tap);
          acc += weights[weight_offset] *
              row[row_channel_offset(input_freq, channel)];
        }
      }

      output[row_channel_offset(freq, channel)] = acc;
    }
  }
}

bool compute_pointwise_row(std::span<const float> input,
                           const int32_t freq_count,
                           std::span<const float, k_pre_channel_count * k_pre_channel_count>
                               weights,
                           std::span<const float, k_pre_channel_count> bias,
                           pre_encoder_workspace & workspace,
                           std::span<float> output) noexcept {
  if (!emel::diarization::sortformer::detail::compute_dense_batch(
          input,
          static_cast<size_t>(freq_count),
          static_cast<size_t>(k_pre_channel_count),
          weights,
          bias,
          static_cast<size_t>(k_pre_channel_count),
          workspace.dense_transposed_input,
          workspace.dense_transposed_output,
          output)) {
    return false;
  }

  for (float & value : output) {
    value = relu(value);
  }

  return true;
}

void flatten_stage2_row(std::span<const float, k_stage2_row_value_count> input,
                        std::span<float, k_pre_expanded_dim> output) noexcept {
  for (int32_t channel = 0; channel < k_pre_channel_count; ++channel) {
    const size_t output_base = static_cast<size_t>(channel) *
        static_cast<size_t>(k_pre_expand_lanes);

    for (int32_t freq = 0; freq < k_stage2_freq_count; ++freq) {
      output[output_base + static_cast<size_t>(freq)] =
          input[row_channel_offset(freq, channel)];
    }
  }
}

const float * row_or_null(const pre_encoder_workspace & workspace,
                          const int32_t index,
                          const int32_t upper_bound,
                          const bool use_conv0_rows) noexcept {
  if (index < 0 || index >= upper_bound) {
    return nullptr;
  }

  const size_t ring_index = static_cast<size_t>(index % 3);
  if (use_conv0_rows) {
    return workspace.conv0_rows[ring_index].data();
  }
  return workspace.stage1_rows[ring_index].data();
}

float sigmoid(const float value) noexcept {
  return 1.0f / (1.0f + std::exp(-value));
}

template <size_t Size>
std::span<float, Size> fixed_span(std::vector<float> & values) noexcept {
  return std::span<float, Size>{values.data(), Size};
}

template <size_t Size>
std::span<const float, Size> fixed_span(const std::vector<float> & values) noexcept {
  return std::span<const float, Size>{values.data(), Size};
}

float silu(const float value) noexcept {
  return value * sigmoid(value);
}

void compute_sinusoidal_relative_positions(
    std::span<float, k_relative_position_count * k_model_dim> output) noexcept {
  static constexpr float k_log_timescale = 9.210340371976184f;
  for (int32_t relative_index = 0; relative_index < k_relative_position_count;
       ++relative_index) {
    const int32_t position = (k_frame_count - 1) - relative_index;
    const size_t base =
        static_cast<size_t>(relative_index) * static_cast<size_t>(k_model_dim);

    for (int32_t dim = 0; dim < k_model_dim; dim += 2) {
      const float scale =
          std::exp((static_cast<float>(dim) * -k_log_timescale) /
                   static_cast<float>(k_model_dim));
      const float angle = static_cast<float>(position) * scale;
      output[base + static_cast<size_t>(dim)] = std::sin(angle);
      output[base + static_cast<size_t>(dim + 1)] = std::cos(angle);
    }
  }
}

bool compute_layer_norm_512(std::span<const float, k_model_dim> input,
                            std::span<const float, k_model_dim> scale,
                            std::span<const float, k_model_dim> bias,
                            std::span<float, k_model_dim> output) noexcept {
  float mean = 0.0f;
  for (const float value : input) {
    mean += value;
  }
  mean /= static_cast<float>(k_model_dim);

  float variance = 0.0f;
  for (const float value : input) {
    const float centered = value - mean;
    variance += centered * centered;
  }
  variance /= static_cast<float>(k_model_dim);

  const float inv_std = 1.0f / std::sqrt(variance + 1.0e-5f);
  for (size_t index = 0u; index < static_cast<size_t>(k_model_dim); ++index) {
    output[index] = ((input[index] - mean) * inv_std * scale[index]) + bias[index];
  }

  return true;
}

bool compute_position_projection(
    std::span<const float, k_relative_position_count * k_model_dim> positions,
    std::span<const float, k_model_dim * k_model_dim> weights,
    const emel::diarization::sortformer::detail::dense_weight_cache & cache,
    pre_encoder_workspace & workspace,
    std::span<float, k_relative_position_count * k_model_dim> output) noexcept {
  return emel::diarization::sortformer::detail::compute_dense_batch_without_bias_prepared(
      positions,
      static_cast<size_t>(k_relative_position_count),
      static_cast<size_t>(k_model_dim),
      weights,
      cache,
      static_cast<size_t>(k_model_dim),
      workspace.dense_transposed_input,
      workspace.dense_transposed_output,
      output);
}

bool compute_feed_forward_block(
    std::span<const float, k_required_encoder_value_count> input_frames,
    std::span<const float, k_model_dim> norm_weight,
    std::span<const float, k_model_dim> norm_bias,
    std::span<const float, k_model_dim * k_feed_forward_dim> linear_1_weight,
    std::span<const float, k_feed_forward_dim> linear_1_bias,
    std::span<const float, k_feed_forward_dim * k_model_dim> linear_2_weight,
    std::span<const float, k_model_dim> linear_2_bias,
    const emel::diarization::sortformer::detail::dense_weight_cache & linear_1_cache,
    const emel::diarization::sortformer::detail::dense_weight_cache & linear_2_cache,
    pre_encoder_workspace & workspace,
    std::span<float, k_required_encoder_value_count> output_frames) noexcept {
  for (int32_t frame = 0; frame < k_frame_count; ++frame) {
    const size_t frame_base = static_cast<size_t>(frame) * static_cast<size_t>(k_model_dim);
    const auto input = std::span<const float, k_model_dim>{input_frames.data() + frame_base,
                                                           k_model_dim};
    auto normalized = std::span<float, k_model_dim>{workspace.layer_norm.data() + frame_base,
                                                    k_model_dim};
    if (!compute_layer_norm_512(input, norm_weight, norm_bias, normalized)) {
      return false;
    }
  }

  if (!emel::diarization::sortformer::detail::compute_dense_batch_to_transposed_prepared(
          fixed_span<k_required_encoder_value_count>(workspace.layer_norm),
          static_cast<size_t>(k_frame_count),
          static_cast<size_t>(k_model_dim),
          linear_1_weight,
          linear_1_cache,
          linear_1_bias,
          static_cast<size_t>(k_feed_forward_dim),
          workspace.dense_transposed_input,
          workspace.dense_transposed_output)) {
    return false;
  }

  const size_t feed_forward_value_count =
      static_cast<size_t>(k_frame_count) * static_cast<size_t>(k_feed_forward_dim);
  for (float & value : std::span<float>{workspace.dense_transposed_output.data(),
                                        feed_forward_value_count}) {
    value = silu(value);
  }

  if (!emel::diarization::sortformer::detail::
          compute_dense_batch_from_transposed_scaled_residual_prepared(
          std::span<const float>{workspace.dense_transposed_output.data(),
                                 feed_forward_value_count},
          static_cast<size_t>(k_frame_count),
          static_cast<size_t>(k_feed_forward_dim),
          linear_2_weight,
          linear_2_cache,
          linear_2_bias,
          static_cast<size_t>(k_model_dim),
          0.5f,
          input_frames,
          workspace.dense_transposed_input,
          output_frames)) {
    return false;
  }

  return true;
}

float compute_attention_content_score(
    std::span<const float, k_required_encoder_value_count> query,
    std::span<const float, k_required_encoder_value_count> key,
    std::span<const float, k_attention_head_dim * k_attention_head_count> position_bias,
    const int32_t query_frame,
    const int32_t key_frame,
    const int32_t head) noexcept {
  const size_t head_offset = static_cast<size_t>(head) *
      static_cast<size_t>(k_attention_head_dim);
  const size_t query_base =
      (static_cast<size_t>(query_frame) * static_cast<size_t>(k_model_dim)) + head_offset;
  const size_t key_base =
      (static_cast<size_t>(key_frame) * static_cast<size_t>(k_model_dim)) + head_offset;
  const size_t bias_base = head_offset;

  float acc = 0.0f;
  for (int32_t dim = 0; dim < k_attention_head_dim; ++dim) {
    const size_t offset = static_cast<size_t>(dim);
    acc += (query[query_base + offset] + position_bias[bias_base + offset]) *
        key[key_base + offset];
  }
  return acc;
}

float compute_attention_position_score(
    std::span<const float, k_required_encoder_value_count> query,
    std::span<const float, k_relative_position_count * k_model_dim> positions,
    std::span<const float, k_attention_head_dim * k_attention_head_count> position_bias,
    const int32_t query_frame,
    const int32_t key_frame,
    const int32_t head) noexcept {
  const int32_t relative_index = (k_frame_count - 1) + key_frame - query_frame;
  const size_t head_offset = static_cast<size_t>(head) *
      static_cast<size_t>(k_attention_head_dim);
  const size_t query_base =
      (static_cast<size_t>(query_frame) * static_cast<size_t>(k_model_dim)) + head_offset;
  const size_t position_base =
      (static_cast<size_t>(relative_index) * static_cast<size_t>(k_model_dim)) + head_offset;
  const size_t bias_base = head_offset;

  float acc = 0.0f;
  for (int32_t dim = 0; dim < k_attention_head_dim; ++dim) {
    const size_t offset = static_cast<size_t>(dim);
    acc += (query[query_base + offset] + position_bias[bias_base + offset]) *
        positions[position_base + offset];
  }
  return acc;
}

void compute_attention_head(
    std::span<const float, k_required_encoder_value_count> query,
    std::span<const float, k_required_encoder_value_count> key,
    std::span<const float, k_required_encoder_value_count> value,
    std::span<const float, k_relative_position_count * k_model_dim> positions,
    std::span<const float, k_attention_head_dim * k_attention_head_count> position_bias_u,
    std::span<const float, k_attention_head_dim * k_attention_head_count> position_bias_v,
    const int32_t query_frame,
    const int32_t head,
    std::span<float, k_frame_count> scores,
    std::span<float, k_model_dim> attended) noexcept {
  float max_score = -std::numeric_limits<float>::infinity();
  for (int32_t key_frame = 0; key_frame < k_frame_count; ++key_frame) {
    const float content = compute_attention_content_score(
        query, key, position_bias_u, query_frame, key_frame, head);
    const float position = compute_attention_position_score(
        query, positions, position_bias_v, query_frame, key_frame, head);
    const float score = (content + position) /
        std::sqrt(static_cast<float>(k_attention_head_dim));
    scores[static_cast<size_t>(key_frame)] = score;
    max_score = std::max(max_score, score);
  }

  float normalizer = 0.0f;
  for (int32_t key_frame = 0; key_frame < k_frame_count; ++key_frame) {
    const float weight = std::exp(scores[static_cast<size_t>(key_frame)] - max_score);
    scores[static_cast<size_t>(key_frame)] = weight;
    normalizer += weight;
  }

  const size_t head_offset = static_cast<size_t>(head) *
      static_cast<size_t>(k_attention_head_dim);
  const float inv_normalizer = 1.0f / normalizer;
  for (int32_t key_frame = 0; key_frame < k_frame_count; ++key_frame) {
    scores[static_cast<size_t>(key_frame)] *= inv_normalizer;
  }
  for (int32_t dim = 0; dim < k_attention_head_dim; ++dim) {
    float acc = 0.0f;
    for (int32_t key_frame = 0; key_frame < k_frame_count; ++key_frame) {
      const size_t value_offset =
          (static_cast<size_t>(key_frame) * static_cast<size_t>(k_model_dim)) +
          head_offset + static_cast<size_t>(dim);
      acc += scores[static_cast<size_t>(key_frame)] * value[value_offset];
    }
    attended[head_offset + static_cast<size_t>(dim)] = acc;
  }
}

bool compute_attention_block(
    std::span<const float, k_required_encoder_value_count> input_frames,
    std::span<const float, k_model_dim> norm_weight,
    std::span<const float, k_model_dim> norm_bias,
    std::span<const float, k_model_dim * k_model_dim> query_weight,
    std::span<const float, k_model_dim> query_bias,
    std::span<const float, k_model_dim * k_model_dim> key_weight,
    std::span<const float, k_model_dim> key_bias,
    std::span<const float, k_model_dim * k_model_dim> value_weight,
    std::span<const float, k_model_dim> value_bias,
    std::span<const float, k_model_dim * k_model_dim> position_weight,
    std::span<const float, k_attention_head_dim * k_attention_head_count> position_bias_u,
    std::span<const float, k_attention_head_dim * k_attention_head_count> position_bias_v,
    std::span<const float, k_model_dim * k_model_dim> output_weight,
    std::span<const float, k_model_dim> output_bias,
    const emel::diarization::sortformer::detail::dense_weight_cache & query_cache,
    const emel::diarization::sortformer::detail::dense_weight_cache & key_cache,
    const emel::diarization::sortformer::detail::dense_weight_cache & value_cache,
    const emel::diarization::sortformer::detail::dense_weight_cache & position_cache,
    const emel::diarization::sortformer::detail::dense_weight_cache & output_cache,
    pre_encoder_workspace & workspace,
    std::span<float, k_required_encoder_value_count> output_frames) noexcept {
  for (int32_t frame = 0; frame < k_frame_count; ++frame) {
    const size_t frame_base = static_cast<size_t>(frame) * static_cast<size_t>(k_model_dim);
    const auto input = std::span<const float, k_model_dim>{input_frames.data() + frame_base,
                                                           k_model_dim};
    auto normalized = std::span<float, k_model_dim>{workspace.layer_norm.data() + frame_base,
                                                    k_model_dim};
    if (!compute_layer_norm_512(input, norm_weight, norm_bias, normalized)) {
      return false;
    }
  }

  const auto qkv_input = fixed_span<k_required_encoder_value_count>(workspace.layer_norm);
  auto qkv_transposed = std::span<float>{
      workspace.dense_transposed_input.data(),
      static_cast<size_t>(k_frame_count) * static_cast<size_t>(k_model_dim)};
  if (!emel::diarization::sortformer::detail::transpose_dense_input(
          qkv_input,
          static_cast<size_t>(k_frame_count),
          static_cast<size_t>(k_model_dim),
          qkv_transposed) ||
      !emel::diarization::sortformer::detail::compute_dense_batch_from_transposed_prepared(
          qkv_transposed,
          static_cast<size_t>(k_frame_count),
          static_cast<size_t>(k_model_dim),
          query_weight,
          query_cache,
          query_bias,
          static_cast<size_t>(k_model_dim),
          workspace.dense_transposed_output,
          fixed_span<k_required_encoder_value_count>(workspace.query)) ||
      !emel::diarization::sortformer::detail::compute_dense_batch_from_transposed_prepared(
          qkv_transposed,
          static_cast<size_t>(k_frame_count),
          static_cast<size_t>(k_model_dim),
          key_weight,
          key_cache,
          key_bias,
          static_cast<size_t>(k_model_dim),
          workspace.dense_transposed_output,
          fixed_span<k_required_encoder_value_count>(workspace.key)) ||
      !emel::diarization::sortformer::detail::compute_dense_batch_from_transposed_prepared(
          qkv_transposed,
          static_cast<size_t>(k_frame_count),
          static_cast<size_t>(k_model_dim),
          value_weight,
          value_cache,
          value_bias,
          static_cast<size_t>(k_model_dim),
          workspace.dense_transposed_output,
          fixed_span<k_required_encoder_value_count>(workspace.value)) ||
      !compute_position_projection(fixed_span<k_relative_position_count * k_model_dim>(
                                       workspace.position),
                                   position_weight,
                                   position_cache,
                                   workspace,
                                   fixed_span<k_relative_position_count * k_model_dim>(
                                       workspace.position_projection))) {
    return false;
  }

  for (int32_t frame = 0; frame < k_frame_count; ++frame) {
    workspace.attended.fill(0.0f);
    for (int32_t head = 0; head < k_attention_head_count; ++head) {
      compute_attention_head(fixed_span<k_required_encoder_value_count>(workspace.query),
                             fixed_span<k_required_encoder_value_count>(workspace.key),
                             fixed_span<k_required_encoder_value_count>(workspace.value),
                             fixed_span<k_relative_position_count * k_model_dim>(
                                 workspace.position_projection),
                             position_bias_u,
                             position_bias_v,
                             frame,
                             head,
                             workspace.scores,
                             workspace.attended);
    }

    const size_t frame_base = static_cast<size_t>(frame) * static_cast<size_t>(k_model_dim);
    for (size_t dim = 0u; dim < static_cast<size_t>(k_model_dim); ++dim) {
      workspace.layer_result[frame_base + dim] = workspace.attended[dim];
    }
  }

  if (!emel::diarization::sortformer::detail::compute_dense_batch_residual_prepared(
          fixed_span<k_required_encoder_value_count>(workspace.layer_result),
          static_cast<size_t>(k_frame_count),
          static_cast<size_t>(k_model_dim),
          output_weight,
          output_cache,
          output_bias,
          static_cast<size_t>(k_model_dim),
          input_frames,
          workspace.dense_transposed_input,
          workspace.dense_transposed_output,
          output_frames)) {
    return false;
  }

  return true;
}

bool compute_convolution_block(
    std::span<const float, k_required_encoder_value_count> input_frames,
    std::span<const float, k_model_dim> norm_weight,
    std::span<const float, k_model_dim> norm_bias,
    std::span<const float, k_model_dim * 2 * k_model_dim> pointwise_1_weight,
    std::span<const float, k_model_dim * 2> pointwise_1_bias,
    std::span<const float, k_model_dim * k_depthwise_kernel> depthwise_weight,
    std::span<const float, k_model_dim> depthwise_bias,
    std::span<const float, k_model_dim> batch_norm_scale,
    std::span<const float, k_model_dim> batch_norm_shift,
    std::span<const float, k_model_dim * k_model_dim> pointwise_2_weight,
    std::span<const float, k_model_dim> pointwise_2_bias,
    const emel::diarization::sortformer::detail::dense_weight_cache & pointwise_1_cache,
    const emel::diarization::sortformer::detail::dense_weight_cache & pointwise_2_cache,
    pre_encoder_workspace & workspace,
    std::span<float, k_required_encoder_value_count> output_frames) noexcept {
  for (int32_t frame = 0; frame < k_frame_count; ++frame) {
    const size_t frame_base = static_cast<size_t>(frame) * static_cast<size_t>(k_model_dim);
    const auto input = std::span<const float, k_model_dim>{input_frames.data() + frame_base,
                                                           k_model_dim};
    auto normalized = std::span<float, k_model_dim>{workspace.layer_norm.data() + frame_base,
                                                    k_model_dim};
    if (!compute_layer_norm_512(input, norm_weight, norm_bias, normalized)) {
      return false;
    }
  }

  auto gated_rows = std::span<float>{
      workspace.feed_forward_rows.data(),
      static_cast<size_t>(k_frame_count * 2 * k_model_dim),
  };
  if (!emel::diarization::sortformer::detail::compute_dense_batch_prepared(
          fixed_span<k_required_encoder_value_count>(workspace.layer_norm),
          static_cast<size_t>(k_frame_count),
          static_cast<size_t>(k_model_dim),
          pointwise_1_weight,
          pointwise_1_cache,
          pointwise_1_bias,
          static_cast<size_t>(2 * k_model_dim),
          workspace.dense_transposed_input,
          workspace.dense_transposed_output,
          gated_rows)) {
    return false;
  }

  for (int32_t frame = 0; frame < k_frame_count; ++frame) {
    const size_t frame_base = static_cast<size_t>(frame) * static_cast<size_t>(k_model_dim);
    const size_t gated_base = static_cast<size_t>(frame) *
        static_cast<size_t>(2 * k_model_dim);
    for (int32_t channel = 0; channel < k_model_dim; ++channel) {
      const size_t offset = static_cast<size_t>(channel);
      workspace.layer_norm[frame_base + offset] =
          gated_rows[gated_base + offset] *
          sigmoid(gated_rows[gated_base + static_cast<size_t>(k_model_dim) + offset]);
    }
  }

  for (int32_t frame = 0; frame < k_frame_count; ++frame) {
    const size_t frame_base = static_cast<size_t>(frame) * static_cast<size_t>(k_model_dim);
    for (int32_t channel = 0; channel < k_model_dim; ++channel) {
      const size_t channel_offset = static_cast<size_t>(channel);
      float acc = depthwise_bias[channel_offset];
      for (int32_t tap = 0; tap < k_depthwise_kernel; ++tap) {
        const int32_t input_frame = frame + tap - (k_depthwise_kernel / 2);
        if (input_frame < 0 || input_frame >= k_frame_count) {
          continue;
        }
        const size_t input_offset =
            (static_cast<size_t>(input_frame) * static_cast<size_t>(k_model_dim)) +
            channel_offset;
        const size_t weight_offset =
            (channel_offset * static_cast<size_t>(k_depthwise_kernel)) +
            static_cast<size_t>(tap);
        acc += workspace.layer_norm[input_offset] * depthwise_weight[weight_offset];
      }
      workspace.layer_output[frame_base + channel_offset] =
          silu((acc * batch_norm_scale[channel_offset]) + batch_norm_shift[channel_offset]);
    }
  }

  if (!emel::diarization::sortformer::detail::compute_dense_batch_residual_prepared(
          fixed_span<k_required_encoder_value_count>(workspace.layer_output),
          static_cast<size_t>(k_frame_count),
          static_cast<size_t>(k_model_dim),
          pointwise_2_weight,
          pointwise_2_cache,
          pointwise_2_bias,
          static_cast<size_t>(k_model_dim),
          input_frames,
          workspace.dense_transposed_input,
          workspace.dense_transposed_output,
          output_frames)) {
    return false;
  }

  return true;
}

bool compute_encoder_layer(
    const int32_t layer_index,
    std::span<const float, k_required_encoder_value_count> input_frames,
    const std::array<tensor_view, k_layer_tensor_count> & layer,
    pre_encoder_workspace & workspace,
    std::span<float, k_required_encoder_value_count> output_frames) noexcept {
  const size_t attention_cache_base = static_cast<size_t>(layer_index) * 5u;
  const size_t convolution_cache_base = static_cast<size_t>(layer_index) * 2u;
  const size_t cache_base = static_cast<size_t>(layer_index) * 4u;
  if (!compute_feed_forward_block(
          input_frames,
          tensor_data<k_model_dim>(*layer[k_norm_feed_forward_1_weight_index].tensor),
          tensor_data<k_model_dim>(*layer[k_norm_feed_forward_1_bias_index].tensor),
          tensor_data<k_model_dim * k_feed_forward_dim>(
              *layer[k_feed_forward_1_linear_1_weight_index].tensor),
          tensor_data<k_feed_forward_dim>(*layer[k_feed_forward_1_linear_1_bias_index].tensor),
          tensor_data<k_feed_forward_dim * k_model_dim>(
              *layer[k_feed_forward_1_linear_2_weight_index].tensor),
          tensor_data<k_model_dim>(*layer[k_feed_forward_1_linear_2_bias_index].tensor),
          workspace.feed_forward_weight_caches[cache_base + 0u],
          workspace.feed_forward_weight_caches[cache_base + 1u],
          workspace,
          fixed_span<k_required_encoder_value_count>(workspace.layer_output))) {
    return false;
  }

  if (!compute_attention_block(
          fixed_span<k_required_encoder_value_count>(workspace.layer_output),
          tensor_data<k_model_dim>(*layer[k_norm_self_attention_weight_index].tensor),
          tensor_data<k_model_dim>(*layer[k_norm_self_attention_bias_index].tensor),
          tensor_data<k_model_dim * k_model_dim>(*layer[k_attention_query_weight_index].tensor),
          tensor_data<k_model_dim>(*layer[k_attention_query_bias_index].tensor),
          tensor_data<k_model_dim * k_model_dim>(*layer[k_attention_key_weight_index].tensor),
          tensor_data<k_model_dim>(*layer[k_attention_key_bias_index].tensor),
          tensor_data<k_model_dim * k_model_dim>(*layer[k_attention_value_weight_index].tensor),
          tensor_data<k_model_dim>(*layer[k_attention_value_bias_index].tensor),
          tensor_data<k_model_dim * k_model_dim>(
              *layer[k_attention_position_weight_index].tensor),
          tensor_data<k_attention_head_dim * k_attention_head_count>(
              *layer[k_attention_position_bias_u_index].tensor),
          tensor_data<k_attention_head_dim * k_attention_head_count>(
              *layer[k_attention_position_bias_v_index].tensor),
          tensor_data<k_model_dim * k_model_dim>(*layer[k_attention_output_weight_index].tensor),
          tensor_data<k_model_dim>(*layer[k_attention_output_bias_index].tensor),
          workspace.attention_weight_caches[attention_cache_base +
                                            k_attention_query_cache_offset],
          workspace.attention_weight_caches[attention_cache_base +
                                            k_attention_key_cache_offset],
          workspace.attention_weight_caches[attention_cache_base +
                                            k_attention_value_cache_offset],
          workspace.attention_weight_caches[attention_cache_base +
                                            k_attention_position_cache_offset],
          workspace.attention_weight_caches[attention_cache_base +
                                            k_attention_output_cache_offset],
          workspace,
          fixed_span<k_required_encoder_value_count>(workspace.layer_input))) {
    return false;
  }

  if (!compute_convolution_block(
          fixed_span<k_required_encoder_value_count>(workspace.layer_input),
          tensor_data<k_model_dim>(*layer[k_norm_conv_weight_index].tensor),
          tensor_data<k_model_dim>(*layer[k_norm_conv_bias_index].tensor),
          tensor_data<k_model_dim * 2 * k_model_dim>(
              *layer[k_conv_pointwise_1_weight_index].tensor),
          tensor_data<k_model_dim * 2>(*layer[k_conv_pointwise_1_bias_index].tensor),
          tensor_data<k_model_dim * k_depthwise_kernel>(
              *layer[k_conv_depthwise_weight_index].tensor),
          tensor_data<k_model_dim>(*layer[k_conv_depthwise_bias_index].tensor),
          tensor_data<k_model_dim>(*layer[k_conv_bn_scale_index].tensor),
          tensor_data<k_model_dim>(*layer[k_conv_bn_shift_index].tensor),
          tensor_data<k_model_dim * k_model_dim>(*layer[k_conv_pointwise_2_weight_index].tensor),
          tensor_data<k_model_dim>(*layer[k_conv_pointwise_2_bias_index].tensor),
          workspace.convolution_weight_caches[convolution_cache_base +
                                              k_convolution_pointwise_1_cache_offset],
          workspace.convolution_weight_caches[convolution_cache_base +
                                              k_convolution_pointwise_2_cache_offset],
          workspace,
          fixed_span<k_required_encoder_value_count>(workspace.layer_output))) {
    return false;
  }

  if (!compute_feed_forward_block(
          fixed_span<k_required_encoder_value_count>(workspace.layer_output),
          tensor_data<k_model_dim>(*layer[k_norm_feed_forward_2_weight_index].tensor),
          tensor_data<k_model_dim>(*layer[k_norm_feed_forward_2_bias_index].tensor),
          tensor_data<k_model_dim * k_feed_forward_dim>(
              *layer[k_feed_forward_2_linear_1_weight_index].tensor),
          tensor_data<k_feed_forward_dim>(*layer[k_feed_forward_2_linear_1_bias_index].tensor),
          tensor_data<k_feed_forward_dim * k_model_dim>(
              *layer[k_feed_forward_2_linear_2_weight_index].tensor),
          tensor_data<k_model_dim>(*layer[k_feed_forward_2_linear_2_bias_index].tensor),
          workspace.feed_forward_weight_caches[cache_base + 2u],
          workspace.feed_forward_weight_caches[cache_base + 3u],
          workspace,
          fixed_span<k_required_encoder_value_count>(workspace.layer_input))) {
    return false;
  }

  for (int32_t frame = 0; frame < k_frame_count; ++frame) {
    const size_t frame_base = static_cast<size_t>(frame) * static_cast<size_t>(k_model_dim);
    const auto input = std::span<const float, k_model_dim>{
        workspace.layer_input.data() + frame_base, k_model_dim};
    auto output = std::span<float, k_model_dim>{output_frames.data() + frame_base,
                                                k_model_dim};
    if (!compute_layer_norm_512(input,
                                tensor_data<k_model_dim>(
                                    *layer[k_norm_out_weight_index].tensor),
                                tensor_data<k_model_dim>(
                                    *layer[k_norm_out_bias_index].tensor),
                                output)) {
      return false;
    }
  }

  return true;
}

bool compute_encoder_stack(const contract & encoder_contract,
                           pre_encoder_workspace & workspace,
                           std::span<float, k_required_encoder_value_count> encoder_frames)
    noexcept {
  static constexpr float k_xscale = 22.627416610717773f;
  for (size_t index = 0u; index < static_cast<size_t>(k_required_encoder_value_count);
       ++index) {
    workspace.layer_input[index] = encoder_frames[index] * k_xscale;
  }

  for (int32_t layer_index = 0; layer_index < k_layer_count; ++layer_index) {
    if (!compute_encoder_layer(layer_index,
                               fixed_span<k_required_encoder_value_count>(
                                   workspace.layer_input),
                               encoder_contract.layers[static_cast<size_t>(layer_index)],
                               workspace,
                               fixed_span<k_required_encoder_value_count>(
                                   workspace.layer_input))) {
      return false;
    }
  }

  std::copy(workspace.layer_input.begin(), workspace.layer_input.end(), encoder_frames.begin());
  return true;
}

bool pre_contract_bound(const contract & encoder_contract) noexcept {
  for (const auto & tensor : encoder_contract.pre) {
    if (tensor.tensor == nullptr) {
      return false;
    }
  }
  for (const auto & layer : encoder_contract.layers) {
    for (const auto & tensor : layer) {
      if (tensor.tensor == nullptr) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

pre_encoder_workspace::pre_encoder_workspace()
    : pre_encoder_rows(static_cast<size_t>(k_frame_count * k_pre_expanded_dim), 0.0f),
      layer_input(static_cast<size_t>(k_required_encoder_value_count), 0.0f),
      layer_output(static_cast<size_t>(k_required_encoder_value_count), 0.0f),
      layer_norm(static_cast<size_t>(k_required_encoder_value_count), 0.0f),
      layer_result(static_cast<size_t>(k_required_encoder_value_count), 0.0f),
      query(static_cast<size_t>(k_required_encoder_value_count), 0.0f),
      key(static_cast<size_t>(k_required_encoder_value_count), 0.0f),
      value(static_cast<size_t>(k_required_encoder_value_count), 0.0f),
      position(static_cast<size_t>(k_relative_position_count * k_model_dim), 0.0f),
      position_projection(static_cast<size_t>(k_relative_position_count * k_model_dim),
                          0.0f),
      feed_forward_rows(static_cast<size_t>(k_frame_count * k_feed_forward_dim), 0.0f),
      dense_transposed_input(static_cast<size_t>(k_frame_count * k_pre_expanded_dim), 0.0f),
      dense_transposed_output(static_cast<size_t>(k_frame_count * k_feed_forward_dim), 0.0f) {
  // Large full-encoder scratch buffers are allocated once with the actor workspace
  // so dispatch remains allocation-free and test binaries do not embed multi-MB contexts.
  pre_output_weight_cache.lhs_4row.resize(k_pre_output_prepared_weight_value_count);
  for (auto & cache : attention_weight_caches) {
    cache.lhs_4row.resize(k_model_projection_prepared_weight_value_count);
  }
  for (int32_t layer_index = 0; layer_index < k_layer_count; ++layer_index) {
    const size_t cache_base = static_cast<size_t>(layer_index) * 2u;
    convolution_weight_caches[cache_base + k_convolution_pointwise_1_cache_offset]
        .lhs_4row.resize(k_convolution_pointwise_1_prepared_weight_value_count);
    convolution_weight_caches[cache_base + k_convolution_pointwise_2_cache_offset]
        .lhs_4row.resize(k_model_projection_prepared_weight_value_count);
  }
  for (auto & cache : feed_forward_weight_caches) {
    cache.lhs_4row.resize(k_feed_forward_prepared_weight_value_count);
  }
  compute_sinusoidal_relative_positions(
      fixed_span<k_relative_position_count * k_model_dim>(position));
}

bool bind_contract(const emel::model::data & model_data,
                   contract & contract_out) noexcept {
  contract next = {};

  for (size_t index = 0u; index < k_pre_specs.size(); ++index) {
    if (!bind_tensor(model_data, k_pre_specs[index], next.pre[index])) {
      return false;
    }
    ++next.tensor_count;
  }

  for (int32_t layer = 0; layer < k_layer_count; ++layer) {
    for (size_t index = 0u; index < k_layer_specs.size(); ++index) {
      if (!bind_layer_tensor(model_data,
                             layer,
                             k_layer_specs[index],
                             next.layers[static_cast<size_t>(layer)][index])) {
        return false;
      }
      ++next.tensor_count;
    }
  }

  contract_out = next;
  return true;
}

bool prepare_encoder_weight_caches(const contract & encoder_contract,
                                   pre_encoder_workspace & workspace) noexcept {
  if (!emel::diarization::sortformer::detail::prepare_dense_weight_cache(
          tensor_data<k_model_dim * k_pre_expanded_dim>(*encoder_contract.pre[11].tensor),
          static_cast<size_t>(k_pre_expanded_dim),
          static_cast<size_t>(k_model_dim),
          workspace.pre_output_weight_cache)) {
    return false;
  }

  for (int32_t layer_index = 0; layer_index < k_layer_count; ++layer_index) {
    const auto & layer = encoder_contract.layers[static_cast<size_t>(layer_index)];
    const size_t attention_cache_base = static_cast<size_t>(layer_index) * 5u;
    const size_t convolution_cache_base = static_cast<size_t>(layer_index) * 2u;
    const size_t feed_forward_cache_base = static_cast<size_t>(layer_index) * 4u;
    if (!emel::diarization::sortformer::detail::prepare_dense_weight_cache(
            tensor_data<k_model_dim * k_model_dim>(
                *layer[k_attention_query_weight_index].tensor),
            static_cast<size_t>(k_model_dim),
            static_cast<size_t>(k_model_dim),
            workspace.attention_weight_caches[attention_cache_base +
                                              k_attention_query_cache_offset]) ||
        !emel::diarization::sortformer::detail::prepare_dense_weight_cache(
            tensor_data<k_model_dim * k_model_dim>(
                *layer[k_attention_key_weight_index].tensor),
            static_cast<size_t>(k_model_dim),
            static_cast<size_t>(k_model_dim),
            workspace.attention_weight_caches[attention_cache_base +
                                              k_attention_key_cache_offset]) ||
        !emel::diarization::sortformer::detail::prepare_dense_weight_cache(
            tensor_data<k_model_dim * k_model_dim>(
                *layer[k_attention_value_weight_index].tensor),
            static_cast<size_t>(k_model_dim),
            static_cast<size_t>(k_model_dim),
            workspace.attention_weight_caches[attention_cache_base +
                                              k_attention_value_cache_offset]) ||
        !emel::diarization::sortformer::detail::prepare_dense_weight_cache(
            tensor_data<k_model_dim * k_model_dim>(
                *layer[k_attention_position_weight_index].tensor),
            static_cast<size_t>(k_model_dim),
            static_cast<size_t>(k_model_dim),
            workspace.attention_weight_caches[attention_cache_base +
                                              k_attention_position_cache_offset]) ||
        !emel::diarization::sortformer::detail::prepare_dense_weight_cache(
            tensor_data<k_model_dim * k_model_dim>(
                *layer[k_attention_output_weight_index].tensor),
            static_cast<size_t>(k_model_dim),
            static_cast<size_t>(k_model_dim),
            workspace.attention_weight_caches[attention_cache_base +
                                              k_attention_output_cache_offset]) ||
        !emel::diarization::sortformer::detail::prepare_dense_weight_cache(
            tensor_data<k_model_dim * 2 * k_model_dim>(
                *layer[k_conv_pointwise_1_weight_index].tensor),
            static_cast<size_t>(k_model_dim),
            static_cast<size_t>(2 * k_model_dim),
            workspace.convolution_weight_caches[
                convolution_cache_base + k_convolution_pointwise_1_cache_offset]) ||
        !emel::diarization::sortformer::detail::prepare_dense_weight_cache(
            tensor_data<k_model_dim * k_model_dim>(
                *layer[k_conv_pointwise_2_weight_index].tensor),
            static_cast<size_t>(k_model_dim),
            static_cast<size_t>(k_model_dim),
            workspace.convolution_weight_caches[
                convolution_cache_base + k_convolution_pointwise_2_cache_offset]) ||
        !emel::diarization::sortformer::detail::prepare_dense_weight_cache(
            tensor_data<k_model_dim * k_feed_forward_dim>(
                *layer[k_feed_forward_1_linear_1_weight_index].tensor),
            static_cast<size_t>(k_model_dim),
            static_cast<size_t>(k_feed_forward_dim),
            workspace.feed_forward_weight_caches[feed_forward_cache_base + 0u]) ||
        !emel::diarization::sortformer::detail::prepare_dense_weight_cache(
            tensor_data<k_feed_forward_dim * k_model_dim>(
                *layer[k_feed_forward_1_linear_2_weight_index].tensor),
            static_cast<size_t>(k_feed_forward_dim),
            static_cast<size_t>(k_model_dim),
            workspace.feed_forward_weight_caches[feed_forward_cache_base + 1u]) ||
        !emel::diarization::sortformer::detail::prepare_dense_weight_cache(
            tensor_data<k_model_dim * k_feed_forward_dim>(
                *layer[k_feed_forward_2_linear_1_weight_index].tensor),
            static_cast<size_t>(k_model_dim),
            static_cast<size_t>(k_feed_forward_dim),
            workspace.feed_forward_weight_caches[feed_forward_cache_base + 2u]) ||
        !emel::diarization::sortformer::detail::prepare_dense_weight_cache(
            tensor_data<k_feed_forward_dim * k_model_dim>(
                *layer[k_feed_forward_2_linear_2_weight_index].tensor),
            static_cast<size_t>(k_feed_forward_dim),
            static_cast<size_t>(k_model_dim),
            workspace.feed_forward_weight_caches[feed_forward_cache_base + 3u])) {
      return false;
    }
  }

  return true;
}

bool compute_affine_512(std::span<const float, k_model_dim> input,
                        std::span<const float, k_model_dim> scale,
                        std::span<const float, k_model_dim> bias,
                        std::span<float, k_model_dim> output) noexcept {
  for (size_t index = 0u; index < static_cast<size_t>(k_model_dim); ++index) {
    output[index] = (input[index] * scale[index]) + bias[index];
  }
  return true;
}

bool compute_encoder_frames_from_features(
    std::span<const float> features,
    const contract & encoder_contract,
    pre_encoder_workspace & workspace,
    std::span<float> encoder_frames) noexcept {
  if (features.data() == nullptr ||
      features.size() != static_cast<size_t>(k_required_feature_value_count) ||
      encoder_frames.data() == nullptr ||
      encoder_frames.size() != static_cast<size_t>(k_required_encoder_value_count) ||
      !pre_contract_bound(encoder_contract)) {
    return false;
  }
  if (!prepare_encoder_weight_caches(encoder_contract, workspace)) {
    return false;
  }

  const auto conv_0_bias = tensor_data<k_pre_channel_count>(*encoder_contract.pre[0].tensor);
  const auto conv_0_weight =
      tensor_data<k_pre_channel_count * 9>(*encoder_contract.pre[1].tensor);
  const auto conv_2_bias = tensor_data<k_pre_channel_count>(*encoder_contract.pre[2].tensor);
  const auto conv_2_weight =
      tensor_data<k_pre_channel_count * 9>(*encoder_contract.pre[3].tensor);
  const auto conv_3_bias = tensor_data<k_pre_channel_count>(*encoder_contract.pre[4].tensor);
  const auto conv_3_weight =
      tensor_data<k_pre_channel_count * k_pre_channel_count>(*encoder_contract.pre[5].tensor);
  const auto conv_5_bias = tensor_data<k_pre_channel_count>(*encoder_contract.pre[6].tensor);
  const auto conv_5_weight =
      tensor_data<k_pre_channel_count * 9>(*encoder_contract.pre[7].tensor);
  const auto conv_6_bias = tensor_data<k_pre_channel_count>(*encoder_contract.pre[8].tensor);
  const auto conv_6_weight =
      tensor_data<k_pre_channel_count * k_pre_channel_count>(*encoder_contract.pre[9].tensor);
  const auto out_bias = tensor_data<k_model_dim>(*encoder_contract.pre[10].tensor);
  const auto out_weight =
      tensor_data<k_model_dim * k_pre_expanded_dim>(*encoder_contract.pre[11].tensor);

  for (int32_t conv0_time = 0; conv0_time < k_conv0_time_count; ++conv0_time) {
    auto conv0_row = std::span<float, k_conv0_row_value_count>{
        workspace.conv0_rows[static_cast<size_t>(conv0_time % 3)].data(),
        k_conv0_row_value_count};
    compute_conv0_row(features, conv0_time, conv_0_weight, conv_0_bias, conv0_row);

    if ((conv0_time & 1) == 0) {
      continue;
    }

    const int32_t stage1_time = conv0_time / 2;
    const std::array<const float *, 3> conv0_inputs{
        row_or_null(workspace, (stage1_time * 2) - 1, k_conv0_time_count, true),
        row_or_null(workspace, stage1_time * 2, k_conv0_time_count, true),
        row_or_null(workspace, (stage1_time * 2) + 1, k_conv0_time_count, true),
    };
    compute_depthwise_row(conv0_inputs,
                          k_conv0_freq_count,
                          k_stage1_freq_count,
                          conv_2_weight,
                          conv_2_bias,
                          workspace.stage1_depthwise);

    auto stage1_row = std::span<float, k_stage1_row_value_count>{
        workspace.stage1_rows[static_cast<size_t>(stage1_time % 3)].data(),
        k_stage1_row_value_count};
    if (!compute_pointwise_row(workspace.stage1_depthwise,
                               k_stage1_freq_count,
                               conv_3_weight,
                               conv_3_bias,
                               workspace,
                               stage1_row)) {
      return false;
    }

    if ((stage1_time & 1) == 0) {
      continue;
    }

    const int32_t frame = stage1_time / 2;
    const std::array<const float *, 3> stage1_inputs{
        row_or_null(workspace, (frame * 2) - 1, k_stage1_time_count, false),
        row_or_null(workspace, frame * 2, k_stage1_time_count, false),
        row_or_null(workspace, (frame * 2) + 1, k_stage1_time_count, false),
    };
    compute_depthwise_row(stage1_inputs,
                          k_stage1_freq_count,
                          k_stage2_freq_count,
                          conv_5_weight,
                          conv_5_bias,
                          workspace.stage2_depthwise);
    if (!compute_pointwise_row(workspace.stage2_depthwise,
                               k_stage2_freq_count,
                               conv_6_weight,
                               conv_6_bias,
                               workspace,
                               workspace.stage2_row)) {
      return false;
    }
    flatten_stage2_row(workspace.stage2_row, workspace.flattened);

    std::copy(workspace.flattened.begin(),
              workspace.flattened.end(),
              workspace.pre_encoder_rows.begin() +
                  (static_cast<size_t>(frame) * static_cast<size_t>(k_pre_expanded_dim)));
  }

  if (!emel::diarization::sortformer::detail::compute_dense_batch_prepared(
          workspace.pre_encoder_rows,
          static_cast<size_t>(k_frame_count),
          static_cast<size_t>(k_pre_expanded_dim),
          out_weight,
          workspace.pre_output_weight_cache,
          out_bias,
          static_cast<size_t>(k_model_dim),
          workspace.dense_transposed_input,
          workspace.dense_transposed_output,
          encoder_frames)) {
    return false;
  }

  auto stack_frames = std::span<float, k_required_encoder_value_count>{
      encoder_frames.data(), k_required_encoder_value_count};
  return compute_encoder_stack(encoder_contract, workspace, stack_frames);
}

}  // namespace emel::diarization::sortformer::encoder::detail
