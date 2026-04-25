#include "emel/diarization/sortformer/transformer/detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

#include "emel/diarization/sortformer/detail.hpp"

namespace emel::diarization::sortformer::transformer::detail {

namespace {

struct tensor_spec {
  std::string_view name = {};
  int32_t n_dims = 0;
  std::array<int64_t, 4> dims = {};
};

constexpr std::array<tensor_spec, k_layer_tensor_count> k_layer_specs{{
    {"sa.k.b", 1, {k_hidden_dim, 0, 0, 0}},
    {"sa.k.w", 2, {k_hidden_dim, k_hidden_dim, 0, 0}},
    {"sa.o.b", 1, {k_hidden_dim, 0, 0, 0}},
    {"sa.o.w", 2, {k_hidden_dim, k_hidden_dim, 0, 0}},
    {"sa.q.b", 1, {k_hidden_dim, 0, 0, 0}},
    {"sa.q.w", 2, {k_hidden_dim, k_hidden_dim, 0, 0}},
    {"sa.v.b", 1, {k_hidden_dim, 0, 0, 0}},
    {"sa.v.w", 2, {k_hidden_dim, k_hidden_dim, 0, 0}},
    {"ln1.b", 1, {k_hidden_dim, 0, 0, 0}},
    {"ln1.w", 1, {k_hidden_dim, 0, 0, 0}},
    {"ln2.b", 1, {k_hidden_dim, 0, 0, 0}},
    {"ln2.w", 1, {k_hidden_dim, 0, 0, 0}},
    {"ff.di.b", 1, {k_inner_dim, 0, 0, 0}},
    {"ff.di.w", 2, {k_hidden_dim, k_inner_dim, 0, 0}},
    {"ff.do.b", 1, {k_hidden_dim, 0, 0, 0}},
    {"ff.do.w", 2, {k_inner_dim, k_hidden_dim, 0, 0}},
}};

enum attention_weight_cache_index : size_t {
  k_attention_query_cache_offset = 0u,
  k_attention_key_cache_offset = 1u,
  k_attention_value_cache_offset = 2u,
  k_attention_output_cache_offset = 3u,
};

enum feed_forward_weight_cache_index : size_t {
  k_feed_forward_in_cache_offset = 0u,
  k_feed_forward_out_cache_offset = 1u,
};

template <size_t Size>
std::span<const float, Size> tensor_data(
    const emel::model::data::tensor_record & tensor) noexcept {
  return std::span<const float, Size>{static_cast<const float *>(tensor.data), Size};
}

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

bool bind_layer_tensor(const emel::model::data & model_data,
                       const int32_t layer,
                       const tensor_spec & spec,
                       tensor_view & view_out) noexcept {
  std::array<char, 64> name = {};
  const int written = std::snprintf(name.data(),
                                    name.size(),
                                    "te.l%d.%.*s",
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

float compute_head_score(std::span<const float, k_max_frame_count * k_hidden_dim> query,
                         std::span<const float, k_max_frame_count * k_hidden_dim> key,
                         uint32_t query_frame,
                         uint32_t key_frame,
                         size_t head_offset) noexcept {
  float acc = 0.0f;
  const size_t query_base = (static_cast<size_t>(query_frame) * k_hidden_dim) + head_offset;
  const size_t key_base = (static_cast<size_t>(key_frame) * k_hidden_dim) + head_offset;
  for (size_t dim = 0u; dim < static_cast<size_t>(k_attention_head_dim); ++dim) {
    acc += query[query_base + dim] * key[key_base + dim];
  }

  return acc / std::sqrt(static_cast<float>(k_attention_head_dim));
}

void compute_head_attention(std::span<const float, k_max_frame_count * k_hidden_dim> query,
                            std::span<const float, k_max_frame_count * k_hidden_dim> key,
                            std::span<const float, k_max_frame_count * k_hidden_dim> value,
                            uint32_t frame_count,
                            uint32_t query_frame,
                            size_t head_offset,
                            std::span<float, k_max_frame_count> scores,
                            std::span<float, k_hidden_dim> attended) noexcept {
  float max_score = compute_head_score(query, key, query_frame, 0u, head_offset);
  scores[0] = max_score;

  for (uint32_t frame_index = 1u; frame_index < frame_count; ++frame_index) {
    const float score = compute_head_score(query, key, query_frame, frame_index, head_offset);
    scores[frame_index] = score;
    max_score = std::max(max_score, score);
  }

  float normalizer = 0.0f;
  for (uint32_t frame_index = 0u; frame_index < frame_count; ++frame_index) {
    const float weight = std::exp(scores[frame_index] - max_score);
    scores[frame_index] = weight;
    normalizer += weight;
  }

  for (size_t dim = 0u; dim < static_cast<size_t>(k_attention_head_dim); ++dim) {
    float acc = 0.0f;
    for (uint32_t frame_index = 0u; frame_index < frame_count; ++frame_index) {
      const size_t value_offset =
          (static_cast<size_t>(frame_index) * k_hidden_dim) + head_offset + dim;
      acc += (scores[frame_index] / normalizer) * value[value_offset];
    }
    attended[head_offset + dim] = acc;
  }
}

}  // namespace

layer_workspace::layer_workspace() {
  for (auto & cache : attention_weight_caches) {
    cache.lhs_4row.resize(k_hidden_projection_prepared_weight_value_count);
  }
  for (auto & cache : feed_forward_weight_caches) {
    cache.lhs_4row.resize(k_feed_forward_prepared_weight_value_count);
  }
}

bool bind_contract(const emel::model::data & model_data,
                   contract & contract_out) noexcept {
  contract next = {};

  for (int32_t layer = 0; layer < k_layer_count; ++layer) {
    auto & view = next.layers[static_cast<size_t>(layer)];
    tensor_view * views[k_layer_tensor_count] = {
        &view.key_bias,
        &view.key_weight,
        &view.output_bias,
        &view.output_weight,
        &view.query_bias,
        &view.query_weight,
        &view.value_bias,
        &view.value_weight,
        &view.layer_norm_1_bias,
        &view.layer_norm_1_weight,
        &view.layer_norm_2_bias,
        &view.layer_norm_2_weight,
        &view.feed_forward_in_bias,
        &view.feed_forward_in_weight,
        &view.feed_forward_out_bias,
        &view.feed_forward_out_weight,
    };

    for (size_t index = 0u; index < k_layer_specs.size(); ++index) {
      if (!bind_layer_tensor(model_data, layer, k_layer_specs[index], *views[index])) {
        return false;
      }
      ++next.tensor_count;
    }
  }

  contract_out = next;
  return true;
}

bool prepare_weight_caches(const contract & transformer_contract,
                           layer_workspace & workspace) noexcept {
  for (int32_t layer_index = 0; layer_index < k_layer_count; ++layer_index) {
    const auto & layer = transformer_contract.layers[static_cast<size_t>(layer_index)];
    const size_t attention_cache_base = static_cast<size_t>(layer_index) * 4u;
    const size_t feed_forward_cache_base = static_cast<size_t>(layer_index) * 2u;
    if (!emel::diarization::sortformer::detail::prepare_dense_weight_cache(
            tensor_data<k_hidden_dim * k_hidden_dim>(*layer.query_weight.tensor),
            static_cast<size_t>(k_hidden_dim),
            static_cast<size_t>(k_hidden_dim),
            workspace.attention_weight_caches[attention_cache_base +
                                              k_attention_query_cache_offset]) ||
        !emel::diarization::sortformer::detail::prepare_dense_weight_cache(
            tensor_data<k_hidden_dim * k_hidden_dim>(*layer.key_weight.tensor),
            static_cast<size_t>(k_hidden_dim),
            static_cast<size_t>(k_hidden_dim),
            workspace.attention_weight_caches[attention_cache_base +
                                              k_attention_key_cache_offset]) ||
        !emel::diarization::sortformer::detail::prepare_dense_weight_cache(
            tensor_data<k_hidden_dim * k_hidden_dim>(*layer.value_weight.tensor),
            static_cast<size_t>(k_hidden_dim),
            static_cast<size_t>(k_hidden_dim),
            workspace.attention_weight_caches[attention_cache_base +
                                              k_attention_value_cache_offset]) ||
        !emel::diarization::sortformer::detail::prepare_dense_weight_cache(
            tensor_data<k_hidden_dim * k_hidden_dim>(*layer.output_weight.tensor),
            static_cast<size_t>(k_hidden_dim),
            static_cast<size_t>(k_hidden_dim),
            workspace.attention_weight_caches[attention_cache_base +
                                              k_attention_output_cache_offset]) ||
        !emel::diarization::sortformer::detail::prepare_dense_weight_cache(
            tensor_data<k_inner_dim * k_hidden_dim>(*layer.feed_forward_in_weight.tensor),
            static_cast<size_t>(k_hidden_dim),
            static_cast<size_t>(k_inner_dim),
            workspace.feed_forward_weight_caches[feed_forward_cache_base +
                                                 k_feed_forward_in_cache_offset]) ||
        !emel::diarization::sortformer::detail::prepare_dense_weight_cache(
            tensor_data<k_hidden_dim * k_inner_dim>(*layer.feed_forward_out_weight.tensor),
            static_cast<size_t>(k_inner_dim),
            static_cast<size_t>(k_hidden_dim),
            workspace.feed_forward_weight_caches[feed_forward_cache_base +
                                                 k_feed_forward_out_cache_offset])) {
      return false;
    }
  }

  return true;
}

bool compute_layer_norm(std::span<const float, k_hidden_dim> input,
                        std::span<const float, k_hidden_dim> scale,
                        std::span<const float, k_hidden_dim> bias,
                        std::span<float, k_hidden_dim> output) noexcept {
  float mean = 0.0f;
  for (const float value : input) {
    mean += value;
  }
  mean /= static_cast<float>(k_hidden_dim);

  float variance = 0.0f;
  for (const float value : input) {
    const float centered = value - mean;
    variance += centered * centered;
  }
  variance /= static_cast<float>(k_hidden_dim);

  const float inv_std = 1.0f / std::sqrt(variance + 1.0e-5f);
  for (size_t index = 0u; index < static_cast<size_t>(k_hidden_dim); ++index) {
    output[index] = ((input[index] - mean) * inv_std * scale[index]) + bias[index];
  }

  return true;
}

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
    std::span<float> output_frames) noexcept {
  if (frame_count == 0u || frame_count > static_cast<uint32_t>(k_max_frame_count) ||
      input_frames.size() != static_cast<size_t>(frame_count) * k_hidden_dim ||
      output_frames.size() != input_frames.size()) {
    return false;
  }

  const size_t frame_value_count = static_cast<size_t>(frame_count) *
      static_cast<size_t>(k_hidden_dim);
  auto qkv_transposed = std::span<float>{workspace.dense_transposed_input.data(),
                                         frame_value_count};
  if (!emel::diarization::sortformer::detail::transpose_dense_input(
          input_frames,
          static_cast<size_t>(frame_count),
          static_cast<size_t>(k_hidden_dim),
          qkv_transposed) ||
      !emel::diarization::sortformer::detail::compute_dense_batch_from_transposed_prepared(
          qkv_transposed,
          static_cast<size_t>(frame_count),
          static_cast<size_t>(k_hidden_dim),
          query_weight,
          query_cache,
          query_bias,
          static_cast<size_t>(k_hidden_dim),
          workspace.dense_transposed_output,
          std::span<float>{workspace.query.data(), frame_value_count}) ||
      !emel::diarization::sortformer::detail::compute_dense_batch_from_transposed_prepared(
          qkv_transposed,
          static_cast<size_t>(frame_count),
          static_cast<size_t>(k_hidden_dim),
          key_weight,
          key_cache,
          key_bias,
          static_cast<size_t>(k_hidden_dim),
          workspace.dense_transposed_output,
          std::span<float>{workspace.key.data(), frame_value_count}) ||
      !emel::diarization::sortformer::detail::compute_dense_batch_from_transposed_prepared(
          qkv_transposed,
          static_cast<size_t>(frame_count),
          static_cast<size_t>(k_hidden_dim),
          value_weight,
          value_cache,
          value_bias,
          static_cast<size_t>(k_hidden_dim),
          workspace.dense_transposed_output,
          std::span<float>{workspace.value.data(), frame_value_count})) {
    return false;
  }

  for (uint32_t frame_index = 0u; frame_index < frame_count; ++frame_index) {
    workspace.attended.fill(0.0f);
    for (size_t head = 0u; head < static_cast<size_t>(k_attention_head_count); ++head) {
      compute_head_attention(workspace.query,
                             workspace.key,
                             workspace.value,
                             frame_count,
                             frame_index,
                             head * static_cast<size_t>(k_attention_head_dim),
                             workspace.scores,
                             workspace.attended);
    }

    const size_t frame_base = static_cast<size_t>(frame_index) * k_hidden_dim;
    std::copy(workspace.attended.begin(),
              workspace.attended.end(),
              workspace.first_norm.begin() + frame_base);
  }

  if (!emel::diarization::sortformer::detail::compute_dense_batch_residual_prepared(
          std::span<const float>{workspace.first_norm.data(), frame_value_count},
          static_cast<size_t>(frame_count),
          static_cast<size_t>(k_hidden_dim),
          output_weight,
          output_cache,
          output_bias,
          static_cast<size_t>(k_hidden_dim),
          input_frames,
          workspace.dense_transposed_input,
          workspace.dense_transposed_output,
          std::span<float>{workspace.query.data(), frame_value_count})) {
    return false;
  }

  for (uint32_t frame_index = 0u; frame_index < frame_count; ++frame_index) {
    const size_t frame_base = static_cast<size_t>(frame_index) * k_hidden_dim;
    const auto residual = std::span<const float, k_hidden_dim>{
        workspace.query.data() + frame_base, k_hidden_dim};
    auto first_norm = std::span<float, k_hidden_dim>{workspace.first_norm.data() + frame_base,
                                                     k_hidden_dim};
    if (!compute_layer_norm(residual,
                            layer_norm_1_weight,
                            layer_norm_1_bias,
                            first_norm)) {
      return false;
    }
  }

  if (!emel::diarization::sortformer::detail::compute_dense_batch_prepared(
          std::span<const float>{workspace.first_norm.data(), frame_value_count},
          static_cast<size_t>(frame_count),
          static_cast<size_t>(k_hidden_dim),
          feed_forward_in_weight,
          feed_forward_in_cache,
          feed_forward_in_bias,
          static_cast<size_t>(k_inner_dim),
          workspace.dense_transposed_input,
          workspace.dense_transposed_output,
          std::span<float>{workspace.feed_forward_rows.data(),
                           static_cast<size_t>(frame_count) * static_cast<size_t>(k_inner_dim)})) {
    return false;
  }

  const size_t feed_forward_value_count = static_cast<size_t>(frame_count) *
      static_cast<size_t>(k_inner_dim);
  for (float & value : std::span<float>{workspace.feed_forward_rows.data(),
                                        feed_forward_value_count}) {
    value = std::max(value, 0.0f);
  }

  if (!emel::diarization::sortformer::detail::compute_dense_batch_residual_prepared(
          std::span<const float>{workspace.feed_forward_rows.data(), feed_forward_value_count},
          static_cast<size_t>(frame_count),
          static_cast<size_t>(k_inner_dim),
          feed_forward_out_weight,
          feed_forward_out_cache,
          feed_forward_out_bias,
          static_cast<size_t>(k_hidden_dim),
          std::span<const float>{workspace.first_norm.data(), frame_value_count},
          workspace.dense_transposed_input,
          workspace.dense_transposed_output,
          std::span<float>{workspace.query.data(), frame_value_count})) {
    return false;
  }

  for (uint32_t frame_index = 0u; frame_index < frame_count; ++frame_index) {
    const size_t frame_base = static_cast<size_t>(frame_index) * k_hidden_dim;
    const auto residual = std::span<const float, k_hidden_dim>{
        workspace.query.data() + frame_base, k_hidden_dim};
    auto output = std::span<float, k_hidden_dim>{output_frames.data() + frame_base,
                                                k_hidden_dim};
    if (!compute_layer_norm(residual,
                            layer_norm_2_weight,
                            layer_norm_2_bias,
                            output)) {
      return false;
    }
  }

  return true;
}

}  // namespace emel::diarization::sortformer::transformer::detail
