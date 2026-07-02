#include "emel/diarization/sortformer/modules/detail.hpp"

#include <array>
#include <cstddef>

#include "emel/diarization/sortformer/detail.hpp"

namespace emel::diarization::sortformer::modules::detail {

namespace {

struct tensor_spec {
  std::string_view name = {};
  int32_t n_dims = 0;
  std::array<int64_t, 4> dims = {};
};

constexpr std::array<tensor_spec, k_tensor_count> k_specs{{
    {"mods.ep.b", 1, {k_hidden_dim, 0, 0, 0}},
    {"mods.ep.w", 2, {k_encoder_dim, k_hidden_dim, 0, 0}},
    {"mods.fh2h.b", 1, {k_hidden_dim, 0, 0, 0}},
    {"mods.fh2h.w", 2, {k_hidden_dim, k_hidden_dim, 0, 0}},
    {"mods.h2s.b", 1, {k_speaker_count, 0, 0, 0}},
    {"mods.h2s.w", 2, {k_pair_hidden_dim, k_speaker_count, 0, 0}},
    {"mods.sh2s.b", 1, {k_speaker_count, 0, 0, 0}},
    {"mods.sh2s.w", 2, {k_hidden_dim, k_speaker_count, 0, 0}},
}};

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

}  // namespace

bool bind_contract(const emel::model::data & model_data,
                   contract & contract_out) noexcept {
  contract next = {};
  tensor_view * views[k_tensor_count] = {
      &next.encoder_projection_bias,
      &next.encoder_projection_weight,
      &next.frame_hidden_bias,
      &next.frame_hidden_weight,
      &next.hidden_to_speaker_bias,
      &next.hidden_to_speaker_weight,
      &next.speaker_hidden_to_speaker_bias,
      &next.speaker_hidden_to_speaker_weight,
  };

  for (size_t index = 0u; index < k_specs.size(); ++index) {
    if (!bind_tensor(model_data, k_specs[index], *views[index])) {
      return false;
    }
    ++next.tensor_count;
  }

  contract_out = next;
  return true;
}

bool compute_encoder_projection(std::span<const float, k_encoder_dim> encoder_frame,
                                std::span<const float, k_hidden_dim * k_encoder_dim> weights,
                                std::span<const float, k_hidden_dim> bias,
                                std::span<float, k_hidden_dim> hidden_out) noexcept {
  return emel::diarization::sortformer::detail::compute_dense(encoder_frame,
                                                              weights,
                                                              bias,
                                                              hidden_out);
}

bool prepare_encoder_projection_weight_cache(
    std::span<const float, k_hidden_dim * k_encoder_dim> weights,
    emel::diarization::sortformer::detail::dense_weight_cache & cache) noexcept {
  return emel::diarization::sortformer::detail::prepare_dense_weight_cache(
      weights,
      static_cast<size_t>(k_encoder_dim),
      static_cast<size_t>(k_hidden_dim),
      cache);
}

bool compute_encoder_projection_batch(
    std::span<const float> encoder_frames,
    const size_t frame_count,
    std::span<const float, k_hidden_dim * k_encoder_dim> weights,
    const emel::diarization::sortformer::detail::dense_weight_cache & cache,
    std::span<const float, k_hidden_dim> bias,
    std::span<float> transposed_input,
    std::span<float> transposed_output,
    std::span<float> hidden_out) noexcept {
  return emel::diarization::sortformer::detail::compute_dense_batch_prepared(
      encoder_frames,
      frame_count,
      static_cast<size_t>(k_encoder_dim),
      weights,
      cache,
      bias,
      static_cast<size_t>(k_hidden_dim),
      transposed_input,
      transposed_output,
      hidden_out);
}

bool compute_speaker_logits(std::span<const float, k_hidden_dim> hidden,
                            std::span<const float, k_hidden_dim> cached_hidden,
                            std::span<const float, k_speaker_count * k_pair_hidden_dim> weights,
                            std::span<const float, k_speaker_count> bias,
                            std::span<float, k_speaker_count> logits_out) noexcept {
  for (size_t speaker = 0u; speaker < static_cast<size_t>(k_speaker_count); ++speaker) {
    float acc = bias[speaker];
    const size_t weight_base = speaker * static_cast<size_t>(k_pair_hidden_dim);
    for (size_t col = 0u; col < static_cast<size_t>(k_hidden_dim); ++col) {
      acc += weights[weight_base + col] * hidden[col];
      acc += weights[weight_base + static_cast<size_t>(k_hidden_dim) + col] *
          cached_hidden[col];
    }
    logits_out[speaker] = acc;
  }
  return true;
}

}  // namespace emel::diarization::sortformer::modules::detail
