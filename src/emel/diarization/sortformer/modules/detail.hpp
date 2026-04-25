#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <span>
#include <string_view>

#include "emel/diarization/sortformer/detail.hpp"
#include "emel/model/data.hpp"

namespace emel::diarization::sortformer::modules::detail {

inline constexpr int32_t k_encoder_dim = 512;
inline constexpr int32_t k_hidden_dim = 192;
inline constexpr int32_t k_pair_hidden_dim = 384;
inline constexpr int32_t k_speaker_count = 4;
inline constexpr int32_t k_tensor_count = 8;
inline constexpr size_t k_encoder_projection_prepared_weight_value_count =
    static_cast<size_t>(k_hidden_dim) * static_cast<size_t>(k_encoder_dim);

struct tensor_view {
  const emel::model::data::tensor_record * tensor = nullptr;
  std::string_view name = {};
};

struct contract {
  tensor_view encoder_projection_weight = {};
  tensor_view encoder_projection_bias = {};
  tensor_view frame_hidden_weight = {};
  tensor_view frame_hidden_bias = {};
  tensor_view hidden_to_speaker_weight = {};
  tensor_view hidden_to_speaker_bias = {};
  tensor_view speaker_hidden_to_speaker_weight = {};
  tensor_view speaker_hidden_to_speaker_bias = {};
  uint32_t tensor_count = 0;
};

bool bind_contract(const emel::model::data & model_data,
                   contract & contract_out) noexcept;

bool compute_encoder_projection(std::span<const float, k_encoder_dim> encoder_frame,
                                std::span<const float, k_hidden_dim * k_encoder_dim> weights,
                                std::span<const float, k_hidden_dim> bias,
                                std::span<float, k_hidden_dim> hidden_out) noexcept;

bool prepare_encoder_projection_weight_cache(
    std::span<const float, k_hidden_dim * k_encoder_dim> weights,
    emel::diarization::sortformer::detail::dense_weight_cache & cache) noexcept;

bool compute_encoder_projection_batch(
    std::span<const float> encoder_frames,
    size_t frame_count,
    std::span<const float, k_hidden_dim * k_encoder_dim> weights,
    const emel::diarization::sortformer::detail::dense_weight_cache & cache,
    std::span<const float, k_hidden_dim> bias,
    std::span<float> transposed_input,
    std::span<float> transposed_output,
    std::span<float> hidden_out) noexcept;

bool compute_speaker_logits(std::span<const float, k_hidden_dim> hidden,
                            std::span<const float, k_hidden_dim> cached_hidden,
                            std::span<const float, k_speaker_count * k_pair_hidden_dim> weights,
                            std::span<const float, k_speaker_count> bias,
                            std::span<float, k_speaker_count> logits_out) noexcept;

}  // namespace emel::diarization::sortformer::modules::detail
