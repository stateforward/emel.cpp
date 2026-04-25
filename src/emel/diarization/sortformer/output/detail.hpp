#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "emel/diarization/sortformer/modules/detail.hpp"

namespace emel::diarization::sortformer::output::detail {

inline constexpr int32_t k_frame_count = 188;
inline constexpr int32_t k_hidden_dim = 192;
inline constexpr int32_t k_speaker_count = 4;
inline constexpr float k_frame_shift_seconds = 0.08f;
inline constexpr float k_default_activity_threshold = 0.5f;
inline constexpr int32_t k_required_hidden_value_count = k_frame_count * k_hidden_dim;
inline constexpr int32_t k_required_probability_value_count = k_frame_count * k_speaker_count;

struct segment_record {
  int32_t speaker = 0;
  int32_t start_frame = 0;
  int32_t end_frame = 0;
  float start_seconds = 0.0f;
  float end_seconds = 0.0f;
  float max_probability = 0.0f;
};

bool compute_speaker_probabilities(
    std::span<const float> hidden_frames,
    const emel::diarization::sortformer::modules::detail::contract & modules_contract,
    std::span<float> probabilities_out) noexcept;

bool decode_segments(std::span<const float> probabilities,
                     float threshold,
                     std::span<segment_record> segments_out,
                     int32_t & segment_count_out) noexcept;

std::string_view speaker_label(int32_t speaker) noexcept;

template <size_t Size>
std::span<const float, Size> tensor_data(
    const emel::model::data::tensor_record & tensor) noexcept {
  return std::span<const float, Size>{static_cast<const float *>(tensor.data), Size};
}

}  // namespace emel::diarization::sortformer::output::detail
