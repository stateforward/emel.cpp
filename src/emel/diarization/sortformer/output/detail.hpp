#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "emel/diarization/sortformer/modules/detail.hpp"
#include "emel/diarization/sortformer/output/any.hpp"
#include "emel/kernel/sm.hpp"

namespace emel::diarization::sortformer::output::detail {

using segment_record = emel::diarization::sortformer::output::segment_record;

inline constexpr int32_t k_frame_count =
    emel::diarization::sortformer::output::k_frame_count;
inline constexpr int32_t k_hidden_dim =
    emel::diarization::sortformer::output::k_hidden_dim;
inline constexpr int32_t k_speaker_count =
    emel::diarization::sortformer::output::k_speaker_count;
inline constexpr float k_frame_shift_seconds =
    emel::diarization::sortformer::output::k_frame_shift_seconds;
inline constexpr float k_default_activity_threshold =
    emel::diarization::sortformer::output::k_default_activity_threshold;
inline constexpr int32_t k_required_hidden_value_count =
    emel::diarization::sortformer::output::k_required_hidden_value_count;
inline constexpr int32_t k_required_probability_value_count =
    emel::diarization::sortformer::output::k_required_probability_value_count;

bool compute_speaker_probabilities(
    emel::kernel::sm & kernel,
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
