#pragma once

#include <cstdint>

namespace emel::diarization::sortformer::output {

inline constexpr int32_t k_frame_count = 188;
inline constexpr int32_t k_hidden_dim = 192;
inline constexpr int32_t k_speaker_count = 4;
inline constexpr float k_frame_shift_seconds = 0.08f;
inline constexpr float k_default_activity_threshold = 0.5f;
inline constexpr int32_t k_required_hidden_value_count =
    k_frame_count * k_hidden_dim;
inline constexpr int32_t k_required_probability_value_count =
    k_frame_count * k_speaker_count;
inline constexpr int32_t k_max_segment_count = k_frame_count * k_speaker_count;

struct segment_record {
  int32_t speaker = 0;
  int32_t start_frame = 0;
  int32_t end_frame = 0;
  float start_seconds = 0.0f;
  float end_seconds = 0.0f;
  float max_probability = 0.0f;
};

} // namespace emel::diarization::sortformer::output
