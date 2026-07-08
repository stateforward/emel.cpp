#pragma once

#include <cstdint>

#include "emel/diarization/sortformer/output/any.hpp"

namespace emel::diarization::sortformer::pipeline {

using segment_record = emel::diarization::sortformer::output::segment_record;

inline constexpr int32_t k_sample_rate = 16000;
inline constexpr int32_t k_channel_count = 1;
inline constexpr int32_t k_frame_count =
    emel::diarization::sortformer::output::k_frame_count;
inline constexpr int32_t k_feature_bin_count = 128;
inline constexpr int32_t k_feature_frame_count = 1504;
inline constexpr int32_t k_hidden_dim =
    emel::diarization::sortformer::output::k_hidden_dim;
inline constexpr int32_t k_speaker_count =
    emel::diarization::sortformer::output::k_speaker_count;
inline constexpr int32_t k_required_sample_count = 240640;
inline constexpr int32_t k_required_feature_count =
    k_feature_frame_count * k_feature_bin_count;
inline constexpr int32_t k_encoder_dim = 512;
inline constexpr int32_t k_required_encoder_value_count =
    k_frame_count * k_encoder_dim;
inline constexpr int32_t k_required_hidden_value_count =
    emel::diarization::sortformer::output::k_required_hidden_value_count;
inline constexpr int32_t k_required_probability_value_count =
    emel::diarization::sortformer::output::k_required_probability_value_count;
inline constexpr int32_t k_max_segment_count =
    emel::diarization::sortformer::output::k_max_segment_count;

}  // namespace emel::diarization::sortformer::pipeline
