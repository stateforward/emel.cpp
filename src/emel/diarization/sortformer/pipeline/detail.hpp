#pragma once

#include <cstdint>

#include "emel/diarization/request/detail.hpp"
#include "emel/diarization/sortformer/encoder/detail.hpp"
#include "emel/diarization/sortformer/executor/detail.hpp"
#include "emel/diarization/sortformer/output/detail.hpp"
#include "emel/diarization/sortformer/pipeline/errors.hpp"
#include "emel/error/error.hpp"

namespace emel::diarization::sortformer::pipeline::detail {

inline constexpr int32_t k_sample_rate = emel::diarization::request::detail::k_sample_rate;
inline constexpr int32_t k_channel_count = emel::diarization::request::detail::k_channel_count;
inline constexpr int32_t k_frame_count =
    emel::diarization::sortformer::output::detail::k_frame_count;
inline constexpr int32_t k_feature_bin_count =
    emel::diarization::sortformer::encoder::detail::k_feature_bin_count;
inline constexpr int32_t k_hidden_dim =
    emel::diarization::sortformer::output::detail::k_hidden_dim;
inline constexpr int32_t k_speaker_count =
    emel::diarization::sortformer::output::detail::k_speaker_count;
inline constexpr int32_t k_required_sample_count =
    emel::diarization::request::detail::k_required_sample_count;
inline constexpr int32_t k_required_feature_count =
    emel::diarization::request::detail::k_required_feature_count;
inline constexpr int32_t k_required_encoder_value_count =
    emel::diarization::sortformer::encoder::detail::k_required_encoder_value_count;
inline constexpr int32_t k_required_hidden_value_count =
    emel::diarization::sortformer::executor::detail::k_required_hidden_value_count;
inline constexpr int32_t k_required_probability_value_count =
    emel::diarization::sortformer::output::detail::k_required_probability_value_count;
inline constexpr int32_t k_max_segment_count = k_frame_count * k_speaker_count;

inline emel::error::type to_error(const error err) noexcept {
  return emel::error::cast(err);
}

}  // namespace emel::diarization::sortformer::pipeline::detail
