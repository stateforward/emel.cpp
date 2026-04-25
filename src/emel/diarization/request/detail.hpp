#pragma once

#include <cstdint>

#include "emel/diarization/request/errors.hpp"
#include "emel/diarization/sortformer/encoder/feature_extractor/detail.hpp"
#include "emel/error/error.hpp"

namespace emel::diarization::request::detail {

inline constexpr int32_t k_sample_rate =
    emel::diarization::sortformer::encoder::feature_extractor::detail::k_sample_rate;
inline constexpr int32_t k_channel_count =
    emel::diarization::sortformer::encoder::feature_extractor::detail::k_channel_count;
inline constexpr int32_t k_speaker_count =
    emel::diarization::sortformer::encoder::feature_extractor::detail::k_speaker_count;
inline constexpr int32_t k_frame_shift_ms =
    emel::diarization::sortformer::encoder::feature_extractor::detail::k_frame_shift_ms;
inline constexpr int32_t k_chunk_len =
    emel::diarization::sortformer::encoder::feature_extractor::detail::k_chunk_len;
inline constexpr int32_t k_chunk_right_context =
    emel::diarization::sortformer::encoder::feature_extractor::detail::k_chunk_right_context;
inline constexpr int32_t k_feature_bin_count =
    emel::diarization::sortformer::encoder::feature_extractor::detail::k_feature_bin_count;
inline constexpr int32_t k_required_sample_count =
    emel::diarization::sortformer::encoder::feature_extractor::detail::k_required_sample_count;
inline constexpr int32_t k_required_feature_count =
    emel::diarization::sortformer::encoder::feature_extractor::detail::k_required_feature_count;

inline emel::error::type to_error(const error err) noexcept {
  return emel::error::cast(err);
}

}  // namespace emel::diarization::request::detail
