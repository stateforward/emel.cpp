#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "emel/diarization/sortformer/executor/errors.hpp"
#include "emel/error/error.hpp"
#include "emel/model/data.hpp"

namespace emel::diarization::sortformer::executor::detail {

inline constexpr int32_t k_frame_count = 188;
inline constexpr int32_t k_encoder_dim = 512;
inline constexpr int32_t k_hidden_dim = 192;
inline constexpr int32_t k_speaker_count = 4;
inline constexpr int32_t k_required_encoder_value_count = k_frame_count * k_encoder_dim;
inline constexpr int32_t k_required_hidden_value_count = k_frame_count * k_hidden_dim;

inline emel::error::type to_error(const error err) noexcept {
  return emel::error::cast(err);
}

template <size_t Size>
std::span<const float, Size> tensor_data(
    const emel::model::data::tensor_record & tensor) noexcept {
  return std::span<const float, Size>{static_cast<const float *>(tensor.data), Size};
}

}  // namespace emel::diarization::sortformer::executor::detail
