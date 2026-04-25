#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include "emel/model/data.hpp"

namespace emel::diarization::sortformer::encoder::feature_extractor::detail {

inline constexpr int32_t k_sample_rate = 16000;
inline constexpr int32_t k_channel_count = 1;
inline constexpr int32_t k_speaker_count = 4;
inline constexpr int32_t k_frame_shift_ms = 80;
inline constexpr int32_t k_chunk_len = 188;
inline constexpr int32_t k_chunk_right_context = 1;
inline constexpr int32_t k_feature_bin_count = 128;
inline constexpr int32_t k_window_length = 400;
inline constexpr int32_t k_hop_length = 160;
inline constexpr int32_t k_fft_size = 512;
inline constexpr int32_t k_fft_bin_count = (k_fft_size / 2) + 1;
inline constexpr int32_t k_feature_frame_count = 1504;
inline constexpr int32_t k_required_sample_count = 240640;
inline constexpr int32_t k_required_feature_count =
    k_feature_frame_count * k_feature_bin_count;
inline constexpr int32_t k_window_fft_padding = (k_fft_size - k_window_length) / 2;
inline constexpr float k_preemphasis = 0.97f;
inline constexpr float k_log_zero_guard_value = 1.0f / 16777216.0f;

struct tensor_view {
  const emel::model::data::tensor_record * tensor = nullptr;
  std::string_view name = {};
};

struct contract {
  tensor_view filter_bank = {};
  tensor_view window = {};
};

contract make_contract(const emel::model::data & model_data) noexcept;

bool contract_valid(const contract & feature_contract) noexcept;

void compute(std::span<const float> pcm,
             const contract & feature_contract,
             std::span<float> features) noexcept;

template <size_t Size>
std::span<const float, Size> tensor_data(
    const emel::model::data::tensor_record & tensor) noexcept {
  return std::span<const float, Size>{static_cast<const float *>(tensor.data), Size};
}

}  // namespace emel::diarization::sortformer::encoder::feature_extractor::detail
