#pragma once

#include <cstdint>

namespace emel::diarization::request {

enum class error : uint32_t {
  none = 0u,
  model_invalid = (1u << 0),
  sample_rate = (1u << 1),
  channel_count = (1u << 2),
  pcm_shape = (1u << 3),
  capacity = (1u << 4),
  feature_extractor = (1u << 5),
  unexpected = (1u << 6),
};

}  // namespace emel::diarization::request
