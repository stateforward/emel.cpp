#pragma once

#include <cstdint>

namespace emel::diarization::sortformer::pipeline {

enum class error : uint32_t {
  none = 0u,
  model_invalid = (1u << 0),
  sample_rate = (1u << 1),
  channel_count = (1u << 2),
  pcm_shape = (1u << 3),
  probability_capacity = (1u << 4),
  segment_capacity = (1u << 5),
  tensor_contract = (1u << 6),
  request = (1u << 7),
  executor = (1u << 8),
  unexpected = (1u << 9),
  kernel = (1u << 10),
};

}  // namespace emel::diarization::sortformer::pipeline
