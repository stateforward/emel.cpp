#pragma once

#include <cstdint>

namespace emel::diarization::sortformer::executor {

enum class error : uint32_t {
  none = 0u,
  model_invalid = (1u << 0),
  tensor_contract = (1u << 1),
  input_shape = (1u << 2),
  output_capacity = (1u << 3),
  kernel = (1u << 4),
  unexpected = (1u << 5),
};

}  // namespace emel::diarization::sortformer::executor
