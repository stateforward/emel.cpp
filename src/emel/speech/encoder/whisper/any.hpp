#pragma once

#include <cstdint>

#include "emel/model/data.hpp"
#include "emel/speech/encoder/whisper/detail.hpp"

namespace emel::speech::encoder::whisper {

using execution_contract = detail::execution_contract;

inline execution_contract
bind_execution_contract(const emel::model::data &model) noexcept {
  return detail::bind_execution_contract(model);
}

inline uint64_t required_workspace_floats(
    const uint64_t sample_count) noexcept {
  return detail::required_workspace_floats(sample_count);
}

inline uint64_t required_encoder_output_floats(
    const uint64_t sample_count) noexcept {
  return detail::required_encoder_output_floats(sample_count);
}

} // namespace emel::speech::encoder::whisper
