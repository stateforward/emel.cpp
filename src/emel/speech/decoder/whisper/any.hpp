#pragma once

#include <cstdint>

#include "emel/model/data.hpp"
#include "emel/speech/decoder/whisper/detail.hpp"

namespace emel::speech::decoder::whisper {

using execution_contract = detail::execution_contract;

inline execution_contract
bind_execution_contract(const emel::model::data &model) noexcept {
  return detail::bind_execution_contract(model);
}

inline uint64_t
required_workspace_floats(const uint64_t encoder_frames) noexcept {
  return detail::required_decoder_workspace_floats(encoder_frames);
}

inline int32_t vocab_size() noexcept {
  return detail::k_vocab_size;
}

} // namespace emel::speech::decoder::whisper
