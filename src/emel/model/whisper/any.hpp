#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/model/whisper/detail.hpp"

namespace emel::model::whisper {

using execution_contract = detail::execution_contract;

inline bool is_legacy_lmgg_whisper(std::span<const uint8_t> source) noexcept {
  return detail::is_legacy_lmgg_whisper(source);
}

inline bool normalize_legacy_lmgg_to_gguf(std::span<const uint8_t> source,
                                          std::vector<uint8_t> &gguf_out) {
  return detail::normalize_legacy_lmgg_to_gguf(source, gguf_out);
}

inline emel::error::type
build_execution_contract(const emel::model::data &model_data,
                         execution_contract &contract_out) noexcept {
  return detail::build_execution_contract(model_data, contract_out);
}

} // namespace emel::model::whisper
