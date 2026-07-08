#pragma once

#include <cstdint>
#include <string_view>

#include "emel/error/error.hpp"
#include "emel/model/data.hpp"

namespace emel::model::sortformer {

struct tensor_view {
  const emel::model::data::tensor_record * tensor = nullptr;
  std::string_view name = {};
};

struct family_view {
  std::string_view prefix = {};
  uint32_t tensor_count = 0;
  tensor_view first = {};
};

struct execution_contract {
  const emel::model::data * model = nullptr;
  int32_t sample_rate = 0;
  int32_t speaker_count = 0;
  int32_t frame_shift_ms = 0;
  int32_t chunk_len = 0;
  int32_t chunk_right_context = 0;
  int32_t fifo_len = 0;
  int32_t spkcache_update_period = 0;
  int32_t spkcache_len = 0;
  family_view feature_extractor = {};
  family_view encoder = {};
  family_view modules = {};
  family_view transformer_encoder = {};
};

emel::error::type build_execution_contract(const emel::model::data & model_data,
                                           execution_contract & contract_out) noexcept;

}  // namespace emel::model::sortformer
