#pragma once

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/model/loader/detail.hpp"

namespace emel::model::whisper::detail {

struct tensor_view {
  const emel::model::data::tensor_record *tensor = nullptr;
  std::string_view name = {};
};

struct family_view {
  std::string_view prefix = {};
  uint32_t tensor_count = 0;
  tensor_view first = {};
};

struct execution_contract {
  const emel::model::data *model = nullptr;
  int32_t sample_rate = 0;
  int32_t mel_bin_count = 0;
  int32_t vocab_size = 0;
  int32_t embedding_length = 0;
  int32_t feed_forward_length = 0;
  int32_t attention_head_count = 0;
  int32_t encoder_context_length = 0;
  int32_t decoder_context_length = 0;
  int32_t encoder_block_count = 0;
  int32_t decoder_block_count = 0;
  family_view mel_filters = {};
  family_view encoder = {};
  family_view decoder = {};
};

bool is_execution_architecture(std::string_view architecture) noexcept;

bool is_legacy_lmgg_whisper(std::span<const uint8_t> source) noexcept;

bool normalize_legacy_lmgg_to_gguf(std::span<const uint8_t> source,
                                   std::vector<uint8_t> &gguf_out);

bool load_hparams(const emel::model::detail::hparam_loader &loader,
                  emel::model::data &model_out) noexcept;

emel::error::type
build_execution_contract(const emel::model::data &model_data,
                         execution_contract &contract_out) noexcept;

emel::error::type validate_data(const emel::model::data &model_data) noexcept;

emel::error::type
validate_execution_contract(const emel::model::data &model_data) noexcept;

} // namespace emel::model::whisper::detail
