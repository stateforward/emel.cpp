#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/error/error.hpp"
#include "emel/model/data.hpp"
#include "emel/model/loader/detail.hpp"

// Moshi-family model binding (Kyutai Moshi / NVIDIA PersonaPlex).
//
// A Moshi deployment is split across three GGUF components, discriminated by
// the required `moshi.component` metadata key:
//   - "lm"    : temporal transformer + depformer weights (`lm.*` tensors)
//   - "mimi"  : Mimi streaming codec weights (`mimi.*` tensors)
//   - "voice" : PersonaPlex voice-prompt tensors (`voice.embeddings`,
//               `voice.cache`)
//
// emel only accepts *enriched* GGUFs produced by
// `tools/bench/moshi_gguf_convert.py`. Raw moshi.cpp GGUF caches carry zero
// metadata and are rejected by the generic loader path (missing
// `general.architecture`); this binding additionally requires the full
// `moshi.*` hparam contract below.
namespace emel::model::moshi::detail {

struct tensor_view {
  const emel::model::data::tensor_record *tensor = nullptr;
  std::string_view name = {};
};

struct family_view {
  std::string_view prefix = {};
  uint32_t tensor_count = 0;
  tensor_view first = {};
};

struct lm_contract {
  family_view text_emb = {};
  family_view audio_emb = {};
  family_view transformer = {};
  family_view out_norm = {};
  family_view text_linear = {};
  family_view linears = {};
  family_view depformer_in = {};
  family_view depformer = {};
  family_view depformer_text_emb = {};
  family_view depformer_emb = {};
  tensor_view text_embedding = {};
  tensor_view output_norm = {};
  tensor_view text_output_projection = {};
  std::array<tensor_view,
             static_cast<std::size_t>(
                 emel::model::data::moshi_lm_hparams::k_max_delays)>
      audio_embeddings = {};
};

struct mimi_contract {
  family_view encoder = {};
  family_view encoder_transformer = {};
  family_view downsample = {};
  family_view quantizer = {};
  family_view upsample = {};
  family_view decoder_transformer = {};
  family_view decoder = {};
};

struct voice_contract {
  tensor_view embeddings = {};
  tensor_view cache = {};
};

struct execution_contract {
  const emel::model::data *model = nullptr;
  emel::model::data::moshi_component component =
      emel::model::data::moshi_component::none;
  lm_contract lm = {};
  mimi_contract mimi = {};
  voice_contract voice = {};
};

bool is_execution_architecture(std::string_view architecture) noexcept;

bool load_hparams(const emel::model::detail::hparam_loader &loader,
                  emel::model::data &model_out) noexcept;

emel::error::type
build_execution_contract(const emel::model::data &model_data,
                         execution_contract &contract_out) noexcept;

emel::error::type validate_data(const emel::model::data &model_data) noexcept;

emel::error::type
validate_execution_contract(const emel::model::data &model_data) noexcept;

} // namespace emel::model::moshi::detail
