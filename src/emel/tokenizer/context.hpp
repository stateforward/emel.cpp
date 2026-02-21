#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/encoder/bpe/context.hpp"
#include "emel/encoder/bpe/sm.hpp"
#include "emel/encoder/events.hpp"
#include "emel/encoder/fallback/context.hpp"
#include "emel/encoder/fallback/sm.hpp"
#include "emel/encoder/plamo2/context.hpp"
#include "emel/encoder/plamo2/sm.hpp"
#include "emel/encoder/rwkv/context.hpp"
#include "emel/encoder/rwkv/sm.hpp"
#include "emel/encoder/spm/context.hpp"
#include "emel/encoder/spm/sm.hpp"
#include "emel/encoder/ugm/context.hpp"
#include "emel/encoder/ugm/sm.hpp"
#include "emel/encoder/wpm/context.hpp"
#include "emel/encoder/wpm/sm.hpp"
#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/tokenizer/events.hpp"
#include "emel/tokenizer/preprocessor/types.hpp"

namespace emel::tokenizer::action {

constexpr size_t k_max_fragments =
    emel::tokenizer::preprocessor::k_max_fragments;
constexpr size_t k_max_special_tokens =
    emel::tokenizer::preprocessor::k_max_special_tokens;
constexpr size_t k_encoder_map_size = 8;

enum class encoder_slot : uint8_t {
  none = 0,
  spm = 1,
  bpe = 2,
  wpm = 3,
  ugm = 4,
  rwkv = 5,
  plamo2 = 6,
  fallback = 7,
};

struct encoder_entry {
  void *handle = nullptr;
  bool (*process)(void *handle,
                  const emel::encoder::event::encode &ev) = nullptr;
};

using fragment_kind = emel::tokenizer::preprocessor::fragment_kind;
using fragment = emel::tokenizer::preprocessor::fragment;
using special_token = emel::tokenizer::preprocessor::special_token;
using special_token_cache = emel::tokenizer::preprocessor::special_token_cache;

struct context {
  emel::encoder::bpe::action::context bpe_ctx = {};
  emel::encoder::spm::action::context spm_ctx = {};
  emel::encoder::wpm::action::context wpm_ctx = {};
  emel::encoder::ugm::action::context ugm_ctx = {};
  emel::encoder::rwkv::action::context rwkv_ctx = {};
  emel::encoder::plamo2::action::context plamo2_ctx = {};
  emel::encoder::fallback::action::context fallback_ctx = {};

  emel::encoder::bpe::sm bpe_encoder;
  emel::encoder::spm::sm spm_encoder;
  emel::encoder::wpm::sm wpm_encoder;
  emel::encoder::ugm::sm ugm_encoder;
  emel::encoder::rwkv::sm rwkv_encoder;
  emel::encoder::plamo2::sm plamo2_encoder;
  emel::encoder::fallback::sm fallback_encoder;

  std::array<encoder_entry, k_encoder_map_size> encoder_map = {};
  encoder_entry *active_encoder = nullptr;
  std::array<fragment, k_max_fragments> fragments = {};
  size_t fragment_count = 0;
  size_t fragment_index = 0;
  special_token_cache special_cache = {};
  const emel::model::data::vocab *vocab = nullptr;
  std::string_view text = {};
  bool add_special = false;
  bool parse_special = false;
  int32_t *token_ids_out = nullptr;
  int32_t token_capacity = 0;
  encoder_slot model_slot = encoder_slot::none;
  int32_t token_count = 0;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;

  context();
};

} // namespace emel::tokenizer::action
