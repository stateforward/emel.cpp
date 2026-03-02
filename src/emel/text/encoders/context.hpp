#pragma once

#include "emel/emel.h"
#include "emel/text/encoders/types.hpp"
#include "emel/model/data.hpp"

namespace emel::text::encoders::action {

struct context {
  const emel::model::data::vocab * vocab = nullptr;
  bool tables_ready = false;
  bool ugm_ready = false;

  int32_t max_token_len = 0;
  detail::token_map token_to_id = {};
  detail::merge_map bpe_ranks = {};
  detail::encode_scratch scratch = {};
};

}  // namespace emel::text::encoders::action
