#pragma once

#include <array>
#include <cstddef>

#include "emel/text/encoders/context.hpp"
#include "emel/text/encoders/types.hpp"

namespace emel::text::encoders::ugm::action {

struct best_tokenization {
  int32_t token_id = 0;
  uint32_t input_offset = 0;
  double score_sum = 0.0;
};

struct context : emel::text::encoders::action::context {
  emel::text::encoders::detail::naive_trie token_matcher = {};
  emel::text::encoders::detail::naive_trie user_defined_token_matcher = {};
  const uint32_t *xcda_table = nullptr;
  size_t xcda_table_size = 0;
  const char *prefix_replacements = nullptr;
  size_t prefix_replacements_size = 0;
  bool ugm_tables_ready = false;
  const emel::model::data::vocab *ugm_vocab = nullptr;
  float min_score = 0.0f;
  float max_score = 0.0f;
  float unknown_token_score_penalty = 10.0f;
  float unknown_token_score = 0.0f;
  std::array<best_tokenization, emel::text::encoders::detail::k_max_encode_bytes + 1> best = {};
  std::array<int32_t, emel::text::encoders::detail::k_max_encode_symbols> token_buffer = {};
};

}  // namespace emel::text::encoders::ugm::action
