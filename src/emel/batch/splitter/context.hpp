#pragma once

#include <array>
#include <cstdint>

#include "emel/batch/splitter/events.hpp"

namespace emel::batch::splitter::action {

inline constexpr int32_t MAX_UBATCHES = 4096;
inline constexpr int32_t MAX_SEQ = 256;
inline constexpr int32_t SEQ_WORDS = (MAX_SEQ + 63) / 64;

struct context {
  const int32_t * token_ids = nullptr;
  int32_t n_tokens = 0;
  int32_t requested_n_ubatch = 0;
  event::split_mode mode = event::split_mode::simple;
  const uint64_t * seq_masks = nullptr;
  const int32_t * seq_primary_ids = nullptr;
  bool equal_sequential = true;
  int32_t seq_mask_words = 1;
  const int8_t * output_mask = nullptr;
  int32_t seq_masks_count = 0;
  int32_t seq_primary_ids_count = 0;
  int32_t output_mask_count = 0;
  bool output_all = false;

  int32_t effective_n_ubatch = 0;
  std::array<int32_t, MAX_UBATCHES> ubatch_sizes = {};
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;
  std::array<int32_t, MAX_UBATCHES> ubatch_token_indices = {};
  std::array<int32_t, MAX_UBATCHES + 1> ubatch_token_offsets = {};
  int32_t token_indices_count = 0;
};

}  // namespace emel::batch::splitter::action
