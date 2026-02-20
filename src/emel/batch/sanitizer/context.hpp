#pragma once

#include <array>
#include <cstdint>

#include "emel/batch/sanitizer/events.hpp"
#include "emel/batch/splitter/context.hpp"
#include "emel/emel.h"

namespace emel::batch::sanitizer::action {

inline constexpr int32_t MAX_TOKENS = emel::batch::splitter::action::MAX_UBATCHES;
inline constexpr int32_t MAX_SEQ = emel::batch::splitter::action::MAX_SEQ;
inline constexpr int32_t SEQ_WORDS = emel::batch::splitter::action::SEQ_WORDS;

struct context {
  const int32_t * token_ids = nullptr;
  int32_t n_tokens = 0;

  const uint64_t * seq_masks = nullptr;
  int32_t seq_mask_words = 1;
  int32_t seq_masks_count = 0;
  const int32_t * seq_primary_ids = nullptr;
  int32_t seq_primary_ids_count = 0;

  const int32_t * positions = nullptr;
  int32_t positions_count = 0;

  const int8_t * output_mask = nullptr;
  int32_t output_mask_count = 0;
  bool output_all = false;
  bool enforce_single_output_per_seq = false;

  int32_t * seq_primary_ids_out = nullptr;
  int32_t seq_primary_ids_capacity = 0;
  uint64_t * seq_masks_out = nullptr;
  int32_t seq_masks_capacity = 0;
  int32_t * positions_out = nullptr;
  int32_t positions_capacity = 0;
  int8_t * output_mask_out = nullptr;
  int32_t output_mask_capacity = 0;
  int32_t * outputs_total_out = nullptr;
  int32_t * seq_mask_words_out = nullptr;
  int32_t * positions_count_out = nullptr;
  int32_t * error_out = nullptr;

  int32_t outputs_total = 0;
  int32_t normalized_seq_mask_words = 1;
  int32_t normalized_positions_count = 0;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::batch::sanitizer::action
