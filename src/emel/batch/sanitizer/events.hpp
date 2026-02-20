#pragma once

#include <cstdint>

namespace emel::batch::sanitizer::events {
struct sanitize_done;
struct sanitize_error;
}  // namespace emel::batch::sanitizer::events

namespace emel::batch::sanitizer::event {

struct sanitize_decode {
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
};

}  // namespace emel::batch::sanitizer::event

namespace emel::batch::sanitizer::events {

struct sanitize_done {
  const event::sanitize_decode * request = nullptr;
};

struct sanitize_error {
  int32_t err = 0;
  const event::sanitize_decode * request = nullptr;
};

}  // namespace emel::batch::sanitizer::events
