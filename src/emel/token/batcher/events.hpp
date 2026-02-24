#pragma once

#include <cstdint>

#include "emel/callback.hpp"

namespace emel::token::batcher::events {
struct batch_done;
struct batch_error;
}  // namespace emel::token::batcher::events

namespace emel::token::batcher::event {

using position_seed_fn =
    bool (*)(void * position_seed_ctx, int32_t seq_id, int32_t * position_out) noexcept;

struct batch {
  const int32_t * token_ids = nullptr;
  int32_t n_tokens = 0;
  int32_t vocab_size = 0;

  const uint64_t * seq_masks = nullptr;
  int32_t seq_mask_words = 1;
  int32_t seq_masks_count = 0;
  const int32_t * seq_primary_ids = nullptr;
  int32_t seq_primary_ids_count = 0;

  const int32_t * positions = nullptr;
  int32_t positions_count = 0;
  void * position_seed_ctx = nullptr;
  position_seed_fn resolve_position_seed = nullptr;

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

  emel::callback<void(const events::batch_done &)> on_done = {};
  emel::callback<void(const events::batch_error &)> on_error = {};
};

}  // namespace emel::token::batcher::event

namespace emel::token::batcher::events {

struct batch_done {
  const event::batch * request = nullptr;
};

struct batch_error {
  int32_t err = 0;
  const event::batch * request = nullptr;
};

}  // namespace emel::token::batcher::events
