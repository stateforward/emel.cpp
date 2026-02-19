#pragma once

#include <cstdint>

#include "emel/callback.hpp"

namespace emel::batch::splitter::events {
struct splitting_done;
struct splitting_error;
}  // namespace emel::batch::splitter::events

namespace emel::batch::splitter::event {

enum class split_mode : int32_t {
  simple = 0,
  equal = 1,
  seq = 2,
};

struct split {
  const int32_t * token_ids = nullptr;
  int32_t n_tokens = 0;
  int32_t n_ubatch = 0;
  split_mode mode = split_mode::simple;
  const uint64_t * seq_masks = nullptr;
  int32_t seq_masks_count = 0;
  const int32_t * seq_primary_ids = nullptr;
  int32_t seq_primary_ids_count = 0;
  bool equal_sequential = true;
  int32_t seq_mask_words = 1;
  const int8_t * output_mask = nullptr;
  int32_t output_mask_count = 0;
  bool output_all = false;

  emel::callback<void(const events::splitting_done &)> on_done = {};
  emel::callback<void(const events::splitting_error &)> on_error = {};
};

}  // namespace emel::batch::splitter::event

namespace emel::batch::splitter::events {

struct splitting_done {
  const event::split * request = nullptr;
  const int32_t * ubatch_sizes = nullptr;
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;
  const int32_t * ubatch_token_indices = nullptr;
  int32_t ubatch_token_indices_count = 0;
  const int32_t * ubatch_token_offsets = nullptr;
  int32_t ubatch_token_offsets_count = 0;
};

struct splitting_error {
  int32_t err = 0;
  const event::split * request = nullptr;
};

}  // namespace emel::batch::splitter::events
