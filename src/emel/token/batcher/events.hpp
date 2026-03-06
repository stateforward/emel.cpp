#pragma once

#include <array>
#include <cstdint>

#include "emel/batch/planner/context.hpp"
#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/token/batcher/errors.hpp"

namespace emel::token::batcher::events {
struct batch_done;
struct batch_error;
}  // namespace emel::token::batcher::events

namespace emel::token::batcher::event {

using position_seed_fn =
    bool (*)(void * position_seed_ctx, int32_t seq_id, int32_t * position_out) noexcept;

struct batch {
  const int32_t & token_ids;
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

  int32_t & seq_primary_ids_out;
  int32_t seq_primary_ids_capacity = 0;
  uint64_t & seq_masks_out;
  int32_t seq_masks_capacity = 0;
  int32_t & positions_out;
  int32_t positions_capacity = 0;
  int8_t & output_mask_out;
  int32_t output_mask_capacity = 0;
  int32_t * outputs_total_out = nullptr;
  int32_t * seq_mask_words_out = nullptr;
  int32_t * positions_count_out = nullptr;
  emel::error::type & error_out;

  emel::callback<void(const events::batch_done &)> on_done = {};
  emel::callback<void(const events::batch_error &)> on_error = {};

  batch(const int32_t & token_ids_in,
        const int32_t n_tokens_in,
        int32_t & seq_primary_ids_out_in,
        const int32_t seq_primary_ids_capacity_in,
        uint64_t & seq_masks_out_in,
        const int32_t seq_masks_capacity_in,
        int32_t & positions_out_in,
        const int32_t positions_capacity_in,
        int8_t & output_mask_out_in,
        const int32_t output_mask_capacity_in,
        emel::error::type & error_out_in) noexcept
      : token_ids(token_ids_in),
        n_tokens(n_tokens_in),
        seq_primary_ids_out(seq_primary_ids_out_in),
        seq_primary_ids_capacity(seq_primary_ids_capacity_in),
        seq_masks_out(seq_masks_out_in),
        seq_masks_capacity(seq_masks_capacity_in),
        positions_out(positions_out_in),
        positions_capacity(positions_capacity_in),
        output_mask_out(output_mask_out_in),
        output_mask_capacity(output_mask_capacity_in),
        error_out(error_out_in) {}
};

struct batch_ctx {
  static constexpr int32_t max_seq = emel::batch::planner::action::MAX_SEQ;

  emel::error::type err = emel::error::cast(error::none);
  int32_t outputs_total = 0;
  int32_t normalized_seq_mask_words = 1;
  int32_t normalized_positions_count = 0;
  std::array<int32_t, static_cast<size_t>(max_seq)> seeded_next_pos = {};
};

struct batch_runtime {
  const batch & request;
  batch_ctx & ctx;
};

}  // namespace emel::token::batcher::event

namespace emel::token::batcher::events {

struct batch_done {
  const event::batch * request = nullptr;
};

struct batch_error {
  emel::error::type err = emel::error::cast(error::none);
  const event::batch * request = nullptr;
};

}  // namespace emel::token::batcher::events
