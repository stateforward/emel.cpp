#pragma once

#include <array>
#include <cstdint>

#include "emel/batch/planner/events.hpp"

namespace emel::batch::planner::modes::detail {

using seq_mask_t = std::array<uint64_t, action::SEQ_WORDS>;
using request_ctx = event::request_ctx;

inline seq_mask_t normalized_seq_mask(const event::request & ev, const int32_t idx) noexcept;
inline bool mask_any_set(const seq_mask_t & mask) noexcept;
inline bool mask_overlaps(const seq_mask_t & lhs, const seq_mask_t & rhs) noexcept;
inline bool mask_equal(const seq_mask_t & lhs, const seq_mask_t & rhs) noexcept;
inline bool mask_is_subset(const seq_mask_t & superset, const seq_mask_t & subset) noexcept;
inline bool mask_has_multiple_bits(const seq_mask_t & mask) noexcept;

inline emel::error::type collect_input_errors(const event::request & ev) noexcept {
  emel::error::type mask = emel::error::type{};
  const auto add = [ &mask ](const error code) {
    mask = emel::error::set(mask, code);
  };

  if (ev.token_ids == nullptr) {
    add(error::invalid_token_data);
  }
  if (ev.n_tokens <= 0) {
    add(error::invalid_request);
  }
  if (ev.n_tokens > action::MAX_PLAN_STEPS) {
    add(error::output_plan_full);
  }
  if (ev.seq_mask_words <= 0 || ev.seq_mask_words > action::SEQ_WORDS) {
    add(error::invalid_sequence_metadata);
  }
  if (ev.output_mask != nullptr && ev.output_mask_count < ev.n_tokens) {
    add(error::invalid_sequence_metadata);
  }
  if (ev.seq_masks != nullptr && ev.seq_masks_count < ev.n_tokens) {
    add(error::invalid_sequence_metadata);
  }
  if (ev.seq_primary_ids != nullptr && ev.seq_primary_ids_count < ev.n_tokens) {
    add(error::invalid_sequence_id);
  }

  const bool require_primary_ids =
      ev.mode == event::plan_mode::equal && ev.equal_sequential && ev.seq_masks != nullptr;
  if (require_primary_ids && ev.seq_primary_ids == nullptr) {
    add(error::invalid_sequence_metadata);
  }

  if (ev.n_tokens <= 0) {
    return mask;
  }

  const bool has_masks = ev.seq_masks != nullptr && ev.seq_mask_words > 0 &&
                         ev.seq_mask_words <= action::SEQ_WORDS;
  const bool has_primary_ids = ev.seq_primary_ids != nullptr;
  const int32_t max_seq = ev.seq_mask_words * 64;
  for (int32_t idx = 0; idx < ev.n_tokens; ++idx) {
    if (has_primary_ids) {
      const int32_t primary_id = ev.seq_primary_ids[static_cast<size_t>(idx)];
      if (primary_id < 0 || primary_id >= max_seq) {
        add(error::invalid_sequence_id);
      }
    }
    if (has_masks) {
      const seq_mask_t mask_value = normalized_seq_mask(ev, idx);
      if (!mask_any_set(mask_value)) {
        add(error::invalid_sequence_mask);
      }
      if (ev.mode == event::plan_mode::equal && ev.equal_sequential &&
          mask_has_multiple_bits(mask_value)) {
        add(error::multiple_bits_in_mask);
      }
    }
  }
  return mask;
}

inline bool has_input_errors(const event::request & ev) noexcept {
  return collect_input_errors(ev) != emel::error::type{};
}

inline seq_mask_t normalized_seq_mask(const event::request & ev, const int32_t idx) noexcept {
  seq_mask_t mask = {};
  if (ev.seq_masks != nullptr) {
    const int32_t words = ev.seq_mask_words;
    for (int32_t w = 0; w < words; ++w) {
      mask[static_cast<size_t>(w)] =
          ev.seq_masks[static_cast<size_t>(idx) * static_cast<size_t>(words) +
                       static_cast<size_t>(w)];
    }
    return mask;
  }
  if (ev.seq_primary_ids != nullptr) {
    const uint32_t bit = static_cast<uint32_t>(ev.seq_primary_ids[idx]);
    if (bit < static_cast<uint32_t>(ev.seq_mask_words * 64)) {
      const uint32_t word = bit / 64U;
      const uint32_t shift = bit % 64U;
      mask[static_cast<size_t>(word)] = (uint64_t{1} << shift);
    }
    return mask;
  }
  mask[0] = uint64_t{1};
  return mask;
}

inline bool mask_any_set(const seq_mask_t & mask) noexcept {
  for (const uint64_t word : mask) {
    if (word != 0) {
      return true;
    }
  }
  return false;
}

inline bool mask_overlaps(const seq_mask_t & lhs, const seq_mask_t & rhs) noexcept {
  for (size_t w = 0; w < action::SEQ_WORDS; ++w) {
    if ((lhs[w] & rhs[w]) != 0) {
      return true;
    }
  }
  return false;
}

inline bool mask_equal(const seq_mask_t & lhs, const seq_mask_t & rhs) noexcept {
  for (size_t w = 0; w < action::SEQ_WORDS; ++w) {
    if (lhs[w] != rhs[w]) {
      return false;
    }
  }
  return true;
}

inline bool mask_is_subset(const seq_mask_t & superset, const seq_mask_t & subset) noexcept {
  for (size_t w = 0; w < action::SEQ_WORDS; ++w) {
    if ((superset[w] & subset[w]) != subset[w]) {
      return false;
    }
  }
  return true;
}

inline bool mask_has_multiple_bits(const seq_mask_t & mask) noexcept {
  bool seen = false;
  for (const uint64_t word : mask) {
    if (word == 0) {
      continue;
    }
    if ((word & (word - 1U)) != 0) {
      return true;
    }
    if (seen) {
      return true;
    }
    seen = true;
  }
  return false;
}

inline int32_t count_total_outputs(const event::request & ev) noexcept {
  if (ev.output_all) {
    return ev.n_tokens;
  }
  if (ev.output_mask == nullptr) {
    return ev.n_tokens > 0 ? 1 : 0;
  }
  int32_t total = 0;
  for (int32_t i = 0; i < ev.n_tokens; ++i) {
    total += (ev.output_mask[i] != 0);
  }
  return total;
}

inline bool append_token_index(request_ctx & ctx, const int32_t idx) noexcept {
  if (ctx.token_indices_count >= action::MAX_PLAN_STEPS) {
    return false;
  }
  ctx.step_token_indices[ctx.token_indices_count] = idx;
  ctx.token_indices_count += 1;
  return true;
}

inline bool begin_step(request_ctx & ctx) noexcept {
  if (ctx.step_count >= action::MAX_PLAN_STEPS) {
    return false;
  }
  ctx.step_token_offsets[ctx.step_count] = ctx.token_indices_count;
  return true;
}

inline void finalize_token_offsets(request_ctx & ctx) noexcept {
  if (ctx.step_count <= action::MAX_PLAN_STEPS) {
    ctx.step_token_offsets[ctx.step_count] = ctx.token_indices_count;
  }
}

inline bool push_step_size(request_ctx & ctx, const int32_t size) noexcept {
  if (size <= 0) {
    return false;
  }
  if (ctx.step_count >= action::MAX_PLAN_STEPS) {
    return false;
  }
  ctx.step_sizes[ctx.step_count] = size;
  ctx.step_count += 1;
  return true;
}

inline void clear_plan(request_ctx & ctx) noexcept {
  ctx.step_sizes.fill(0);
  ctx.step_count = 0;
  ctx.total_outputs = 0;
  ctx.step_token_indices.fill(0);
  ctx.token_indices_count = 0;
  ctx.step_token_offsets.fill(0);
}

inline void fail_plan(const event::request_runtime & ev, const error code) noexcept {
  ev.ctx.err = emel::error::set(ev.ctx.err, code);
  clear_plan(ev.ctx);
}

inline void prepare_plan(const event::request_runtime & ev) noexcept {
  clear_plan(ev.ctx);
  ev.ctx.total_outputs = count_total_outputs(ev.request);
}

}  // namespace emel::batch::planner::modes::detail
