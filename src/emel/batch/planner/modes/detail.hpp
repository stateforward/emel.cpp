#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
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
inline void finalize_token_offsets(request_ctx & ctx) noexcept;
inline void fail_plan(const event::request_runtime & ev, const error code) noexcept;

inline int32_t select_i32(const bool choose_true,
                          const int32_t true_value,
                          const int32_t false_value) noexcept {
  const int32_t mask = -static_cast<int32_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline uint32_t select_u32(const bool choose_true,
                           const uint32_t true_value,
                           const uint32_t false_value) noexcept {
  const uint32_t mask = static_cast<uint32_t>(0) - static_cast<uint32_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline uint64_t select_u64(const bool choose_true,
                           const uint64_t true_value,
                           const uint64_t false_value) noexcept {
  const uint64_t mask = static_cast<uint64_t>(0) - static_cast<uint64_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline size_t select_size(const bool choose_true,
                          const size_t true_value,
                          const size_t false_value) noexcept {
  const size_t mask = static_cast<size_t>(0) - static_cast<size_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline uint8_t select_u8(const bool choose_true,
                         const uint8_t true_value,
                         const uint8_t false_value) noexcept {
  const uint8_t mask = static_cast<uint8_t>(0) - static_cast<uint8_t>(choose_true);
  return static_cast<uint8_t>((false_value & static_cast<uint8_t>(~mask)) |
                              (true_value & mask));
}

inline emel::error::type select_error(const bool choose_true,
                                      const emel::error::type true_value,
                                      const emel::error::type false_value) noexcept {
  const emel::error::type mask = static_cast<emel::error::type>(0) -
      static_cast<emel::error::type>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

template <class value_type>
inline value_type * pick_ptr(const bool choose_true,
                             value_type * true_value,
                             value_type * false_value) noexcept {
  value_type * values[2] = {false_value, true_value};
  return values[static_cast<size_t>(choose_true)];
}

inline void copy_mask_if(seq_mask_t & destination,
                         const seq_mask_t & source,
                         const bool copy_source) noexcept {
  for (size_t w = 0; w < action::SEQ_WORDS; ++w) {
    destination[w] = select_u64(copy_source, source[w], destination[w]);
  }
}

inline void add_error_if(emel::error::type & mask,
                         const bool condition,
                         const error code) noexcept {
  const emel::error::type next = emel::error::set(mask, code);
  mask = select_error(condition, next, mask);
}

inline void fail_noop(const event::request_runtime &, const error) noexcept {
}

inline void fail_apply(const event::request_runtime & ev, const error code) noexcept {
  fail_plan(ev, code);
}

inline void fail_if(const bool condition,
                    bool & failed,
                    const event::request_runtime & ev,
                    const error code) noexcept {
  using fail_handler = void (*)(const event::request_runtime &, error) noexcept;
  const bool trigger = condition && !failed;
  const std::array<fail_handler, 2> handlers = {&fail_noop, &fail_apply};
  handlers[static_cast<size_t>(trigger)](ev, code);
  failed = failed || trigger;
}

inline void finalize_offsets_noop(request_ctx &) noexcept {
}

inline void finalize_offsets_apply(request_ctx & ctx) noexcept {
  finalize_token_offsets(ctx);
}

inline void finalize_offsets_if_success(request_ctx & ctx, const bool failed) noexcept {
  using finalize_handler = void (*)(request_ctx &) noexcept;
  const std::array<finalize_handler, 2> handlers = {
      &finalize_offsets_noop,
      &finalize_offsets_apply,
  };
  handlers[static_cast<size_t>(!failed)](ctx);
}

inline emel::error::type collect_input_errors(const event::request & ev) noexcept {
  emel::error::type mask = emel::error::type{};

  add_error_if(mask, ev.token_ids == nullptr, error::invalid_token_data);
  add_error_if(mask, ev.n_tokens <= 0, error::invalid_request);
  add_error_if(mask, ev.n_tokens > action::MAX_PLAN_STEPS, error::output_plan_full);
  add_error_if(mask,
               ev.seq_mask_words <= 0 || ev.seq_mask_words > action::SEQ_WORDS,
               error::invalid_sequence_metadata);
  add_error_if(mask,
               ev.output_mask != nullptr && ev.output_mask_count < ev.n_tokens,
               error::invalid_sequence_metadata);
  add_error_if(mask,
               ev.seq_masks != nullptr && ev.seq_masks_count < ev.n_tokens,
               error::invalid_sequence_metadata);
  add_error_if(mask,
               ev.seq_primary_ids != nullptr && ev.seq_primary_ids_count < ev.n_tokens,
               error::invalid_sequence_id);

  const bool require_primary_ids =
      ev.mode == event::plan_mode::equal && ev.equal_sequential && ev.seq_masks != nullptr;
  add_error_if(mask,
               require_primary_ids && ev.seq_primary_ids == nullptr,
               error::invalid_sequence_metadata);

  const bool valid_words = ev.seq_mask_words > 0 && ev.seq_mask_words <= action::SEQ_WORDS;
  const bool has_masks = ev.seq_masks != nullptr && valid_words;
  const bool has_primary_ids = ev.seq_primary_ids != nullptr;
  const int32_t max_seq = ev.seq_mask_words * 64;
  const int32_t tokens_to_scan = select_i32(ev.n_tokens > 0, ev.n_tokens, 0);

  const int32_t primary_sink = 0;
  const std::array<const int32_t *, 2> primary_ptrs = {&primary_sink, ev.seq_primary_ids};

  for (int32_t idx = 0; idx < tokens_to_scan; ++idx) {
    const bool read_primary = has_primary_ids;
    const int32_t primary_index = select_i32(read_primary, idx, 0);
    const int32_t primary_id = primary_ptrs[static_cast<size_t>(read_primary)]
        [static_cast<size_t>(primary_index)];
    add_error_if(mask,
                 read_primary && (primary_id < 0 || primary_id >= max_seq),
                 error::invalid_sequence_id);

    const bool read_mask = has_masks;
    const seq_mask_t mask_value = normalized_seq_mask(ev, select_i32(read_mask, idx, 0));
    add_error_if(mask,
                 read_mask && !mask_any_set(mask_value),
                 error::invalid_sequence_mask);
    add_error_if(mask,
                 read_mask && ev.mode == event::plan_mode::equal && ev.equal_sequential &&
                     mask_has_multiple_bits(mask_value),
                 error::multiple_bits_in_mask);
  }

  return mask;
}

inline bool has_input_errors(const event::request & ev) noexcept {
  return collect_input_errors(ev) != emel::error::type{};
}

inline seq_mask_t normalized_seq_mask(const event::request & ev, const int32_t idx) noexcept {
  seq_mask_t mask = {};

  const bool valid_words = ev.seq_mask_words > 0 && ev.seq_mask_words <= action::SEQ_WORDS;
  const int32_t words = select_i32(valid_words, ev.seq_mask_words, 0);

  const bool has_masks = ev.seq_masks != nullptr;
  const bool has_primary = ev.seq_primary_ids != nullptr;
  const bool valid_index = idx >= 0;

  const bool use_masks = has_masks && valid_words && valid_index;
  const bool use_primary = !use_masks && has_primary && valid_index;

  const uint64_t mask_sink = 0;
  const std::array<const uint64_t *, 2> mask_ptrs = {&mask_sink, ev.seq_masks};
  const int32_t row_index = select_i32(use_masks, idx, 0);
  const size_t row_base = static_cast<size_t>(row_index) * static_cast<size_t>(words);

  for (int32_t w = 0; w < words; ++w) {
    const size_t read_offset = row_base + static_cast<size_t>(w);
    const size_t safe_offset = select_size(use_masks, read_offset, 0u);
    const uint64_t value = mask_ptrs[static_cast<size_t>(use_masks)][safe_offset];
    mask[static_cast<size_t>(w)] = select_u64(use_masks, value, mask[static_cast<size_t>(w)]);
  }

  const int32_t primary_sink = 0;
  const std::array<const int32_t *, 2> primary_ptrs = {&primary_sink, ev.seq_primary_ids};
  const int32_t primary_idx = select_i32(use_primary, idx, 0);
  const int32_t primary_id = primary_ptrs[static_cast<size_t>(use_primary)]
      [static_cast<size_t>(primary_idx)];

  const bool valid_primary_id = use_primary && primary_id >= 0 &&
      static_cast<uint32_t>(primary_id) < static_cast<uint32_t>(words * 64);
  const uint32_t bit = static_cast<uint32_t>(primary_id);
  const uint32_t word = bit / 64u;
  const uint32_t shift = bit % 64u;
  const size_t word_index = static_cast<size_t>(select_u32(valid_primary_id, word, 0u));
  const uint64_t bit_mask = uint64_t{1} << shift;
  mask[word_index] = select_u64(valid_primary_id, bit_mask, mask[word_index]);

  mask[0] = select_u64(!use_masks && !use_primary, uint64_t{1}, mask[0]);
  return mask;
}

inline bool mask_any_set(const seq_mask_t & mask) noexcept {
  uint8_t any = 0;
  for (const uint64_t word : mask) {
    any |= static_cast<uint8_t>(word != 0);
  }
  return any != 0;
}

inline bool mask_overlaps(const seq_mask_t & lhs, const seq_mask_t & rhs) noexcept {
  uint8_t overlaps = 0;
  for (size_t w = 0; w < action::SEQ_WORDS; ++w) {
    overlaps |= static_cast<uint8_t>((lhs[w] & rhs[w]) != 0);
  }
  return overlaps != 0;
}

inline bool mask_equal(const seq_mask_t & lhs, const seq_mask_t & rhs) noexcept {
  uint8_t mismatch = 0;
  for (size_t w = 0; w < action::SEQ_WORDS; ++w) {
    mismatch |= static_cast<uint8_t>(lhs[w] != rhs[w]);
  }
  return mismatch == 0;
}

inline bool mask_is_subset(const seq_mask_t & superset, const seq_mask_t & subset) noexcept {
  uint8_t violates_subset = 0;
  for (size_t w = 0; w < action::SEQ_WORDS; ++w) {
    violates_subset |= static_cast<uint8_t>((superset[w] & subset[w]) != subset[w]);
  }
  return violates_subset == 0;
}

inline bool mask_has_multiple_bits(const seq_mask_t & mask) noexcept {
  bool seen = false;
  bool multiple = false;
  for (const uint64_t word : mask) {
    const bool has_any = word != 0;
    const bool has_multiple_in_word = has_any && ((word & (word - 1U)) != 0);
    const bool repeats_single_bit = has_any && seen;
    multiple = multiple || has_multiple_in_word || repeats_single_bit;
    seen = seen || has_any;
  }
  return multiple;
}

inline int32_t count_total_outputs(const event::request & ev) noexcept {
  const bool has_output_mask = ev.output_mask != nullptr;
  const int8_t output_mask_sink = 0;
  const std::array<const int8_t *, 2> output_mask_ptrs = {&output_mask_sink, ev.output_mask};

  int32_t masked_total = 0;
  const int32_t mask_scan_count = select_i32(has_output_mask, ev.n_tokens, 0);
  for (int32_t i = 0; i < mask_scan_count; ++i) {
    masked_total += (output_mask_ptrs[static_cast<size_t>(has_output_mask)]
        [static_cast<size_t>(i)] != 0);
  }

  const std::array<int32_t, 2> single_output_counts = {0, 1};
  const int32_t fallback_total = single_output_counts[static_cast<size_t>(ev.n_tokens > 0)];
  const int32_t non_all_total = select_i32(has_output_mask, masked_total, fallback_total);
  return select_i32(ev.output_all, ev.n_tokens, non_all_total);
}

inline bool append_token_index(request_ctx & ctx, const int32_t idx) noexcept {
  const bool has_space = ctx.token_indices_count < action::MAX_PLAN_STEPS;
  const int32_t write_index = select_i32(has_space, ctx.token_indices_count, 0);
  ctx.step_token_indices[static_cast<size_t>(write_index)] =
      select_i32(has_space,
                 idx,
                 ctx.step_token_indices[static_cast<size_t>(write_index)]);
  ctx.token_indices_count += static_cast<int32_t>(has_space);
  return has_space;
}

inline bool begin_step(request_ctx & ctx) noexcept {
  const bool has_space = ctx.step_count < action::MAX_PLAN_STEPS;
  const int32_t write_index = select_i32(has_space, ctx.step_count, 0);
  ctx.step_token_offsets[static_cast<size_t>(write_index)] =
      select_i32(has_space,
                 ctx.token_indices_count,
                 ctx.step_token_offsets[static_cast<size_t>(write_index)]);
  return has_space;
}

inline void finalize_token_offsets(request_ctx & ctx) noexcept {
  const bool valid_index = ctx.step_count >= 0 && ctx.step_count <= action::MAX_PLAN_STEPS;
  const int32_t write_index = select_i32(valid_index, ctx.step_count, 0);
  ctx.step_token_offsets[static_cast<size_t>(write_index)] =
      select_i32(valid_index,
                 ctx.token_indices_count,
                 ctx.step_token_offsets[static_cast<size_t>(write_index)]);
}

inline bool push_step_size(request_ctx & ctx, const int32_t size) noexcept {
  const bool valid_size = size > 0;
  const bool has_space = ctx.step_count < action::MAX_PLAN_STEPS;
  const bool can_push = valid_size && has_space;

  const int32_t write_index = select_i32(can_push, ctx.step_count, 0);
  ctx.step_sizes[static_cast<size_t>(write_index)] =
      select_i32(can_push,
                 size,
                 ctx.step_sizes[static_cast<size_t>(write_index)]);
  ctx.step_count += static_cast<int32_t>(can_push);
  return can_push;
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
