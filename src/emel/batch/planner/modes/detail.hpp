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

inline void create_simple_plan(const event::request_runtime & ev) noexcept {
  bool failed = false;
  fail_if(ev.ctx.effective_step_size <= 0,
          failed,
          ev,
          emel::batch::planner::error::invalid_step_size);

  int32_t next_token = 0;
  while (next_token < ev.request.n_tokens && !failed) {
    const bool active_chunk = !failed;
    fail_if(active_chunk && !begin_step(ev.ctx),
            failed,
            ev,
            emel::batch::planner::error::output_steps_full);

    const int32_t chunk_raw =
        std::min<int32_t>(ev.ctx.effective_step_size, ev.request.n_tokens - next_token);
    const int32_t chunk = select_i32(active_chunk && !failed, chunk_raw, 0);

    for (int32_t i = 0; i < chunk && !failed; ++i) {
      fail_if(!append_token_index(ev.ctx, next_token + i),
              failed,
              ev,
              emel::batch::planner::error::output_indices_full);
    }

    next_token += chunk;
    fail_if(!failed && !push_step_size(ev.ctx, chunk),
            failed,
            ev,
            emel::batch::planner::error::output_steps_full);
  }

  finalize_offsets_if_success(ev.ctx, failed);
}

inline void create_sequential_plan(const event::request_runtime & ev) noexcept {
  bool failed = false;
  fail_if(ev.ctx.effective_step_size <= 0,
          failed,
          ev,
          emel::batch::planner::error::invalid_step_size);

  std::array<uint8_t, emel::batch::planner::action::MAX_PLAN_STEPS> used = {};
  int32_t used_count = 0;
  bool done = false;

  while (used_count < ev.request.n_tokens && !done && !failed) {
    int32_t cur_idx = 0;
    while (cur_idx < ev.request.n_tokens && used[static_cast<size_t>(cur_idx)] != 0) {
      ++cur_idx;
    }

    const bool exhausted = cur_idx >= ev.request.n_tokens;
    done = done || exhausted;
    const bool process = !done && !failed;

    int32_t chunk = 0;
    seq_mask_t cur_mask = normalized_seq_mask(ev.request, select_i32(process, cur_idx, 0));
    fail_if(process && !begin_step(ev.ctx),
            failed,
            ev,
            emel::batch::planner::error::output_steps_full);

    bool continue_chunk = process && !failed;
    while (continue_chunk && !failed) {
      used[static_cast<size_t>(cur_idx)] = 1;
      used_count += 1;
      chunk += 1;

      fail_if(!append_token_index(ev.ctx, cur_idx),
              failed,
              ev,
              emel::batch::planner::error::output_indices_full);

      const bool reached_step_size = chunk >= ev.ctx.effective_step_size;
      const bool scan_next = !failed && !reached_step_size;

      int32_t next_idx = cur_idx + 1;
      while (scan_next && next_idx < ev.request.n_tokens &&
             (used[static_cast<size_t>(next_idx)] != 0 ||
              !mask_is_subset(cur_mask, normalized_seq_mask(ev.request, next_idx)))) {
        ++next_idx;
      }

      const bool has_candidate = scan_next && next_idx < ev.request.n_tokens;
      const seq_mask_t next_mask = normalized_seq_mask(
          ev.request,
          select_i32(has_candidate, next_idx, 0));

      cur_idx = select_i32(has_candidate, next_idx, cur_idx);
      copy_mask_if(cur_mask, next_mask, has_candidate);

      continue_chunk = continue_chunk && !failed && !reached_step_size && has_candidate;
    }

    fail_if(!failed && !push_step_size(ev.ctx, chunk),
            failed,
            ev,
            emel::batch::planner::error::output_steps_full);
  }

  finalize_offsets_if_success(ev.ctx, failed);
}

inline void create_equal_plan(const event::request_runtime & ev) noexcept {
  bool failed = false;
  fail_if(ev.ctx.effective_step_size <= 0,
          failed,
          ev,
          emel::batch::planner::error::invalid_step_size);

  std::array<uint8_t, emel::batch::planner::action::MAX_PLAN_STEPS> used = {};
  int32_t used_count = 0;

  const int32_t primary_sink = 0;
  const std::array<const int32_t *, 2> primary_ptrs = {&primary_sink, ev.request.seq_primary_ids};

  while (used_count < ev.request.n_tokens && !failed) {
    struct group_state {
      seq_mask_t mask = {};
    };

    std::array<group_state, emel::batch::planner::action::MAX_PLAN_STEPS> groups = {};
    int32_t group_count = 0;
    int32_t last_primary = -1;
    bool stop_group_scan = false;

    for (int32_t i = 0; i < ev.request.n_tokens && !stop_group_scan && !failed; ++i) {
      const bool is_unused = used[static_cast<size_t>(i)] == 0;
      const seq_mask_t mask = normalized_seq_mask(ev.request, i);

      bool overlap = false;
      for (int32_t g = 0; g < group_count; ++g) {
        overlap = overlap || mask_overlaps(groups[static_cast<size_t>(g)].mask, mask);
      }

      const bool requires_sequential_primary =
          ev.request.equal_sequential && ev.request.seq_primary_ids != nullptr;
      const int32_t primary_read = primary_ptrs[static_cast<size_t>(requires_sequential_primary)]
          [static_cast<size_t>(select_i32(requires_sequential_primary, i, 0))];
      const int32_t primary = select_i32(requires_sequential_primary, primary_read, last_primary);

      const bool out_of_order =
          requires_sequential_primary && group_count > 0 && primary != last_primary + 1;
      const bool can_add_group = is_unused && !overlap && !out_of_order;

      const int32_t group_index = select_i32(can_add_group, group_count, 0);
      copy_mask_if(groups[static_cast<size_t>(group_index)].mask, mask, can_add_group);

      group_count += static_cast<int32_t>(can_add_group);
      last_primary = select_i32(can_add_group && requires_sequential_primary,
                                primary,
                                last_primary);
      stop_group_scan = stop_group_scan ||
          (can_add_group && group_count > ev.ctx.effective_step_size);
    }

    fail_if(group_count == 0,
            failed,
            ev,
            emel::batch::planner::error::planning_progress_stalled);

    int32_t min_avail = ev.request.n_tokens + 1;
    for (int32_t g = 0; g < group_count; ++g) {
      int32_t avail = 0;
      for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
        const bool available =
            used[static_cast<size_t>(i)] == 0 &&
            mask_equal(normalized_seq_mask(ev.request, i),
                       groups[static_cast<size_t>(g)].mask);
        avail += static_cast<int32_t>(available);
      }
      min_avail = std::min(min_avail, avail);
    }

    const int32_t safe_group_count = select_i32(group_count > 0, group_count, 1);
    const int32_t max_rows = ev.ctx.effective_step_size / safe_group_count;
    const int32_t n_seq_tokens = std::min(max_rows, min_avail);

    fail_if(n_seq_tokens <= 0,
            failed,
            ev,
            emel::batch::planner::error::planning_progress_stalled);

    fail_if(!failed && !begin_step(ev.ctx),
            failed,
            ev,
            emel::batch::planner::error::output_steps_full);

    for (int32_t g = 0; g < group_count && !failed; ++g) {
      int32_t remaining = n_seq_tokens;
      for (int32_t i = 0; i < ev.request.n_tokens && remaining > 0 && !failed; ++i) {
        const bool match =
            used[static_cast<size_t>(i)] == 0 &&
            mask_equal(normalized_seq_mask(ev.request, i),
                       groups[static_cast<size_t>(g)].mask);

        used[static_cast<size_t>(i)] =
            select_u8(match, 1u, used[static_cast<size_t>(i)]);
        used_count += static_cast<int32_t>(match);

        fail_if(match && !append_token_index(ev.ctx, i),
                failed,
                ev,
                emel::batch::planner::error::output_indices_full);

        remaining -= static_cast<int32_t>(match);
      }

      fail_if(!failed && remaining != 0,
              failed,
              ev,
              emel::batch::planner::error::algorithm_failed);
    }

    const int32_t added = n_seq_tokens * group_count;
    fail_if(!failed && !push_step_size(ev.ctx, added),
            failed,
            ev,
            emel::batch::planner::error::output_steps_full);
  }

  finalize_offsets_if_success(ev.ctx, failed);
}

inline void create_equal_plan_primary_fast_path(const event::request_runtime & ev) noexcept {
  bool failed = false;
  fail_if(ev.ctx.effective_step_size <= 0,
          failed,
          ev,
          emel::batch::planner::error::invalid_step_size);
  fail_if(ev.request.seq_primary_ids == nullptr,
          failed,
          ev,
          emel::batch::planner::error::invalid_sequence_id);

  const int32_t max_seq = ev.request.seq_mask_words * 64;
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ> seq_counts = {};
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ + 1> seq_offsets = {};
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ> seq_used = {};
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ> seq_cursor = {};
  std::array<int32_t, emel::batch::planner::action::MAX_PLAN_STEPS> seq_indices = {};

  for (int32_t i = 0; i < ev.request.n_tokens && !failed; ++i) {
    const int32_t seq_id = ev.request.seq_primary_ids[i];
    const bool valid_seq = seq_id >= 0 && seq_id < max_seq;
    fail_if(!valid_seq,
            failed,
            ev,
            emel::batch::planner::error::invalid_sequence_id);

    const size_t slot = static_cast<size_t>(select_i32(valid_seq, seq_id, 0));
    seq_counts[slot] += static_cast<int32_t>(valid_seq);
  }

  for (int32_t s = 0; s < max_seq; ++s) {
    seq_offsets[static_cast<size_t>(s + 1)] =
        seq_offsets[static_cast<size_t>(s)] + seq_counts[static_cast<size_t>(s)];
    seq_cursor[static_cast<size_t>(s)] = seq_offsets[static_cast<size_t>(s)];
  }

  for (int32_t i = 0; i < ev.request.n_tokens && !failed; ++i) {
    const int32_t seq_id = ev.request.seq_primary_ids[i];
    const bool valid_seq = seq_id >= 0 && seq_id < max_seq;
    const size_t slot = static_cast<size_t>(select_i32(valid_seq, seq_id, 0));

    const int32_t pos = seq_cursor[slot];
    const bool valid_pos = pos >= 0 && pos < ev.request.n_tokens;
    fail_if(!valid_pos,
            failed,
            ev,
            emel::batch::planner::error::algorithm_failed);

    const size_t write_pos = static_cast<size_t>(select_i32(valid_pos, pos, 0));
    seq_indices[write_pos] = select_i32(valid_pos, i, seq_indices[write_pos]);
    seq_cursor[slot] = pos + static_cast<int32_t>(valid_pos);
  }

  int32_t remaining = ev.request.n_tokens;
  while (remaining > 0 && !failed) {
    std::array<uint8_t, emel::batch::planner::action::MAX_SEQ> group_used = {};
    std::array<int32_t, emel::batch::planner::action::MAX_SEQ> group_ids = {};
    int32_t group_count = 0;
    int32_t last_primary = -1;
    bool stop_group_scan = false;

    for (int32_t i = 0; i < ev.request.n_tokens && !stop_group_scan && !failed; ++i) {
      const int32_t seq_id = ev.request.seq_primary_ids[i];
      const bool valid_seq = seq_id >= 0 && seq_id < max_seq;
      const size_t slot = static_cast<size_t>(select_i32(valid_seq, seq_id, 0));

      const bool slot_exhausted = !valid_seq || seq_used[slot] >= seq_counts[slot];
      const bool already_grouped = group_used[slot] != 0;
      const bool out_of_order =
          ev.request.equal_sequential && group_count > 0 && seq_id != last_primary + 1;
      const bool skip_slot = slot_exhausted || already_grouped || out_of_order;
      const bool use_slot = !skip_slot;

      group_used[slot] = select_u8(use_slot, 1u, group_used[slot]);
      const int32_t group_index = select_i32(use_slot, group_count, 0);
      group_ids[static_cast<size_t>(group_index)] =
          select_i32(use_slot, seq_id, group_ids[static_cast<size_t>(group_index)]);
      group_count += static_cast<int32_t>(use_slot);
      last_primary = select_i32(use_slot, seq_id, last_primary);
      stop_group_scan = stop_group_scan ||
          (use_slot && group_count > ev.ctx.effective_step_size);
    }

    fail_if(group_count == 0,
            failed,
            ev,
            emel::batch::planner::error::planning_progress_stalled);

    int32_t min_avail = ev.request.n_tokens + 1;
    for (int32_t g = 0; g < group_count; ++g) {
      const int32_t seq_id = group_ids[static_cast<size_t>(g)];
      const size_t slot = static_cast<size_t>(select_i32(seq_id >= 0, seq_id, 0));
      const int32_t avail = seq_counts[slot] - seq_used[slot];
      min_avail = std::min(min_avail, avail);
    }

    const int32_t safe_group_count = select_i32(group_count > 0, group_count, 1);
    const int32_t max_rows = ev.ctx.effective_step_size / safe_group_count;
    const int32_t n_seq_tokens = std::min(max_rows, min_avail);

    fail_if(n_seq_tokens <= 0,
            failed,
            ev,
            emel::batch::planner::error::planning_progress_stalled);

    fail_if(!failed && !begin_step(ev.ctx),
            failed,
            ev,
            emel::batch::planner::error::output_steps_full);

    for (int32_t g = 0; g < group_count && !failed; ++g) {
      const int32_t seq_id = group_ids[static_cast<size_t>(g)];
      const size_t slot = static_cast<size_t>(select_i32(seq_id >= 0, seq_id, 0));
      const int32_t base = seq_offsets[slot] + seq_used[slot];

      for (int32_t i = 0; i < n_seq_tokens && !failed; ++i) {
        const int32_t read_pos = base + i;
        const bool valid_read = read_pos >= 0 && read_pos < ev.request.n_tokens;
        fail_if(!valid_read,
                failed,
                ev,
                emel::batch::planner::error::algorithm_failed);

        const size_t read_index = static_cast<size_t>(select_i32(valid_read, read_pos, 0));
        const int32_t idx = seq_indices[read_index];
        fail_if(valid_read && !append_token_index(ev.ctx, idx),
                failed,
                ev,
                emel::batch::planner::error::output_indices_full);
      }

      const bool group_complete = !failed;
      const int32_t consumed = select_i32(group_complete, n_seq_tokens, 0);
      seq_used[slot] += consumed;
      remaining -= consumed;
    }

    const int32_t added = n_seq_tokens * group_count;
    fail_if(!failed && !push_step_size(ev.ctx, added),
            failed,
            ev,
            emel::batch::planner::error::output_steps_full);
  }

  finalize_offsets_if_success(ev.ctx, failed);
}

inline void prepare_plan(const event::request_runtime & ev) noexcept {
  clear_plan(ev.ctx);
  ev.ctx.total_outputs = count_total_outputs(ev.request);
}

}  // namespace emel::batch::planner::modes::detail
