#pragma once

#include <algorithm>
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
  const auto add_if = [ &add ](const bool condition, const error code) {
        {
      const size_t emel_branch_1 = static_cast<size_t>(condition);
      for (size_t emel_case_1 = emel_branch_1; emel_case_1 == 1u; emel_case_1 = 2u) {
                add(code);
      }
      for (size_t emel_case_1 = emel_branch_1; emel_case_1 == 0u; emel_case_1 = 2u) {

      }
    }
  };

  add_if(ev.token_ids == nullptr, error::invalid_token_data);
  add_if(ev.n_tokens <= 0, error::invalid_request);
  add_if(ev.n_tokens > action::MAX_PLAN_STEPS, error::output_plan_full);
  add_if(ev.seq_mask_words <= 0 || ev.seq_mask_words > action::SEQ_WORDS,
         error::invalid_sequence_metadata);
  add_if(ev.output_mask != nullptr && ev.output_mask_count < ev.n_tokens,
         error::invalid_sequence_metadata);
  add_if(ev.seq_masks != nullptr && ev.seq_masks_count < ev.n_tokens,
         error::invalid_sequence_metadata);
  add_if(ev.seq_primary_ids != nullptr && ev.seq_primary_ids_count < ev.n_tokens,
         error::invalid_sequence_id);

  const bool require_primary_ids =
      ev.mode == event::plan_mode::equal && ev.equal_sequential && ev.seq_masks != nullptr;
  add_if(require_primary_ids && ev.seq_primary_ids == nullptr,
         error::invalid_sequence_metadata);

    {
    const size_t emel_branch_2 = static_cast<size_t>(ev.n_tokens <= 0);
    for (size_t emel_case_2 = emel_branch_2; emel_case_2 == 1u; emel_case_2 = 2u) {
            return mask;
    }
    for (size_t emel_case_2 = emel_branch_2; emel_case_2 == 0u; emel_case_2 = 2u) {

    }
  }

  const bool has_masks = ev.seq_masks != nullptr && ev.seq_mask_words > 0 &&
                         ev.seq_mask_words <= action::SEQ_WORDS;
  const bool has_primary_ids = ev.seq_primary_ids != nullptr;
  const int32_t max_seq = ev.seq_mask_words * 64;
  for (int32_t idx = 0; idx < ev.n_tokens; ++idx) {
        {
      const size_t emel_branch_3 = static_cast<size_t>(has_primary_ids);
      for (size_t emel_case_3 = emel_branch_3; emel_case_3 == 1u; emel_case_3 = 2u) {
         {
                const int32_t primary_id = ev.seq_primary_ids[static_cast<size_t>(idx)];
                add_if(primary_id < 0 || primary_id >= max_seq, error::invalid_sequence_id);
                break;
              }
      }
      for (size_t emel_case_3 = emel_branch_3; emel_case_3 == 0u; emel_case_3 = 2u) {

      }
    }
        {
      const size_t emel_branch_4 = static_cast<size_t>(has_masks);
      for (size_t emel_case_4 = emel_branch_4; emel_case_4 == 1u; emel_case_4 = 2u) {
         {
                const seq_mask_t mask_value = normalized_seq_mask(ev, idx);
                add_if(!mask_any_set(mask_value), error::invalid_sequence_mask);
                add_if(ev.mode == event::plan_mode::equal && ev.equal_sequential &&
                           mask_has_multiple_bits(mask_value),
                       error::multiple_bits_in_mask);
                break;
              }
      }
      for (size_t emel_case_4 = emel_branch_4; emel_case_4 == 0u; emel_case_4 = 2u) {

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
  const bool has_masks = ev.seq_masks != nullptr;
  const bool has_primary = ev.seq_primary_ids != nullptr;
  const size_t mode =
      static_cast<size_t>(has_masks) * 2 + static_cast<size_t>(has_primary);
  const size_t use_masks = static_cast<size_t>(mode >= 2u);
  const size_t use_primary = static_cast<size_t>(mode == 1u);
  {
    const size_t emel_branch_masks = use_masks;
    for (size_t emel_case_masks = emel_branch_masks; emel_case_masks == 1u;
         emel_case_masks = 2u) {
      const int32_t words = ev.seq_mask_words;
      for (int32_t w = 0; w < words; ++w) {
        mask[static_cast<size_t>(w)] =
            ev.seq_masks[static_cast<size_t>(idx) * static_cast<size_t>(words) +
                         static_cast<size_t>(w)];
      }
      return mask;
    }
    for (size_t emel_case_masks = emel_branch_masks; emel_case_masks == 0u;
         emel_case_masks = 2u) {

    }
  }
  {
    const size_t emel_branch_primary = use_primary;
    for (size_t emel_case_primary = emel_branch_primary; emel_case_primary == 1u;
         emel_case_primary = 2u) {
      const uint32_t bit = static_cast<uint32_t>(ev.seq_primary_ids[idx]);
      {
        const size_t emel_branch_valid_bit =
            static_cast<size_t>(bit < static_cast<uint32_t>(ev.seq_mask_words * 64));
        for (size_t emel_case_valid_bit = emel_branch_valid_bit; emel_case_valid_bit == 1u;
             emel_case_valid_bit = 2u) {
          const uint32_t word = bit / 64U;
          const uint32_t shift = bit % 64U;
          mask[static_cast<size_t>(word)] = (uint64_t{1} << shift);
        }
        for (size_t emel_case_valid_bit = emel_branch_valid_bit; emel_case_valid_bit == 0u;
             emel_case_valid_bit = 2u) {

        }
      }
      return mask;
    }
    for (size_t emel_case_primary = emel_branch_primary; emel_case_primary == 0u;
         emel_case_primary = 2u) {

    }
  }
  mask[0] = uint64_t{1};
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
    {
    const size_t emel_branch_5 = static_cast<size_t>(ev.output_all);
    for (size_t emel_case_5 = emel_branch_5; emel_case_5 == 1u; emel_case_5 = 2u) {
            return ev.n_tokens;
    }
    for (size_t emel_case_5 = emel_branch_5; emel_case_5 == 0u; emel_case_5 = 2u) {

    }
  }
    {
    const size_t emel_branch_6 = static_cast<size_t>(ev.output_mask == nullptr);
    for (size_t emel_case_6 = emel_branch_6; emel_case_6 == 1u; emel_case_6 = 2u) {
            const std::array<int32_t, 2> single_output_counts = {0, 1};
            return single_output_counts[static_cast<size_t>(ev.n_tokens > 0)];
    }
    for (size_t emel_case_6 = emel_branch_6; emel_case_6 == 0u; emel_case_6 = 2u) {

    }
  }
  int32_t total = 0;
  for (int32_t i = 0; i < ev.n_tokens; ++i) {
    total += (ev.output_mask[i] != 0);
  }
  return total;
}

inline bool append_token_index(request_ctx & ctx, const int32_t idx) noexcept {
    {
    const size_t emel_branch_7 = static_cast<size_t>(ctx.token_indices_count >= action::MAX_PLAN_STEPS);
    for (size_t emel_case_7 = emel_branch_7; emel_case_7 == 1u; emel_case_7 = 2u) {
            return false;
    }
    for (size_t emel_case_7 = emel_branch_7; emel_case_7 == 0u; emel_case_7 = 2u) {

    }
  }
  ctx.step_token_indices[ctx.token_indices_count] = idx;
  ctx.token_indices_count += 1;
  return true;
}

inline bool begin_step(request_ctx & ctx) noexcept {
    {
    const size_t emel_branch_8 = static_cast<size_t>(ctx.step_count >= action::MAX_PLAN_STEPS);
    for (size_t emel_case_8 = emel_branch_8; emel_case_8 == 1u; emel_case_8 = 2u) {
            return false;
    }
    for (size_t emel_case_8 = emel_branch_8; emel_case_8 == 0u; emel_case_8 = 2u) {

    }
  }
  ctx.step_token_offsets[ctx.step_count] = ctx.token_indices_count;
  return true;
}

inline void finalize_token_offsets(request_ctx & ctx) noexcept {
    {
    const size_t emel_branch_9 = static_cast<size_t>(ctx.step_count <= action::MAX_PLAN_STEPS);
    for (size_t emel_case_9 = emel_branch_9; emel_case_9 == 1u; emel_case_9 = 2u) {
            ctx.step_token_offsets[ctx.step_count] = ctx.token_indices_count;
    }
    for (size_t emel_case_9 = emel_branch_9; emel_case_9 == 0u; emel_case_9 = 2u) {

    }
  }
}

inline bool push_step_size(request_ctx & ctx, const int32_t size) noexcept {
  const bool invalid_size = size <= 0;
  const bool full_steps = ctx.step_count >= action::MAX_PLAN_STEPS;
    {
    const size_t emel_branch_10 = static_cast<size_t>(invalid_size || full_steps);
    for (size_t emel_case_10 = emel_branch_10; emel_case_10 == 1u; emel_case_10 = 2u) {
            return false;
    }
    for (size_t emel_case_10 = emel_branch_10; emel_case_10 == 0u; emel_case_10 = 2u) {

    }
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

inline void create_simple_plan(const event::request_runtime & ev) noexcept {
    {
    const size_t emel_branch_11 = static_cast<size_t>(ev.ctx.effective_step_size <= 0);
    for (size_t emel_case_11 = emel_branch_11; emel_case_11 == 1u; emel_case_11 = 2u) {
            fail_plan(ev, emel::batch::planner::error::invalid_step_size);
            return;
    }
    for (size_t emel_case_11 = emel_branch_11; emel_case_11 == 0u; emel_case_11 = 2u) {

    }
  }

  int32_t next_token = 0;
  while (next_token < ev.request.n_tokens) {
        {
      const size_t emel_branch_12 = static_cast<size_t>(!begin_step(ev.ctx));
      for (size_t emel_case_12 = emel_branch_12; emel_case_12 == 1u; emel_case_12 = 2u) {
                fail_plan(ev, emel::batch::planner::error::output_steps_full);
                return;
      }
      for (size_t emel_case_12 = emel_branch_12; emel_case_12 == 0u; emel_case_12 = 2u) {

      }
    }
    const int32_t chunk =
        std::min<int32_t>(ev.ctx.effective_step_size, ev.request.n_tokens - next_token);
    for (int32_t i = 0; i < chunk; ++i) {
            {
        const size_t emel_branch_13 = static_cast<size_t>(!append_token_index(ev.ctx, next_token + i));
        for (size_t emel_case_13 = emel_branch_13; emel_case_13 == 1u; emel_case_13 = 2u) {
                    fail_plan(ev, emel::batch::planner::error::output_indices_full);
                    return;
        }
        for (size_t emel_case_13 = emel_branch_13; emel_case_13 == 0u; emel_case_13 = 2u) {

        }
      }
    }
    next_token += chunk;
        {
      const size_t emel_branch_14 = static_cast<size_t>(!push_step_size(ev.ctx, chunk));
      for (size_t emel_case_14 = emel_branch_14; emel_case_14 == 1u; emel_case_14 = 2u) {
                fail_plan(ev, emel::batch::planner::error::output_steps_full);
                return;
      }
      for (size_t emel_case_14 = emel_branch_14; emel_case_14 == 0u; emel_case_14 = 2u) {

      }
    }
  }
  finalize_token_offsets(ev.ctx);
}

inline void create_sequential_plan(const event::request_runtime & ev) noexcept {
    {
    const size_t emel_branch_15 = static_cast<size_t>(ev.ctx.effective_step_size <= 0);
    for (size_t emel_case_15 = emel_branch_15; emel_case_15 == 1u; emel_case_15 = 2u) {
            fail_plan(ev, emel::batch::planner::error::invalid_step_size);
            return;
    }
    for (size_t emel_case_15 = emel_branch_15; emel_case_15 == 0u; emel_case_15 = 2u) {

    }
  }

  std::array<uint8_t, emel::batch::planner::action::MAX_PLAN_STEPS> used = {};
  int32_t used_count = 0;
  bool done = false;

  while (used_count < ev.request.n_tokens && !done) {
    int32_t cur_idx = 0;
    while (cur_idx < ev.request.n_tokens && used[static_cast<size_t>(cur_idx)] != 0) {
      ++cur_idx;
    }
    const bool exhausted = cur_idx >= ev.request.n_tokens;
        {
      const size_t emel_branch_16 = static_cast<size_t>(exhausted);
      for (size_t emel_case_16 = emel_branch_16; emel_case_16 == 1u; emel_case_16 = 2u) {
                done = true;
      }
      for (size_t emel_case_16 = emel_branch_16; emel_case_16 == 0u; emel_case_16 = 2u) {

      }
    }
    {
      const size_t emel_branch_process = static_cast<size_t>(!done);
      for (size_t emel_case_process = emel_branch_process; emel_case_process == 1u;
           emel_case_process = 2u) {
        int32_t chunk = 0;
        seq_mask_t cur_mask = normalized_seq_mask(ev.request, cur_idx);
            {
          const size_t emel_branch_17 = static_cast<size_t>(!begin_step(ev.ctx));
          for (size_t emel_case_17 = emel_branch_17; emel_case_17 == 1u; emel_case_17 = 2u) {
                    fail_plan(ev, emel::batch::planner::error::output_steps_full);
                    return;
          }
          for (size_t emel_case_17 = emel_branch_17; emel_case_17 == 0u; emel_case_17 = 2u) {

          }
        }

        bool continue_chunk = true;
        while (continue_chunk) {
          used[static_cast<size_t>(cur_idx)] = 1;
          used_count += 1;
          chunk += 1;
                {
            const size_t emel_branch_18 =
                static_cast<size_t>(!append_token_index(ev.ctx, cur_idx));
            for (size_t emel_case_18 = emel_branch_18; emel_case_18 == 1u;
                 emel_case_18 = 2u) {
                        fail_plan(ev, emel::batch::planner::error::output_indices_full);
                        return;
            }
            for (size_t emel_case_18 = emel_branch_18; emel_case_18 == 0u;
                 emel_case_18 = 2u) {

            }
          }

          const bool reached_step_size = chunk >= ev.ctx.effective_step_size;
          continue_chunk = continue_chunk && !reached_step_size;
          {
            const size_t emel_branch_find_next = static_cast<size_t>(!reached_step_size);
            for (size_t emel_case_find_next = emel_branch_find_next;
                 emel_case_find_next == 1u;
                 emel_case_find_next = 2u) {
              int32_t next_idx = cur_idx + 1;
              while (next_idx < ev.request.n_tokens &&
                     (used[static_cast<size_t>(next_idx)] != 0 ||
                      !mask_is_subset(cur_mask, normalized_seq_mask(ev.request, next_idx)))) {
                ++next_idx;
              }

              const bool no_candidate = next_idx >= ev.request.n_tokens;
                    {
                const size_t emel_branch_19 = static_cast<size_t>(no_candidate);
                for (size_t emel_case_19 = emel_branch_19; emel_case_19 == 1u;
                     emel_case_19 = 2u) {
                            continue_chunk = false;
                }
                for (size_t emel_case_19 = emel_branch_19; emel_case_19 == 0u;
                     emel_case_19 = 2u) {
                            cur_idx = next_idx;
                            cur_mask = normalized_seq_mask(ev.request, cur_idx);
                }
              }
            }
            for (size_t emel_case_find_next = emel_branch_find_next;
                 emel_case_find_next == 0u;
                 emel_case_find_next = 2u) {

            }
          }
        }

            {
          const size_t emel_branch_20 = static_cast<size_t>(!push_step_size(ev.ctx, chunk));
          for (size_t emel_case_20 = emel_branch_20; emel_case_20 == 1u;
               emel_case_20 = 2u) {
                    fail_plan(ev, emel::batch::planner::error::output_steps_full);
                    return;
          }
          for (size_t emel_case_20 = emel_branch_20; emel_case_20 == 0u;
               emel_case_20 = 2u) {

          }
        }
      }
      for (size_t emel_case_process = emel_branch_process; emel_case_process == 0u;
           emel_case_process = 2u) {

      }
    }
  }
  finalize_token_offsets(ev.ctx);
}

inline void create_equal_plan(const event::request_runtime & ev) noexcept {
    {
    const size_t emel_branch_21 = static_cast<size_t>(ev.ctx.effective_step_size <= 0);
    for (size_t emel_case_21 = emel_branch_21; emel_case_21 == 1u; emel_case_21 = 2u) {
            fail_plan(ev, emel::batch::planner::error::invalid_step_size);
            return;
    }
    for (size_t emel_case_21 = emel_branch_21; emel_case_21 == 0u; emel_case_21 = 2u) {

    }
  }

  std::array<uint8_t, emel::batch::planner::action::MAX_PLAN_STEPS> used = {};
  int32_t used_count = 0;

  while (used_count < ev.request.n_tokens) {
    struct group_state {
      seq_mask_t mask = {};
    };
    std::array<group_state, emel::batch::planner::action::MAX_PLAN_STEPS> groups = {};
    int32_t group_count = 0;
    int32_t last_primary = -1;
    bool stop_group_scan = false;

    for (int32_t i = 0; i < ev.request.n_tokens && !stop_group_scan; ++i) {
      const bool is_unused = used[static_cast<size_t>(i)] == 0;
      const seq_mask_t mask = normalized_seq_mask(ev.request, i);
      bool overlap = false;
      for (int32_t g = 0; g < group_count; ++g) {
        overlap = overlap || mask_overlaps(groups[g].mask, mask);
      }
      const bool requires_sequential_primary =
          ev.request.equal_sequential && ev.request.seq_primary_ids != nullptr;
      int32_t primary = last_primary;
      {
        const size_t emel_branch_has_primary = static_cast<size_t>(requires_sequential_primary);
        for (size_t emel_case_has_primary = emel_branch_has_primary;
             emel_case_has_primary == 1u;
             emel_case_has_primary = 2u) {
          primary = ev.request.seq_primary_ids[i];
        }
        for (size_t emel_case_has_primary = emel_branch_has_primary;
             emel_case_has_primary == 0u;
             emel_case_has_primary = 2u) {

        }
      }
      const bool out_of_order =
          requires_sequential_primary && group_count > 0 && primary != last_primary + 1;
      const bool can_add_group = is_unused && !overlap && !out_of_order;
      {
        const size_t emel_branch_can_add = static_cast<size_t>(can_add_group);
        for (size_t emel_case_can_add = emel_branch_can_add; emel_case_can_add == 1u;
             emel_case_can_add = 2u) {
          {
            const size_t emel_branch_update_primary =
                static_cast<size_t>(requires_sequential_primary);
            for (size_t emel_case_update_primary = emel_branch_update_primary;
                 emel_case_update_primary == 1u;
                 emel_case_update_primary = 2u) {
              last_primary = primary;
            }
            for (size_t emel_case_update_primary = emel_branch_update_primary;
                 emel_case_update_primary == 0u;
                 emel_case_update_primary = 2u) {

            }
          }
          groups[group_count] = group_state{.mask = mask};
          group_count += 1;
          stop_group_scan = group_count > ev.ctx.effective_step_size;
        }
        for (size_t emel_case_can_add = emel_branch_can_add; emel_case_can_add == 0u;
             emel_case_can_add = 2u) {

        }
      }
    }

        {
      const size_t emel_branch_22 = static_cast<size_t>(group_count == 0);
      for (size_t emel_case_22 = emel_branch_22; emel_case_22 == 1u; emel_case_22 = 2u) {
                fail_plan(ev, emel::batch::planner::error::planning_progress_stalled);
                return;
      }
      for (size_t emel_case_22 = emel_branch_22; emel_case_22 == 0u; emel_case_22 = 2u) {

      }
    }

    int32_t min_avail = ev.request.n_tokens + 1;
    for (int32_t g = 0; g < group_count; ++g) {
      int32_t avail = 0;
      for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
        const bool available =
            used[static_cast<size_t>(i)] == 0 &&
            mask_equal(normalized_seq_mask(ev.request, i), groups[g].mask);
        avail += static_cast<int32_t>(available);
      }
      min_avail = std::min(min_avail, avail);
    }

    const int32_t max_rows = ev.ctx.effective_step_size / group_count;
    const int32_t n_seq_tokens = std::min(max_rows, min_avail);
        {
      const size_t emel_branch_23 = static_cast<size_t>(n_seq_tokens <= 0);
      for (size_t emel_case_23 = emel_branch_23; emel_case_23 == 1u; emel_case_23 = 2u) {
                fail_plan(ev, emel::batch::planner::error::planning_progress_stalled);
                return;
      }
      for (size_t emel_case_23 = emel_branch_23; emel_case_23 == 0u; emel_case_23 = 2u) {

      }
    }

        {
      const size_t emel_branch_24 = static_cast<size_t>(!begin_step(ev.ctx));
      for (size_t emel_case_24 = emel_branch_24; emel_case_24 == 1u; emel_case_24 = 2u) {
                fail_plan(ev, emel::batch::planner::error::output_steps_full);
                return;
      }
      for (size_t emel_case_24 = emel_branch_24; emel_case_24 == 0u; emel_case_24 = 2u) {

      }
    }

    for (int32_t g = 0; g < group_count; ++g) {
      int32_t remaining = n_seq_tokens;
      for (int32_t i = 0; i < ev.request.n_tokens && remaining > 0; ++i) {
        const bool match = used[static_cast<size_t>(i)] == 0 &&
            mask_equal(normalized_seq_mask(ev.request, i), groups[g].mask);
                {
          const size_t emel_branch_25 = static_cast<size_t>(match);
          for (size_t emel_case_25 = emel_branch_25; emel_case_25 == 1u; emel_case_25 = 2u) {
                        used[static_cast<size_t>(i)] = 1;
                        used_count += 1;
                        {
                          const size_t emel_branch_append =
                              static_cast<size_t>(!append_token_index(ev.ctx, i));
                          for (size_t emel_case_append = emel_branch_append;
                               emel_case_append == 1u;
                               emel_case_append = 2u) {
                            fail_plan(ev, emel::batch::planner::error::output_indices_full);
                            return;
                          }
                          for (size_t emel_case_append = emel_branch_append;
                               emel_case_append == 0u;
                               emel_case_append = 2u) {

                          }
                        }
                        remaining -= 1;
          }
          for (size_t emel_case_25 = emel_branch_25; emel_case_25 == 0u; emel_case_25 = 2u) {

          }
        }
      }
            {
        const size_t emel_branch_26 = static_cast<size_t>(remaining != 0);
        for (size_t emel_case_26 = emel_branch_26; emel_case_26 == 1u; emel_case_26 = 2u) {
                    fail_plan(ev, emel::batch::planner::error::algorithm_failed);
                    return;
        }
        for (size_t emel_case_26 = emel_branch_26; emel_case_26 == 0u; emel_case_26 = 2u) {

        }
      }
    }

    const int32_t added = n_seq_tokens * group_count;
        {
      const size_t emel_branch_27 = static_cast<size_t>(!push_step_size(ev.ctx, added));
      for (size_t emel_case_27 = emel_branch_27; emel_case_27 == 1u; emel_case_27 = 2u) {
                fail_plan(ev, emel::batch::planner::error::output_steps_full);
                return;
      }
      for (size_t emel_case_27 = emel_branch_27; emel_case_27 == 0u; emel_case_27 = 2u) {

      }
    }
  }
  finalize_token_offsets(ev.ctx);
}

inline void create_equal_plan_primary_fast_path(const event::request_runtime & ev) noexcept {
    {
    const size_t emel_branch_28 = static_cast<size_t>(ev.ctx.effective_step_size <= 0);
    for (size_t emel_case_28 = emel_branch_28; emel_case_28 == 1u; emel_case_28 = 2u) {
            fail_plan(ev, emel::batch::planner::error::invalid_step_size);
            return;
    }
    for (size_t emel_case_28 = emel_branch_28; emel_case_28 == 0u; emel_case_28 = 2u) {

    }
  }
    {
    const size_t emel_branch_29 = static_cast<size_t>(ev.request.seq_primary_ids == nullptr);
    for (size_t emel_case_29 = emel_branch_29; emel_case_29 == 1u; emel_case_29 = 2u) {
            fail_plan(ev, emel::batch::planner::error::invalid_sequence_id);
            return;
    }
    for (size_t emel_case_29 = emel_branch_29; emel_case_29 == 0u; emel_case_29 = 2u) {

    }
  }

  const int32_t max_seq = ev.request.seq_mask_words * 64;
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ> seq_counts = {};
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ + 1> seq_offsets = {};
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ> seq_used = {};
  std::array<int32_t, emel::batch::planner::action::MAX_SEQ> seq_cursor = {};
  std::array<int32_t, emel::batch::planner::action::MAX_PLAN_STEPS> seq_indices = {};

  for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
    const int32_t seq_id = ev.request.seq_primary_ids[i];
        {
      const size_t emel_branch_30 = static_cast<size_t>(seq_id < 0 || seq_id >= max_seq);
      for (size_t emel_case_30 = emel_branch_30; emel_case_30 == 1u; emel_case_30 = 2u) {
                fail_plan(ev, emel::batch::planner::error::invalid_sequence_id);
                return;
      }
      for (size_t emel_case_30 = emel_branch_30; emel_case_30 == 0u; emel_case_30 = 2u) {

      }
    }
    seq_counts[static_cast<size_t>(seq_id)] += 1;
  }

  for (int32_t s = 0; s < max_seq; ++s) {
    seq_offsets[static_cast<size_t>(s + 1)] =
        seq_offsets[static_cast<size_t>(s)] + seq_counts[static_cast<size_t>(s)];
    seq_cursor[static_cast<size_t>(s)] = seq_offsets[static_cast<size_t>(s)];
  }

  for (int32_t i = 0; i < ev.request.n_tokens; ++i) {
    const int32_t seq_id = ev.request.seq_primary_ids[i];
    const size_t slot = static_cast<size_t>(seq_id);
    const int32_t pos = seq_cursor[slot];
        {
      const size_t emel_branch_31 = static_cast<size_t>(pos < 0 || pos >= ev.request.n_tokens);
      for (size_t emel_case_31 = emel_branch_31; emel_case_31 == 1u; emel_case_31 = 2u) {
                fail_plan(ev, emel::batch::planner::error::algorithm_failed);
                return;
      }
      for (size_t emel_case_31 = emel_branch_31; emel_case_31 == 0u; emel_case_31 = 2u) {

      }
    }
    seq_indices[static_cast<size_t>(pos)] = i;
    seq_cursor[slot] = pos + 1;
  }

  int32_t remaining = ev.request.n_tokens;
  while (remaining > 0) {
    std::array<uint8_t, emel::batch::planner::action::MAX_SEQ> group_used = {};
    std::array<int32_t, emel::batch::planner::action::MAX_SEQ> group_ids = {};
    int32_t group_count = 0;
    int32_t last_primary = -1;
    bool stop_group_scan = false;

    for (int32_t i = 0; i < ev.request.n_tokens && !stop_group_scan; ++i) {
      const int32_t seq_id = ev.request.seq_primary_ids[i];
      const size_t slot = static_cast<size_t>(seq_id);
      const bool slot_exhausted = seq_used[slot] >= seq_counts[slot];
      const bool already_grouped = group_used[slot] != 0;
      const bool out_of_order =
          ev.request.equal_sequential && group_count > 0 && seq_id != last_primary + 1;
      const bool skip_slot = slot_exhausted || already_grouped || out_of_order;
      {
        const size_t emel_branch_use_slot = static_cast<size_t>(!skip_slot);
        for (size_t emel_case_use_slot = emel_branch_use_slot; emel_case_use_slot == 1u;
             emel_case_use_slot = 2u) {
          group_used[slot] = 1;
          group_ids[static_cast<size_t>(group_count)] = seq_id;
          group_count += 1;
          last_primary = seq_id;
          stop_group_scan = group_count > ev.ctx.effective_step_size;
        }
        for (size_t emel_case_use_slot = emel_branch_use_slot; emel_case_use_slot == 0u;
             emel_case_use_slot = 2u) {

        }
      }
    }

        {
      const size_t emel_branch_32 = static_cast<size_t>(group_count == 0);
      for (size_t emel_case_32 = emel_branch_32; emel_case_32 == 1u; emel_case_32 = 2u) {
                fail_plan(ev, emel::batch::planner::error::planning_progress_stalled);
                return;
      }
      for (size_t emel_case_32 = emel_branch_32; emel_case_32 == 0u; emel_case_32 = 2u) {

      }
    }

    int32_t min_avail = ev.request.n_tokens + 1;
    for (int32_t g = 0; g < group_count; ++g) {
      const int32_t seq_id = group_ids[static_cast<size_t>(g)];
      const size_t slot = static_cast<size_t>(seq_id);
      const int32_t avail = seq_counts[slot] - seq_used[slot];
      min_avail = std::min(min_avail, avail);
    }

    const int32_t max_rows = ev.ctx.effective_step_size / group_count;
    const int32_t n_seq_tokens = std::min(max_rows, min_avail);
        {
      const size_t emel_branch_33 = static_cast<size_t>(n_seq_tokens <= 0);
      for (size_t emel_case_33 = emel_branch_33; emel_case_33 == 1u; emel_case_33 = 2u) {
                fail_plan(ev, emel::batch::planner::error::planning_progress_stalled);
                return;
      }
      for (size_t emel_case_33 = emel_branch_33; emel_case_33 == 0u; emel_case_33 = 2u) {

      }
    }

        {
      const size_t emel_branch_34 = static_cast<size_t>(!begin_step(ev.ctx));
      for (size_t emel_case_34 = emel_branch_34; emel_case_34 == 1u; emel_case_34 = 2u) {
                fail_plan(ev, emel::batch::planner::error::output_steps_full);
                return;
      }
      for (size_t emel_case_34 = emel_branch_34; emel_case_34 == 0u; emel_case_34 = 2u) {

      }
    }

    for (int32_t g = 0; g < group_count; ++g) {
      const int32_t seq_id = group_ids[static_cast<size_t>(g)];
      const size_t slot = static_cast<size_t>(seq_id);
      const int32_t base = seq_offsets[slot] + seq_used[slot];
      for (int32_t i = 0; i < n_seq_tokens; ++i) {
        const int32_t idx = seq_indices[static_cast<size_t>(base + i)];
                {
          const size_t emel_branch_35 = static_cast<size_t>(!append_token_index(ev.ctx, idx));
          for (size_t emel_case_35 = emel_branch_35; emel_case_35 == 1u; emel_case_35 = 2u) {
                        fail_plan(ev, emel::batch::planner::error::output_indices_full);
                        return;
          }
          for (size_t emel_case_35 = emel_branch_35; emel_case_35 == 0u; emel_case_35 = 2u) {

          }
        }
      }
      seq_used[slot] += n_seq_tokens;
      remaining -= n_seq_tokens;
    }

    const int32_t added = n_seq_tokens * group_count;
        {
      const size_t emel_branch_36 = static_cast<size_t>(!push_step_size(ev.ctx, added));
      for (size_t emel_case_36 = emel_branch_36; emel_case_36 == 1u; emel_case_36 = 2u) {
                fail_plan(ev, emel::batch::planner::error::output_steps_full);
                return;
      }
      for (size_t emel_case_36 = emel_branch_36; emel_case_36 == 0u; emel_case_36 = 2u) {

      }
    }
  }

  finalize_token_offsets(ev.ctx);
}

inline void prepare_plan(const event::request_runtime & ev) noexcept {
  clear_plan(ev.ctx);
  ev.ctx.total_outputs = count_total_outputs(ev.request);
}

}  // namespace emel::batch::planner::modes::detail
