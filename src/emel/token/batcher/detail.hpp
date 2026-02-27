#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <limits>

#include "emel/error/error.hpp"
#include "emel/token/batcher/context.hpp"
#include "emel/token/batcher/errors.hpp"
#include "emel/token/batcher/events.hpp"

namespace emel::token::batcher::detail {

template <class runtime_event_type>
constexpr decltype(auto) unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires { ev.event_; }) {
    return ev.event_;
  } else {
    return (ev);
  }
}

enum class probe_status : uint8_t {
  ok = 0u,
  backend_error = 1u,
  invalid = 2u,
};

inline bool has_seq_masks_input(const event::batch & req) noexcept {
  return req.seq_masks != nullptr && req.seq_masks_count >= req.n_tokens;
}

inline bool has_seq_primary_input(const event::batch & req) noexcept {
  return req.seq_primary_ids != nullptr && req.seq_primary_ids_count >= req.n_tokens;
}

inline bool has_output_mask_input(const event::batch & req) noexcept {
  return req.output_mask != nullptr && req.output_mask_count >= req.n_tokens;
}

inline int32_t effective_mask_words(const event::batch & req) noexcept {
  return has_seq_masks_input(req) ? req.seq_mask_words : 1;
}

inline int32_t positions_stride(const event::batch & req) noexcept {
  if (req.positions == nullptr) {
    return 0;
  }
  if (req.positions_count >= req.n_tokens * 3) {
    return 3;
  }
  if (req.positions_count >= req.n_tokens) {
    return 1;
  }
  return -1;
}

inline int32_t normalized_positions_count(const event::batch & req) noexcept {
  return positions_stride(req) == 3 ? req.n_tokens * 3 : req.n_tokens;
}

inline const int32_t * token_ids_ptr(const event::batch & req) noexcept {
  return &req.token_ids;
}

inline int32_t * seq_primary_ids_out_ptr(const event::batch & req) noexcept {
  return &req.seq_primary_ids_out;
}

inline uint64_t * seq_masks_out_ptr(const event::batch & req) noexcept {
  return &req.seq_masks_out;
}

inline int32_t * positions_out_ptr(const event::batch & req) noexcept {
  return &req.positions_out;
}

inline int8_t * output_mask_out_ptr(const event::batch & req) noexcept {
  return &req.output_mask_out;
}

inline void write_error(const event::batch_runtime & ev, const emel::error::type value) noexcept {
  ev.request.error_out = value;
}

inline bool mask_empty(const uint64_t * mask, const int32_t words) noexcept {
  for (int32_t w = 0; w < words; ++w) {
    if (mask[static_cast<size_t>(w)] != 0U) {
      return false;
    }
  }
  return true;
}

inline void clear_mask(uint64_t * mask, const int32_t words) noexcept {
  for (int32_t w = 0; w < words; ++w) {
    mask[static_cast<size_t>(w)] = 0U;
  }
}

inline void set_mask_bit(uint64_t * mask, const int32_t words, const int32_t seq_id) noexcept {
  const int32_t word = seq_id / 64;
  if (word < 0 || word >= words) {
    return;
  }
  const int32_t bit = seq_id % 64;
  mask[static_cast<size_t>(word)] |= (uint64_t{1} << bit);
}

inline bool mask_has_bit(const uint64_t * mask,
                         const int32_t words,
                         const int32_t seq_id) noexcept {
  if (seq_id < 0) {
    return false;
  }
  const int32_t word = seq_id / 64;
  if (word < 0 || word >= words) {
    return false;
  }
  const int32_t bit = seq_id % 64;
  return (mask[static_cast<size_t>(word)] & (uint64_t{1} << bit)) != 0U;
}

inline int32_t mask_primary_id(const uint64_t * mask, const int32_t words) noexcept {
  for (int32_t w = 0; w < words; ++w) {
    const uint64_t bits = mask[static_cast<size_t>(w)];
    if (bits == 0U) {
      continue;
    }
    const int32_t bit = static_cast<int32_t>(std::countr_zero(bits));
    return w * 64 + bit;
  }
  return -1;
}

template <class fn_type>
inline bool for_each_mask_seq_id(const uint64_t * mask,
                                 const int32_t words,
                                 const fn_type & fn) noexcept {
  for (int32_t w = 0; w < words; ++w) {
    uint64_t bits = mask[static_cast<size_t>(w)];
    while (bits != 0U) {
      const int32_t bit = static_cast<int32_t>(std::countr_zero(bits));
      const int32_t seq_id = w * 64 + bit;
      if (!fn(seq_id)) {
        return false;
      }
      bits &= (bits - 1U);
    }
  }
  return true;
}

inline bool primary_ids_in_range(const int32_t * primary_ids,
                                 const int32_t count,
                                 const int32_t seq_limit) noexcept {
  for (int32_t i = 0; i < count; ++i) {
    const int32_t seq_id = primary_ids[i];
    if (seq_id < 0 || seq_id >= seq_limit) {
      return false;
    }
  }
  return true;
}

inline bool masks_have_non_empty_rows(const event::batch & req) noexcept {
  if (!has_seq_masks_input(req)) {
    return true;
  }
  const int32_t mask_words = req.seq_mask_words;
  for (int32_t i = 0; i < req.n_tokens; ++i) {
    const uint64_t * in_mask = req.seq_masks + static_cast<size_t>(i) * mask_words;
    if (mask_empty(in_mask, mask_words)) {
      return false;
    }
  }
  return true;
}

inline bool primary_in_mask_when_both_inputs(const event::batch & req) noexcept {
  if (!has_seq_masks_input(req) || !has_seq_primary_input(req)) {
    return true;
  }
  const int32_t mask_words = req.seq_mask_words;
  for (int32_t i = 0; i < req.n_tokens; ++i) {
    const int32_t primary = req.seq_primary_ids[i];
    const uint64_t * in_mask = req.seq_masks + static_cast<size_t>(i) * mask_words;
    if (!mask_has_bit(in_mask, mask_words, primary)) {
      return false;
    }
  }
  return true;
}

inline bool single_output_per_seq_ok(const event::batch_runtime & ev) noexcept {
  const auto & req = ev.request;
  const int32_t mask_words = ev.ctx.normalized_seq_mask_words;
  const uint64_t * seq_masks_out = seq_masks_out_ptr(req);
  const int8_t * output_mask_out = output_mask_out_ptr(req);
  std::array<int32_t, action::MAX_SEQ> seq_output_count = {};

  for (int32_t i = 0; i < req.n_tokens; ++i) {
    if (output_mask_out[i] == 0) {
      continue;
    }

    const uint64_t * mask = seq_masks_out + static_cast<size_t>(i) * mask_words;
    if (!for_each_mask_seq_id(mask, mask_words, [&](const int32_t seq_id) noexcept {
          seq_output_count[seq_id] += 1;
          return seq_output_count[seq_id] <= 1;
        })) {
      return false;
    }
  }

  return true;
}

inline bool continuity_ok(const event::batch_runtime & ev) noexcept {
  const auto & req = ev.request;
  const int32_t mask_words = ev.ctx.normalized_seq_mask_words;
  const uint64_t * seq_masks_out = seq_masks_out_ptr(req);
  const int32_t * positions_out = positions_out_ptr(req);

  std::array<int32_t, action::MAX_SEQ> seq_last_pos = {};
  std::array<int32_t, action::MAX_SEQ> seq_pos_min = {};
  std::array<int32_t, action::MAX_SEQ> seq_pos_max = {};
  std::array<int32_t, action::MAX_SEQ> seq_pos_count = {};
  std::array<uint8_t, action::MAX_SEQ> seq_seen = {};
  std::array<int32_t, action::MAX_SEQ> active_seq_ids = {};
  std::array<uint64_t, static_cast<size_t>(action::MAX_SEQ * action::SEQ_WORDS)> cur_seq_set = {};
  int32_t active_seq_count = 0;

  seq_last_pos.fill(-1);
  seq_pos_min.fill(std::numeric_limits<int32_t>::max());
  seq_pos_max.fill(std::numeric_limits<int32_t>::min());

  for (int32_t i = 0; i < req.n_tokens; ++i) {
    const int32_t pos = positions_out[i];
    const uint64_t * mask = seq_masks_out + static_cast<size_t>(i) * mask_words;

    if (!for_each_mask_seq_id(mask, mask_words, [&](const int32_t seq_id) noexcept {
          const int32_t last = seq_last_pos[seq_id];
          if (last >= 0 && pos < last) {
            return false;
          }

          if (pos != last) {
            seq_pos_count[seq_id] += 1;
          }
          seq_last_pos[seq_id] = pos;
          if (pos < seq_pos_min[seq_id]) {
            seq_pos_min[seq_id] = pos;
          }
          if (pos > seq_pos_max[seq_id]) {
            seq_pos_max[seq_id] = pos;
          }

          uint64_t * cur_mask = cur_seq_set.data() + static_cast<size_t>(seq_id * mask_words);
          if (seq_seen[seq_id] == 0U) {
            seq_seen[seq_id] = 1U;
            active_seq_ids[active_seq_count] = seq_id;
            active_seq_count += 1;
            for (int32_t mw = 0; mw < mask_words; ++mw) {
              cur_mask[static_cast<size_t>(mw)] = ~uint64_t{0};
            }
          }

          for (int32_t mw = 0; mw < mask_words; ++mw) {
            cur_mask[static_cast<size_t>(mw)] &= mask[static_cast<size_t>(mw)];
          }
          return !mask_empty(cur_mask, mask_words);
        })) {
      return false;
    }
  }

  for (int32_t i = 0; i < active_seq_count; ++i) {
    const int32_t seq_id = active_seq_ids[i];
    const int32_t min_pos = seq_pos_min[seq_id];
    const int32_t max_pos = seq_pos_max[seq_id];
    const int32_t count = seq_pos_count[seq_id];
    if (min_pos == std::numeric_limits<int32_t>::max() ||
        max_pos == std::numeric_limits<int32_t>::min()) {
      continue;
    }
    if (max_pos - min_pos + 1 > count) {
      return false;
    }
  }

  return true;
}

inline probe_status seeded_generation_probe(
    const event::batch_runtime & ev,
    std::array<int32_t, action::MAX_SEQ> & seeded_next_pos_out) noexcept {
  const auto & req = ev.request;
  const int32_t mask_words = ev.ctx.normalized_seq_mask_words;
  const int32_t * seq_primary_ids_out = seq_primary_ids_out_ptr(req);
  const uint64_t * seq_masks_out = seq_masks_out_ptr(req);
  std::array<int32_t, action::MAX_SEQ> next_pos = {};

  for (int32_t seq_id = 0; seq_id < action::MAX_SEQ; ++seq_id) {
    int32_t seed = 0;
    if (!req.resolve_position_seed(req.position_seed_ctx, seq_id, &seed)) {
      return probe_status::backend_error;
    }
    if (seed < 0) {
      return probe_status::invalid;
    }
    next_pos[seq_id] = seed;
  }
  seeded_next_pos_out = next_pos;

  for (int32_t i = 0; i < req.n_tokens; ++i) {
    const int32_t primary = seq_primary_ids_out[i];
    const int32_t pos = next_pos[primary];
    if (pos == std::numeric_limits<int32_t>::max()) {
      return probe_status::invalid;
    }

    const uint64_t * mask = seq_masks_out + static_cast<size_t>(i) * mask_words;
    if (!for_each_mask_seq_id(mask, mask_words, [&](const int32_t seq_id) noexcept {
          return next_pos[seq_id] == pos;
        })) {
      return probe_status::invalid;
    }

    for_each_mask_seq_id(mask, mask_words, [&](const int32_t seq_id) noexcept {
      next_pos[seq_id] = pos + 1;
      return true;
    });
  }

  return probe_status::ok;
}

inline bool unseeded_generation_probe(const event::batch_runtime & ev) noexcept {
  const auto & req = ev.request;
  const int32_t mask_words = ev.ctx.normalized_seq_mask_words;
  const int32_t * seq_primary_ids_out = seq_primary_ids_out_ptr(req);
  const uint64_t * seq_masks_out = seq_masks_out_ptr(req);
  std::array<int32_t, action::MAX_SEQ> next_pos = {};
  std::array<uint8_t, action::MAX_SEQ> seeded = {};

  for (int32_t i = 0; i < req.n_tokens; ++i) {
    const int32_t primary = seq_primary_ids_out[i];
    const int32_t pos = next_pos[primary];
    if (pos == std::numeric_limits<int32_t>::max()) {
      return false;
    }

    const uint64_t * mask = seq_masks_out + static_cast<size_t>(i) * mask_words;
    if (!for_each_mask_seq_id(mask, mask_words, [&](const int32_t seq_id) noexcept {
          const bool first_seen = seeded[seq_id] == 0U;
          const int32_t current = first_seen ? pos : next_pos[seq_id];
          return current == pos;
        })) {
      return false;
    }

    for_each_mask_seq_id(mask, mask_words, [&](const int32_t seq_id) noexcept {
      seeded[seq_id] = 1U;
      next_pos[seq_id] = pos + 1;
      return true;
    });
  }

  return true;
}

}  // namespace emel::token::batcher::detail
