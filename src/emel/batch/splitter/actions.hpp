#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "emel/batch/splitter/events.hpp"
#include "emel/emel.h"

namespace emel::batch::splitter::action {

inline constexpr int32_t MAX_UBATCHES = 4096;
inline constexpr int32_t MAX_SEQ = 256;
inline constexpr int32_t SEQ_WORDS = (MAX_SEQ + 63) / 64;

struct context {
  const int32_t * token_ids = nullptr;
  int32_t n_tokens = 0;
  int32_t requested_n_ubatch = 0;
  event::split_mode mode = event::split_mode::simple;
  const uint64_t * seq_masks = nullptr;
  const int32_t * seq_primary_ids = nullptr;
  bool equal_sequential = true;
  int32_t seq_mask_words = 1;
  const int8_t * output_mask = nullptr;

  int32_t effective_n_ubatch = 0;
  std::array<int32_t, MAX_UBATCHES> ubatch_sizes = {};
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;
};

// Initializes context for a new split request.
inline constexpr auto begin_split = [](const event::split & ev, context & ctx) noexcept {
  ctx.token_ids = ev.token_ids;
  ctx.n_tokens = ev.n_tokens;
  ctx.requested_n_ubatch = ev.n_ubatch;
  ctx.mode = ev.mode;
  ctx.seq_masks = ev.seq_masks;
  ctx.seq_primary_ids = ev.seq_primary_ids;
  ctx.equal_sequential = ev.equal_sequential;
  ctx.seq_mask_words = ev.seq_mask_words;
  ctx.output_mask = ev.output_mask;

  ctx.effective_n_ubatch = 0;
  ctx.ubatch_count = 0;
  ctx.total_outputs = 0;
  ctx.ubatch_sizes.fill(0);
};

// Normalizes the requested micro-batch size.
inline constexpr auto normalize_batch = [](context & ctx) noexcept {
  const int32_t default_ubatch = ctx.n_tokens;
  const int32_t requested = ctx.requested_n_ubatch > 0 ? ctx.requested_n_ubatch : default_ubatch;
  ctx.effective_n_ubatch = std::max<int32_t>(1, std::min<int32_t>(requested, ctx.n_tokens));
};

using seq_mask_t = std::array<uint64_t, SEQ_WORDS>;

inline seq_mask_t normalized_seq_mask(const context & ctx, const int32_t idx) noexcept {
  seq_mask_t mask = {};
  if (ctx.seq_masks != nullptr) {
    const int32_t words = ctx.seq_mask_words;
    for (int32_t w = 0; w < words; ++w) {
      mask[static_cast<size_t>(w)] =
          ctx.seq_masks[static_cast<size_t>(idx) * static_cast<size_t>(words) +
                        static_cast<size_t>(w)];
    }
    return mask;
  }
  if (ctx.seq_primary_ids != nullptr) {
    const uint32_t bit = static_cast<uint32_t>(ctx.seq_primary_ids[idx]);
    if (bit < static_cast<uint32_t>(ctx.seq_mask_words * 64)) {
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
  for (size_t w = 0; w < SEQ_WORDS; ++w) {
    if ((lhs[w] & rhs[w]) != 0) {
      return true;
    }
  }
  return false;
}

inline bool mask_equal(const seq_mask_t & lhs, const seq_mask_t & rhs) noexcept {
  for (size_t w = 0; w < SEQ_WORDS; ++w) {
    if (lhs[w] != rhs[w]) {
      return false;
    }
  }
  return true;
}

inline bool mask_is_subset(const seq_mask_t & superset, const seq_mask_t & subset) noexcept {
  for (size_t w = 0; w < SEQ_WORDS; ++w) {
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

inline int32_t count_total_outputs(const context & ctx) noexcept {
  if (ctx.output_mask == nullptr) {
    return ctx.n_tokens;
  }
  int32_t total = 0;
  for (int32_t i = 0; i < ctx.n_tokens; ++i) {
    total += (ctx.output_mask[i] != 0);
  }
  return total;
}

inline bool push_ubatch_size(context & ctx, const int32_t size) noexcept {
  if (size <= 0) {
    return false;
  }
  if (ctx.ubatch_count >= MAX_UBATCHES) {
    return false;
  }
  ctx.ubatch_sizes[ctx.ubatch_count] = size;
  ctx.ubatch_count += 1;
  return true;
}

inline void fail_split(context & ctx) noexcept {
  ctx.ubatch_sizes.fill(0);
  ctx.ubatch_count = 0;
  ctx.total_outputs = 0;
}

inline void prepare_split(context & ctx) noexcept {
  ctx.ubatch_sizes.fill(0);
  ctx.ubatch_count = 0;
  ctx.total_outputs = count_total_outputs(ctx);
}

// Materializes micro-batch boundaries in simple mode. On failure, leaves ubatch_count == 0.
inline auto create_ubatches_simple = [](context & ctx) noexcept {
  prepare_split(ctx);

  int32_t remaining = ctx.n_tokens;
  while (remaining > 0) {
    const int32_t chunk = std::min<int32_t>(remaining, ctx.effective_n_ubatch);
    if (!push_ubatch_size(ctx, chunk)) {
      fail_split(ctx);
      return;
    }
    remaining -= chunk;
  }
};

// Materializes micro-batch boundaries in equal mode. On failure, leaves ubatch_count == 0.
inline auto create_ubatches_equal = [](context & ctx) noexcept {
  prepare_split(ctx);

  if (ctx.effective_n_ubatch <= 0) {
    fail_split(ctx);
    return;
  }

  std::array<uint8_t, MAX_UBATCHES> used = {};
  int32_t used_count = 0;

  while (used_count < ctx.n_tokens) {
    struct group_state {
      seq_mask_t mask = {};
      int32_t next_idx = -1;
    };
    std::array<group_state, MAX_UBATCHES> groups = {};
    int32_t group_count = 0;
    int32_t last_primary = -1;

    for (int32_t i = 0; i < ctx.n_tokens; ++i) {
      if (used[static_cast<size_t>(i)] != 0) {
        continue;
      }

      const seq_mask_t mask = normalized_seq_mask(ctx, i);
      bool overlap = false;
      for (int32_t g = 0; g < group_count; ++g) {
        if (mask_overlaps(groups[g].mask, mask)) {
          overlap = true;
          break;
        }
      }
      if (overlap) {
        continue;
      }

      if (ctx.equal_sequential && ctx.seq_primary_ids != nullptr) {
        const int32_t primary = ctx.seq_primary_ids[i];
        if (group_count > 0 && primary != last_primary + 1) {
          continue;
        }
        last_primary = primary;
      }

      groups[group_count] = group_state{
        .mask = mask,
        .next_idx = -1,
      };
      group_count += 1;
      if (group_count > ctx.effective_n_ubatch) {
        break;
      }
    }

    if (group_count == 0) {
      fail_split(ctx);
      return;
    }

    for (int32_t g = 0; g < group_count; ++g) {
      int32_t idx = 0;
      while (idx < ctx.n_tokens) {
        if (used[static_cast<size_t>(idx)] == 0 &&
            mask_equal(normalized_seq_mask(ctx, idx), groups[g].mask)) {
          break;
        }
        ++idx;
      }
      groups[g].next_idx = idx;
    }

    int32_t n_seq_tokens = 0;
    int32_t added = 0;
    while (true) {
      bool can_expand = true;
      for (int32_t g = 0; g < group_count; ++g) {
        if (groups[g].next_idx >= ctx.n_tokens) {
          can_expand = false;
          break;
        }
      }
      if (!can_expand) {
        break;
      }

      for (int32_t g = 0; g < group_count; ++g) {
        const int32_t idx = groups[g].next_idx;
        used[static_cast<size_t>(idx)] = 1;
        used_count += 1;
        added += 1;

        int32_t next = idx + 1;
        while (next < ctx.n_tokens) {
          if (used[static_cast<size_t>(next)] == 0 &&
              mask_equal(normalized_seq_mask(ctx, next), groups[g].mask)) {
            break;
          }
          ++next;
        }
        groups[g].next_idx = next;
      }

      n_seq_tokens += 1;
      if ((n_seq_tokens + 1) * group_count > ctx.effective_n_ubatch) {
        break;
      }
    }

    if (!push_ubatch_size(ctx, added)) {
      fail_split(ctx);
      return;
    }
  }
};

// Materializes micro-batch boundaries in seq mode. On failure, leaves ubatch_count == 0.
inline auto create_ubatches_seq = [](context & ctx) noexcept {
  prepare_split(ctx);

  if (ctx.effective_n_ubatch <= 0) {
    fail_split(ctx);
    return;
  }

  std::array<uint8_t, MAX_UBATCHES> used = {};
  int32_t used_count = 0;

  while (used_count < ctx.n_tokens) {
    int32_t cur_idx = 0;
    while (cur_idx < ctx.n_tokens && used[static_cast<size_t>(cur_idx)] != 0) {
      ++cur_idx;
    }
    if (cur_idx >= ctx.n_tokens) {
      break;
    }

    int32_t chunk = 0;
    seq_mask_t cur_mask = normalized_seq_mask(ctx, cur_idx);
    while (true) {
      used[static_cast<size_t>(cur_idx)] = 1;
      used_count += 1;
      chunk += 1;

      if (chunk >= ctx.effective_n_ubatch) {
        break;
      }

      int32_t next_idx = cur_idx + 1;
      while (next_idx < ctx.n_tokens) {
        if (used[static_cast<size_t>(next_idx)] == 0) {
          const seq_mask_t next_mask = normalized_seq_mask(ctx, next_idx);
          if (mask_is_subset(cur_mask, next_mask)) {
            break;
          }
        }
        ++next_idx;
      }
      if (next_idx >= ctx.n_tokens) {
        break;
      }

      cur_idx = next_idx;
      cur_mask = normalized_seq_mask(ctx, cur_idx);
    }

    if (!push_ubatch_size(ctx, chunk)) {
      fail_split(ctx);
      return;
    }
  }
};

// Publishes split outputs (output write-back happens in caller via callbacks).
inline constexpr auto publish = [](context &) noexcept {};

inline constexpr auto dispatch_done = [](const event::split & ev, const context & ctx) noexcept {
  if (!ev.on_done) {
    return;
  }

  ev.on_done(events::splitting_done{
    .request = &ev,
    .ubatch_sizes = ctx.ubatch_sizes.data(),
    .ubatch_count = ctx.ubatch_count,
    .total_outputs = ctx.total_outputs,
  });
};

inline constexpr auto dispatch_invalid_request = [](const event::split & ev) noexcept {
  if (!ev.on_error) {
    return;
  }

  ev.on_error(events::splitting_error{
    .err = EMEL_ERR_INVALID_ARGUMENT,
    .request = &ev,
  });
};

inline constexpr auto dispatch_split_failed = [](const event::split & ev) noexcept {
  if (!ev.on_error) {
    return;
  }

  ev.on_error(events::splitting_error{
    .err = EMEL_ERR_BACKEND,
    .request = &ev,
  });
};

inline constexpr auto dispatch_unexpected = [](const auto & ev) noexcept {
  if constexpr (requires { ev.on_error; }) {
    if (!ev.on_error) {
      return;
    }

    ev.on_error(events::splitting_error{
      .err = EMEL_ERR_INVALID_ARGUMENT,
      .request = nullptr,
    });
  }
};

}  // namespace emel::batch::splitter::action
