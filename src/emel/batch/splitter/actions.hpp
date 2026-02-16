#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "emel/batch/splitter/events.hpp"
#include "emel/emel.h"

namespace emel::batch::splitter::action {

inline constexpr int32_t MAX_UBATCHES = 4096;

struct context {
  const int32_t * token_ids = nullptr;
  int32_t n_tokens = 0;
  int32_t requested_n_ubatch = 0;
  event::split_mode mode = event::split_mode::simple;
  const uint64_t * seq_masks = nullptr;
  const int32_t * seq_primary_ids = nullptr;
  bool equal_sequential = true;

  int32_t effective_n_ubatch = 0;
  std::array<int32_t, MAX_UBATCHES> ubatch_sizes = {};
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;
};

inline constexpr auto begin_split = [](const event::split & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  if (ev.ubatch_count_out != nullptr) {
    *ev.ubatch_count_out = 0;
  }
  if (ev.total_outputs_out != nullptr) {
    *ev.total_outputs_out = 0;
  }
  ctx.token_ids = ev.token_ids;
  ctx.n_tokens = ev.n_tokens;
  ctx.requested_n_ubatch = ev.n_ubatch;
  ctx.mode = ev.mode;
  ctx.seq_masks = ev.seq_masks;
  ctx.seq_primary_ids = ev.seq_primary_ids;
  ctx.equal_sequential = ev.equal_sequential;
  ctx.effective_n_ubatch = 0;
  ctx.ubatch_count = 0;
  ctx.total_outputs = 0;
  ctx.ubatch_sizes.fill(0);
};

inline constexpr auto run_validate = [](const event::validate & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.token_ids == nullptr || ctx.n_tokens <= 0) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  if (ctx.n_tokens > MAX_UBATCHES) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  switch (ctx.mode) {
    case event::split_mode::simple:
    case event::split_mode::equal:
    case event::split_mode::seq:
      break;
    default:
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      return;
  }
};

inline constexpr auto run_normalize_batch = [](const event::normalize_batch & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  const int32_t default_ubatch = ctx.n_tokens;
  const int32_t requested = ctx.requested_n_ubatch > 0 ? ctx.requested_n_ubatch : default_ubatch;
  ctx.effective_n_ubatch = std::max<int32_t>(1, std::min<int32_t>(requested, ctx.n_tokens));
};

inline uint64_t normalized_seq_mask(const context & ctx, const int32_t idx) {
  if (ctx.seq_masks != nullptr && ctx.seq_masks[idx] != 0) {
    return ctx.seq_masks[idx];
  }
  if (ctx.seq_primary_ids != nullptr && ctx.seq_primary_ids[idx] >= 0) {
    return (uint64_t{1} << (static_cast<uint32_t>(ctx.seq_primary_ids[idx]) & 63U));
  }
  return (uint64_t{1} << (static_cast<uint32_t>(idx) & 63U));
}

inline bool push_ubatch_size(context & ctx, const int32_t size, int32_t * err) {
  if (size <= 0) {
    *err = EMEL_ERR_BACKEND;
    return false;
  }
  if (ctx.ubatch_count >= MAX_UBATCHES) {
    *err = EMEL_ERR_BACKEND;
    return false;
  }
  ctx.ubatch_sizes[ctx.ubatch_count] = size;
  ctx.ubatch_count += 1;
  return true;
}

inline constexpr auto run_create_ubatches = [](const event::create_ubatches & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  ctx.ubatch_count = 0;
  ctx.total_outputs = ctx.n_tokens;

  if (ctx.effective_n_ubatch <= 0) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  switch (ctx.mode) {
    case event::split_mode::simple:
    {
      int32_t remaining = ctx.n_tokens;
      while (remaining > 0) {
        const int32_t chunk = std::min<int32_t>(remaining, ctx.effective_n_ubatch);
        if (!push_ubatch_size(ctx, chunk, ev.error_out)) {
          return;
        }
        remaining -= chunk;
      }
      break;
    }
    case event::split_mode::equal: {
      if (ctx.seq_masks != nullptr) {
        std::array<uint8_t, MAX_UBATCHES> used = {};
        int32_t used_count = 0;

        while (used_count < ctx.n_tokens) {
          struct group_state {
            uint64_t mask = 0;
            int32_t next_idx = -1;
          };
          std::array<group_state, MAX_UBATCHES> groups = {};
          int32_t group_count = 0;
          int32_t last_primary = -1;

          for (int32_t i = 0; i < ctx.n_tokens; ++i) {
            if (used[static_cast<size_t>(i)] != 0) {
              continue;
            }

            const uint64_t mask = normalized_seq_mask(ctx, i);
            bool overlap = false;
            for (int32_t g = 0; g < group_count; ++g) {
              if ((groups[g].mask & mask) != 0) {
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
            if (group_count >= ctx.effective_n_ubatch) {
              break;
            }
          }

          if (group_count == 0) {
            *ev.error_out = EMEL_ERR_BACKEND;
            return;
          }

          for (int32_t g = 0; g < group_count; ++g) {
            int32_t idx = 0;
            while (idx < ctx.n_tokens) {
              if (used[static_cast<size_t>(idx)] == 0 &&
                  normalized_seq_mask(ctx, idx) == groups[g].mask) {
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
                    normalized_seq_mask(ctx, next) == groups[g].mask) {
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

          if (!push_ubatch_size(ctx, added, ev.error_out)) {
            return;
          }
        }
        break;
      }

      const int32_t n_chunks =
          (ctx.n_tokens + ctx.effective_n_ubatch - 1) / ctx.effective_n_ubatch;
      if (n_chunks <= 0 || n_chunks > MAX_UBATCHES) {
        *ev.error_out = EMEL_ERR_BACKEND;
        return;
      }

      const int32_t base = ctx.n_tokens / n_chunks;
      const int32_t extra = ctx.n_tokens % n_chunks;
      for (int32_t i = 0; i < n_chunks; ++i) {
        ctx.ubatch_sizes[i] = base + (i < extra ? 1 : 0);
      }
      ctx.ubatch_count = n_chunks;
      break;
    }
    case event::split_mode::seq: {
      if (ctx.seq_masks == nullptr) {
        int32_t remaining = ctx.n_tokens;
        while (remaining > 0) {
          const int32_t chunk = std::min<int32_t>(remaining, ctx.effective_n_ubatch);
          if (!push_ubatch_size(ctx, chunk, ev.error_out)) {
            return;
          }
          remaining -= chunk;
        }
        break;
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
        uint64_t cur_mask = normalized_seq_mask(ctx, cur_idx);
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
              const uint64_t next_mask = normalized_seq_mask(ctx, next_idx);
              if ((cur_mask & next_mask) == next_mask) {
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

        if (!push_ubatch_size(ctx, chunk, ev.error_out)) {
          return;
        }
      }
      break;
    }
    default:
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      return;
  }
};

inline constexpr auto run_publish = [](const event::publish & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
};

inline constexpr auto on_splitting_done = [](const events::splitting_done & ev, context & ctx) {
  const event::split * request = ev.request;
  if (request == nullptr) {
    return;
  }
  if (request->ubatch_sizes_out != nullptr) {
    for (int32_t i = 0; i < ctx.ubatch_count; ++i) {
      request->ubatch_sizes_out[i] = ctx.ubatch_sizes[i];
    }
  }
  if (request->ubatch_count_out != nullptr) {
    *request->ubatch_count_out = ctx.ubatch_count;
  }
  if (request->total_outputs_out != nullptr) {
    *request->total_outputs_out = ctx.total_outputs;
  }
  if (request->error_out != nullptr) {
    *request->error_out = EMEL_OK;
  }
};
inline constexpr auto on_splitting_error = [](const events::splitting_error & ev, context &) {
  const event::split * request = ev.request;
  if (request != nullptr && request->error_out != nullptr) {
    *request->error_out = ev.err;
  }
};

}  // namespace emel::batch::splitter::action
