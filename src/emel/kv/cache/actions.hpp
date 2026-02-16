#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "emel/emel.h"
#include "emel/kv/cache/events.hpp"

namespace emel::kv::cache::action {

inline constexpr int32_t MAX_UBATCHES = 4096;
inline constexpr int32_t MAX_KV_CELLS = 32768;

enum class operation : int32_t {
  none = 0,
  prepare,
  apply,
  rollback,
};

struct context {
  operation op = operation::none;

  std::array<int32_t, MAX_UBATCHES> ubatch_sizes = {};
  std::array<int32_t, MAX_UBATCHES> slot_offsets = {};
  int32_t ubatch_count = 0;
  int32_t planned_ubatch_count = 0;
  int32_t requested_capacity = 0;
  int32_t kv_size = 0;
  int32_t head = 0;

  std::array<int32_t, MAX_KV_CELLS> cells = {};
  int32_t next_pos = 1;

  int32_t current_ubatch_index = 0;
  int32_t applied_ubatches = 0;
  int32_t kv_tokens = 0;
};

inline int32_t count_used_cells(const context & ctx) {
  int32_t used = 0;
  for (int32_t i = 0; i < ctx.kv_size; ++i) {
    if (ctx.cells[i] != 0) {
      ++used;
    }
  }
  return used;
}

inline int32_t used_max_p1(const context & ctx) {
  for (int32_t i = ctx.kv_size - 1; i >= 0; --i) {
    if (ctx.cells[i] != 0) {
      return i + 1;
    }
  }
  return 0;
}

inline int32_t find_contiguous_slot(
    const std::array<int32_t, MAX_KV_CELLS> & cells,
    int32_t kv_size,
    int32_t head_start,
    int32_t n_tokens,
    int32_t used_cells,
    int32_t & head_after) {
  if (kv_size <= 0 || n_tokens <= 0 || n_tokens > kv_size) {
    return -1;
  }

  int32_t head_cur = head_start;
  if (head_cur > used_cells + 2 * n_tokens) {
    head_cur = 0;
  }

  int32_t n_tested = 0;
  while (n_tested < kv_size) {
    if (head_cur + n_tokens > kv_size) {
      n_tested += kv_size - head_cur;
      head_cur = 0;
      continue;
    }

    bool can_use = true;
    for (int32_t i = 0; i < n_tokens; ++i) {
      if (cells[head_cur + i] != 0) {
        can_use = false;
        break;
      }
    }

    if (can_use) {
      head_after = head_cur + n_tokens;
      if (head_after >= kv_size) {
        head_after %= kv_size;
      }
      return head_cur;
    }

    ++head_cur;
    ++n_tested;
  }

  return -1;
}

inline constexpr auto begin_prepare = [](const event::prepare & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  if (ev.ubatch_count_out != nullptr) {
    *ev.ubatch_count_out = 0;
  }
  ctx.op = operation::prepare;
  ctx.ubatch_count = ev.ubatch_count;
  ctx.planned_ubatch_count = 0;
  ctx.requested_capacity = ev.requested_capacity;
  ctx.current_ubatch_index = 0;

  ctx.ubatch_sizes.fill(0);
  ctx.slot_offsets.fill(0);
  if (ev.ubatch_sizes != nullptr && ev.ubatch_count > 0 && ev.ubatch_count <= MAX_UBATCHES) {
    for (int32_t i = 0; i < ev.ubatch_count; ++i) {
      ctx.ubatch_sizes[i] = ev.ubatch_sizes[i];
    }
  }

  if (ev.requested_capacity > 0 && ev.requested_capacity <= MAX_KV_CELLS) {
    ctx.kv_size = std::max(ctx.kv_size, ev.requested_capacity);
    if (ctx.head >= ctx.kv_size) {
      ctx.head = 0;
    }
  }
};

inline constexpr auto begin_apply = [](const event::apply_ubatch & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  if (ev.kv_tokens_out != nullptr) {
    *ev.kv_tokens_out = 0;
  }
  ctx.op = operation::apply;
  ctx.current_ubatch_index = ev.ubatch_index;
};

inline constexpr auto begin_rollback = [](const event::rollback & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  ctx.op = operation::rollback;
  ctx.current_ubatch_index = ev.from_ubatch_index;
};

inline constexpr auto run_validate = [](const event::validate & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  switch (ctx.op) {
    case operation::prepare:
      if (ctx.ubatch_count <= 0 || ctx.ubatch_count > MAX_UBATCHES) {
        *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
        return;
      }
      if (ctx.requested_capacity > MAX_KV_CELLS) {
        *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
        return;
      }
      if (ctx.kv_size <= 0 || ctx.kv_size > MAX_KV_CELLS) {
        *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
        return;
      }
      break;
    case operation::apply:
      if (ctx.planned_ubatch_count <= 0) {
        *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
        return;
      }
      if (ctx.current_ubatch_index < 0 || ctx.current_ubatch_index >= ctx.planned_ubatch_count) {
        *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
        return;
      }
      if (ctx.current_ubatch_index != ctx.applied_ubatches) {
        *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
        return;
      }
      break;
    case operation::rollback:
      if (ctx.current_ubatch_index < 0 ||
          ctx.current_ubatch_index > ctx.applied_ubatches ||
          ctx.current_ubatch_index > ctx.planned_ubatch_count) {
        *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
        return;
      }
      break;
    case operation::none:
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      return;
  }
};

inline constexpr auto run_prepare_slots = [](const event::prepare_slots & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  auto cells_sim = ctx.cells;
  int32_t head_sim = ctx.head;
  int32_t used_sim = count_used_cells(ctx);

  for (int32_t i = 0; i < ctx.ubatch_count; ++i) {
    const int32_t size = ctx.ubatch_sizes[i];
    if (size <= 0) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }

    int32_t head_after = head_sim;
    const int32_t slot = find_contiguous_slot(
        cells_sim,
        ctx.kv_size,
        head_sim,
        size,
        used_sim,
        head_after);
    if (slot < 0) {
      *ev.error_out = EMEL_ERR_BACKEND;
      return;
    }

    ctx.slot_offsets[i] = slot;
    for (int32_t j = 0; j < size; ++j) {
      cells_sim[slot + j] = 1;
    }

    head_sim = head_after;
    used_sim = std::min(ctx.kv_size, used_sim + size);
  }

  ctx.planned_ubatch_count = ctx.ubatch_count;
  ctx.applied_ubatches = 0;
  ctx.kv_tokens = used_max_p1(ctx);
};

inline constexpr auto run_apply_step = [](const event::apply_step & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.op != operation::apply) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  if (ctx.current_ubatch_index >= ctx.planned_ubatch_count) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  const int32_t size = ctx.ubatch_sizes[ctx.current_ubatch_index];
  const int32_t start = ctx.slot_offsets[ctx.current_ubatch_index];
  if (size <= 0 || start < 0 || start + size > ctx.kv_size) {
    *ev.error_out = EMEL_ERR_BACKEND;
    return;
  }

  for (int32_t i = 0; i < size; ++i) {
    if (ctx.cells[start + i] != 0) {
      *ev.error_out = EMEL_ERR_BACKEND;
      return;
    }
    ctx.cells[start + i] = ctx.next_pos++;
  }

  ctx.head = start + size;
  if (ctx.head >= ctx.kv_size) {
    ctx.head %= ctx.kv_size;
  }

  ctx.applied_ubatches = ctx.current_ubatch_index + 1;
  ctx.kv_tokens = used_max_p1(ctx);
};

inline constexpr auto run_rollback_step = [](const event::rollback_step & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.op != operation::rollback) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  int32_t new_head = ctx.kv_size;
  for (int32_t i = ctx.current_ubatch_index; i < ctx.applied_ubatches; ++i) {
    const int32_t size = ctx.ubatch_sizes[i];
    const int32_t start = ctx.slot_offsets[i];
    if (size <= 0 || start < 0 || start + size > ctx.kv_size) {
      *ev.error_out = EMEL_ERR_BACKEND;
      return;
    }

    for (int32_t j = 0; j < size; ++j) {
      ctx.cells[start + j] = 0;
    }
    new_head = std::min(new_head, start);
  }

  if (new_head < ctx.kv_size && new_head < ctx.head) {
    ctx.head = new_head;
  }

  ctx.applied_ubatches = ctx.current_ubatch_index;
  ctx.kv_tokens = used_max_p1(ctx);
};

inline constexpr auto run_publish = [](const event::publish & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
};

inline constexpr auto on_kv_done = [](const events::kv_done &, context & ctx) {
  ctx.op = operation::none;
  ctx.current_ubatch_index = 0;
};

inline constexpr auto on_kv_error = [](const events::kv_error &, context & ctx) {
  ctx.op = operation::none;
  ctx.current_ubatch_index = 0;
};

}  // namespace emel::kv::cache::action
