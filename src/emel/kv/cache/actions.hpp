#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include "emel/emel.h"
#include "emel/kv/cache/events.hpp"

namespace emel::kv::cache::action {

inline constexpr int32_t MAX_UBATCHES = 4096;
inline constexpr int32_t MAX_KV_CELLS = 32768;
inline constexpr int32_t MAX_SEQ = 256;
inline constexpr int32_t MAX_STREAMS = MAX_SEQ;
inline constexpr int32_t SEQ_WORDS = (MAX_SEQ + 63) / 64;
inline constexpr int32_t MAX_STREAM_COPY = MAX_STREAMS;
inline constexpr int32_t POS_NONE = -1;

struct stream_state {
  int32_t head = 0;
  int32_t used_count = 0;
  int32_t used_max_p1 = 0;
  bool has_shift = false;

  std::array<int32_t, MAX_KV_CELLS> pos = {};
  std::array<int32_t, MAX_KV_CELLS> shift = {};
  std::array<int32_t, MAX_KV_CELLS> ext_x = {};
  std::array<int32_t, MAX_KV_CELLS> ext_y = {};
  std::array<uint16_t, MAX_KV_CELLS> seq_count = {};
  std::array<std::array<uint64_t, SEQ_WORDS>, MAX_KV_CELLS> seq_mask = {};
  std::array<int32_t, MAX_SEQ> seq_pos_min = {};
  std::array<int32_t, MAX_SEQ> seq_pos_max = {};
  std::array<uint16_t, MAX_SEQ> seq_pos_min_count = {};
  std::array<uint16_t, MAX_SEQ> seq_pos_max_count = {};
};

struct context {
  std::array<int32_t, MAX_UBATCHES> ubatch_sizes = {};
  std::array<int32_t, MAX_UBATCHES> slot_offsets = {};
  std::array<int32_t, MAX_UBATCHES> ubatch_stream_ids = {};
  std::array<int32_t, MAX_UBATCHES> ubatch_seq_ids = {};
  int32_t ubatch_count = 0;
  int32_t planned_ubatch_count = 0;
  int32_t applied_ubatches = 0;

  int32_t kv_size = 0;
  int32_t n_stream = 1;
  int32_t n_swa = 0;
  int32_t swa_type = 0;
  std::array<int32_t, MAX_SEQ> seq_to_stream = {};
  std::array<int32_t, MAX_SEQ> next_pos = {};
  std::array<stream_state, MAX_STREAMS> streams = {};

  std::array<int32_t, MAX_STREAM_COPY> pending_copy_src = {};
  std::array<int32_t, MAX_STREAM_COPY> pending_copy_dst = {};
  int32_t pending_copy_count = 0;

  int32_t kv_tokens = 0;

  context();
};

inline void reset_stream(stream_state & s) {
  s.head = 0;
  s.used_count = 0;
  s.used_max_p1 = 0;
  s.has_shift = false;
  s.pos.fill(POS_NONE);
  s.shift.fill(0);
  s.ext_x.fill(0);
  s.ext_y.fill(0);
  s.seq_count.fill(0);
  for (auto & mask : s.seq_mask) {
    mask.fill(0);
  }
  s.seq_pos_min.fill(POS_NONE);
  s.seq_pos_max.fill(POS_NONE);
  s.seq_pos_min_count.fill(0);
  s.seq_pos_max_count.fill(0);
}

inline void reset_context_state(context & ctx) {
  ctx.ubatch_sizes.fill(0);
  ctx.slot_offsets.fill(0);
  ctx.ubatch_stream_ids.fill(0);
  ctx.ubatch_seq_ids.fill(0);
  ctx.ubatch_count = 0;
  ctx.planned_ubatch_count = 0;
  ctx.applied_ubatches = 0;
  ctx.kv_size = 0;
  ctx.n_stream = 1;
  ctx.n_swa = 0;
  ctx.swa_type = 0;
  ctx.seq_to_stream.fill(0);
  ctx.next_pos.fill(0);
  for (auto & stream : ctx.streams) {
    reset_stream(stream);
  }
  ctx.pending_copy_src.fill(0);
  ctx.pending_copy_dst.fill(0);
  ctx.pending_copy_count = 0;
  ctx.kv_tokens = 0;
}

inline context::context() {
  reset_context_state(*this);
}

inline int32_t count_used_cells(const stream_state & s) {
  return s.used_count;
}

inline int32_t used_max_p1(const stream_state & s) {
  return s.used_max_p1;
}

inline bool ranges_overlap(int32_t start_a, int32_t size_a, int32_t start_b, int32_t size_b) {
  return start_a < start_b + size_b && start_b < start_a + size_a;
}

inline bool pos_in_range(int32_t pos, int32_t p0, int32_t p1) {
  if (p0 < 0 && p1 < 0) {
    return true;
  }
  if (p0 < 0) {
    return pos < p1;
  }
  if (p1 < 0) {
    return pos >= p0;
  }
  return pos >= p0 && pos < p1;
}

inline bool cell_has_seq(const stream_state & s, int32_t idx, int32_t seq_id) {
  if (seq_id < 0 || seq_id >= MAX_SEQ) {
    return false;
  }
  const int32_t word = seq_id / 64;
  const int32_t bit = seq_id % 64;
  return (s.seq_mask[idx][word] & (uint64_t(1) << bit)) != 0;
}

inline void set_seq_bit(stream_state & s, int32_t idx, int32_t seq_id) {
  const int32_t word = seq_id / 64;
  const int32_t bit = seq_id % 64;
  s.seq_mask[idx][word] |= (uint64_t(1) << bit);
}

inline void clear_seq_bit(stream_state & s, int32_t idx, int32_t seq_id) {
  const int32_t word = seq_id / 64;
  const int32_t bit = seq_id % 64;
  s.seq_mask[idx][word] &= ~(uint64_t(1) << bit);
}

inline void recompute_seq_pos_min(stream_state & s, int32_t seq_id) {
  int32_t min_pos = POS_NONE;
  uint16_t count = 0;
  for (int32_t i = 0; i < MAX_KV_CELLS; ++i) {
    if (s.pos[i] == POS_NONE) {
      continue;
    }
    if (!cell_has_seq(s, i, seq_id)) {
      continue;
    }
    if (min_pos == POS_NONE || s.pos[i] < min_pos) {
      min_pos = s.pos[i];
      count = 1;
    } else if (s.pos[i] == min_pos) {
      ++count;
    }
  }
  s.seq_pos_min[seq_id] = min_pos;
  s.seq_pos_min_count[seq_id] = count;
}

inline void recompute_seq_pos_max(stream_state & s, int32_t seq_id) {
  int32_t max_pos = POS_NONE;
  uint16_t count = 0;
  for (int32_t i = 0; i < MAX_KV_CELLS; ++i) {
    if (s.pos[i] == POS_NONE) {
      continue;
    }
    if (!cell_has_seq(s, i, seq_id)) {
      continue;
    }
    if (max_pos == POS_NONE || s.pos[i] > max_pos) {
      max_pos = s.pos[i];
      count = 1;
    } else if (s.pos[i] == max_pos) {
      ++count;
    }
  }
  s.seq_pos_max[seq_id] = max_pos;
  s.seq_pos_max_count[seq_id] = count;
}

inline void seq_pos_add(stream_state & s, int32_t seq_id, int32_t pos) {
  if (seq_id < 0 || seq_id >= MAX_SEQ || pos < 0) {
    return;
  }
  if (s.seq_pos_min[seq_id] == POS_NONE || pos < s.seq_pos_min[seq_id]) {
    s.seq_pos_min[seq_id] = pos;
    s.seq_pos_min_count[seq_id] = 1;
  } else if (pos == s.seq_pos_min[seq_id]) {
    ++s.seq_pos_min_count[seq_id];
  }
  if (s.seq_pos_max[seq_id] == POS_NONE || pos > s.seq_pos_max[seq_id]) {
    s.seq_pos_max[seq_id] = pos;
    s.seq_pos_max_count[seq_id] = 1;
  } else if (pos == s.seq_pos_max[seq_id]) {
    ++s.seq_pos_max_count[seq_id];
  }
}

inline void seq_pos_remove(stream_state & s, int32_t seq_id, int32_t pos) {
  if (seq_id < 0 || seq_id >= MAX_SEQ || pos < 0) {
    return;
  }
  if (pos == s.seq_pos_min[seq_id]) {
    if (s.seq_pos_min_count[seq_id] > 0) {
      --s.seq_pos_min_count[seq_id];
    }
    if (s.seq_pos_min_count[seq_id] == 0) {
      recompute_seq_pos_min(s, seq_id);
    }
  }
  if (pos == s.seq_pos_max[seq_id]) {
    if (s.seq_pos_max_count[seq_id] > 0) {
      --s.seq_pos_max_count[seq_id];
    }
    if (s.seq_pos_max_count[seq_id] == 0) {
      recompute_seq_pos_max(s, seq_id);
    }
  }
}

inline void recompute_used_max_p1(stream_state & s) {
  int32_t max_p1 = 0;
  for (int32_t i = MAX_KV_CELLS - 1; i >= 0; --i) {
    if (s.pos[i] != POS_NONE) {
      max_p1 = i + 1;
      break;
    }
  }
  s.used_max_p1 = max_p1;
}

inline void set_cell_empty(stream_state & s, int32_t idx) {
  if (s.pos[idx] == POS_NONE) {
    return;
  }
  for (int32_t word = 0; word < SEQ_WORDS; ++word) {
    uint64_t mask = s.seq_mask[idx][word];
    while (mask != 0) {
      const int32_t bit = __builtin_ctzll(mask);
      const int32_t seq_id = word * 64 + bit;
      seq_pos_remove(s, seq_id, s.pos[idx]);
      mask &= (mask - 1);
    }
  }
  s.pos[idx] = POS_NONE;
  s.shift[idx] = 0;
  s.ext_x[idx] = 0;
  s.ext_y[idx] = 0;
  s.seq_count[idx] = 0;
  for (int32_t word = 0; word < SEQ_WORDS; ++word) {
    s.seq_mask[idx][word] = 0;
  }
  if (s.used_count > 0) {
    --s.used_count;
  }
  if (idx + 1 == s.used_max_p1) {
    recompute_used_max_p1(s);
  }
  if (s.head > idx) {
    s.head = idx;
  }
}

inline void set_cell_pos(stream_state & s, int32_t idx, int32_t pos) {
  if (s.pos[idx] == POS_NONE) {
    ++s.used_count;
    if (idx + 1 > s.used_max_p1) {
      s.used_max_p1 = idx + 1;
    }
  } else {
    for (int32_t word = 0; word < SEQ_WORDS; ++word) {
      uint64_t mask = s.seq_mask[idx][word];
      while (mask != 0) {
        const int32_t bit = __builtin_ctzll(mask);
        const int32_t seq_id = word * 64 + bit;
        seq_pos_remove(s, seq_id, s.pos[idx]);
        mask &= (mask - 1);
      }
    }
  }
  s.pos[idx] = pos;
  for (int32_t word = 0; word < SEQ_WORDS; ++word) {
    uint64_t mask = s.seq_mask[idx][word];
    while (mask != 0) {
      const int32_t bit = __builtin_ctzll(mask);
      const int32_t seq_id = word * 64 + bit;
      seq_pos_add(s, seq_id, pos);
      mask &= (mask - 1);
    }
  }
}

inline void add_seq_to_cell(stream_state & s, int32_t idx, int32_t seq_id) {
  if (seq_id < 0 || seq_id >= MAX_SEQ) {
    return;
  }
  if (cell_has_seq(s, idx, seq_id)) {
    return;
  }
  set_seq_bit(s, idx, seq_id);
  ++s.seq_count[idx];
  if (s.pos[idx] != POS_NONE) {
    seq_pos_add(s, seq_id, s.pos[idx]);
  }
}

inline void remove_seq_from_cell(stream_state & s, int32_t idx, int32_t seq_id) {
  if (!cell_has_seq(s, idx, seq_id)) {
    return;
  }
  clear_seq_bit(s, idx, seq_id);
  if (s.seq_count[idx] > 0) {
    --s.seq_count[idx];
  }
  if (s.pos[idx] != POS_NONE) {
    seq_pos_remove(s, seq_id, s.pos[idx]);
  }
  if (s.seq_count[idx] == 0) {
    set_cell_empty(s, idx);
  }
}

inline bool pos_add_cell(stream_state & s, int32_t idx, int32_t delta) {
  if (s.pos[idx] == POS_NONE) {
    return false;
  }
  const int32_t old_pos = s.pos[idx];
  const int32_t new_pos = old_pos + delta;
  for (int32_t word = 0; word < SEQ_WORDS; ++word) {
    uint64_t mask = s.seq_mask[idx][word];
    while (mask != 0) {
      const int32_t bit = __builtin_ctzll(mask);
      const int32_t seq_id = word * 64 + bit;
      seq_pos_remove(s, seq_id, old_pos);
      mask &= (mask - 1);
    }
  }
  if (new_pos < 0) {
    set_cell_empty(s, idx);
    return true;
  }
  s.pos[idx] = new_pos;
  s.shift[idx] += delta;
  s.has_shift = true;
  for (int32_t word = 0; word < SEQ_WORDS; ++word) {
    uint64_t mask = s.seq_mask[idx][word];
    while (mask != 0) {
      const int32_t bit = __builtin_ctzll(mask);
      const int32_t seq_id = word * 64 + bit;
      seq_pos_add(s, seq_id, new_pos);
      mask &= (mask - 1);
    }
  }
  return false;
}

inline void pos_div_cell(stream_state & s, int32_t idx, int32_t divisor) {
  if (s.pos[idx] == POS_NONE) {
    return;
  }
  const int32_t old_pos = s.pos[idx];
  const int32_t new_pos = divisor == 0 ? old_pos : old_pos / divisor;
  for (int32_t word = 0; word < SEQ_WORDS; ++word) {
    uint64_t mask = s.seq_mask[idx][word];
    while (mask != 0) {
      const int32_t bit = __builtin_ctzll(mask);
      const int32_t seq_id = word * 64 + bit;
      seq_pos_remove(s, seq_id, old_pos);
      mask &= (mask - 1);
    }
  }
  s.pos[idx] = new_pos;
  s.shift[idx] += old_pos - new_pos;
  s.has_shift = true;
  for (int32_t word = 0; word < SEQ_WORDS; ++word) {
    uint64_t mask = s.seq_mask[idx][word];
    while (mask != 0) {
      const int32_t bit = __builtin_ctzll(mask);
      const int32_t seq_id = word * 64 + bit;
      seq_pos_add(s, seq_id, new_pos);
      mask &= (mask - 1);
    }
  }
}

inline int32_t max_used_max_p1(const context & ctx) {
  int32_t max_used = 0;
  for (int32_t s = 0; s < ctx.n_stream; ++s) {
    max_used = std::max(max_used, ctx.streams[s].used_max_p1);
  }
  return max_used;
}

inline bool is_masked_swa(int32_t n_swa, int32_t pos_cell, int32_t pos_max) {
  if (n_swa <= 0) {
    return false;
  }
  if (pos_max < 0) {
    return false;
  }
  return pos_cell + n_swa < pos_max;
}

inline bool range_overlaps_planned(
    const context & ctx,
    int32_t stream_id,
    int32_t start,
    int32_t size,
    int32_t upto) {
  for (int32_t i = 0; i < upto; ++i) {
    if (ctx.ubatch_stream_ids[i] != stream_id) {
      continue;
    }
    if (ranges_overlap(start, size, ctx.slot_offsets[i], ctx.ubatch_sizes[i])) {
      return true;
    }
  }
  return false;
}

inline int32_t find_contiguous_slot(
    const context & ctx,
    const stream_state & s,
    int32_t kv_size,
    int32_t head_start,
    int32_t n_tokens,
    int32_t used_cells,
    int32_t stream_id,
    int32_t ubatch_index,
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
      if (s.pos[head_cur + i] != POS_NONE) {
        can_use = false;
        break;
      }
    }
    if (can_use &&
        range_overlaps_planned(ctx, stream_id, head_cur, n_tokens, ubatch_index)) {
      can_use = false;
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

inline void ensure_next_pos_for_seq(context & ctx, int32_t seq_id, const stream_state & s) {
  if (seq_id < 0 || seq_id >= MAX_SEQ) {
    return;
  }
  if (s.seq_pos_max[seq_id] != POS_NONE &&
      ctx.next_pos[seq_id] <= s.seq_pos_max[seq_id]) {
    ctx.next_pos[seq_id] = s.seq_pos_max[seq_id] + 1;
  }
}

inline void reset_next_pos_for_seq(context & ctx, int32_t seq_id, const stream_state & s) {
  if (seq_id < 0 || seq_id >= MAX_SEQ) {
    return;
  }
  if (s.seq_pos_max[seq_id] == POS_NONE) {
    ctx.next_pos[seq_id] = 0;
  } else {
    ctx.next_pos[seq_id] = s.seq_pos_max[seq_id] + 1;
  }
}

inline void add_pending_copy(context & ctx, int32_t src_stream, int32_t dst_stream) {
  if (src_stream == dst_stream) {
    return;
  }
  for (int32_t i = 0; i < ctx.pending_copy_count; ++i) {
    if (ctx.pending_copy_src[i] == src_stream && ctx.pending_copy_dst[i] == dst_stream) {
      return;
    }
  }
  if (ctx.pending_copy_count < MAX_STREAM_COPY) {
    ctx.pending_copy_src[ctx.pending_copy_count] = src_stream;
    ctx.pending_copy_dst[ctx.pending_copy_count] = dst_stream;
    ++ctx.pending_copy_count;
  }
}

inline constexpr auto begin_prepare = [](const event::prepare & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  if (ev.ubatch_count_out != nullptr) {
    *ev.ubatch_count_out = 0;
  }
  ctx.ubatch_count = ev.ubatch_count;
  ctx.planned_ubatch_count = 0;

  ctx.ubatch_sizes.fill(0);
  ctx.slot_offsets.fill(0);
  ctx.ubatch_stream_ids.fill(0);
  ctx.ubatch_seq_ids.fill(0);

  if (ev.ubatch_sizes != nullptr && ev.ubatch_count > 0 && ev.ubatch_count <= MAX_UBATCHES) {
    for (int32_t i = 0; i < ev.ubatch_count; ++i) {
      ctx.ubatch_sizes[i] = ev.ubatch_sizes[i];
    }
  }

  if (ev.ubatch_stream_ids != nullptr && ev.ubatch_stream_ids_count >= ev.ubatch_count) {
    for (int32_t i = 0; i < ev.ubatch_count; ++i) {
      ctx.ubatch_stream_ids[i] = ev.ubatch_stream_ids[i];
    }
  }

  if (ev.ubatch_seq_ids != nullptr && ev.ubatch_seq_ids_count >= ev.ubatch_count) {
    for (int32_t i = 0; i < ev.ubatch_count; ++i) {
      ctx.ubatch_seq_ids[i] = ev.ubatch_seq_ids[i];
    }
  } else {
    for (int32_t i = 0; i < ev.ubatch_count; ++i) {
      ctx.ubatch_seq_ids[i] = ctx.ubatch_stream_ids[i];
    }
  }

  const int32_t prev_streams = ctx.n_stream;
  if (ev.n_stream > 0) {
    ctx.n_stream = std::min(ev.n_stream, MAX_STREAMS);
  } else if (ctx.n_stream <= 0) {
    ctx.n_stream = 1;
  }

  if (ev.seq_to_stream != nullptr && ev.seq_to_stream_count > 0) {
    const int32_t count = std::min(ev.seq_to_stream_count, MAX_SEQ);
    ctx.seq_to_stream.fill(0);
    for (int32_t i = 0; i < count; ++i) {
      ctx.seq_to_stream[i] = ev.seq_to_stream[i];
    }
  } else {
    ctx.seq_to_stream.fill(0);
    if (ctx.n_stream > 1) {
      for (int32_t i = 0; i < ctx.n_stream; ++i) {
        ctx.seq_to_stream[i] = i;
      }
    }
  }

  const int32_t prev_kv_size = ctx.kv_size;
  if (ev.requested_capacity > 0 && ev.requested_capacity <= MAX_KV_CELLS) {
    ctx.kv_size = std::max(ctx.kv_size, ev.requested_capacity);
  }

  if (prev_kv_size == 0 || prev_streams != ctx.n_stream) {
    for (int32_t s = 0; s < ctx.n_stream; ++s) {
      reset_stream(ctx.streams[s]);
    }
    ctx.next_pos.fill(0);
  } else if (ctx.kv_size > prev_kv_size) {
    for (int32_t s = 0; s < ctx.n_stream; ++s) {
      auto & stream = ctx.streams[s];
      for (int32_t i = prev_kv_size; i < ctx.kv_size; ++i) {
        stream.pos[i] = POS_NONE;
        stream.shift[i] = 0;
        stream.ext_x[i] = 0;
        stream.ext_y[i] = 0;
        stream.seq_count[i] = 0;
        for (int32_t w = 0; w < SEQ_WORDS; ++w) {
          stream.seq_mask[i][w] = 0;
        }
      }
      if (stream.head >= ctx.kv_size) {
        stream.head = 0;
      }
      recompute_used_max_p1(stream);
    }
  }

  if (ctx.kv_size > 0) {
    for (int32_t s = 0; s < ctx.n_stream; ++s) {
      auto & stream = ctx.streams[s];
      if (stream.head >= ctx.kv_size) {
        stream.head %= ctx.kv_size;
        recompute_used_max_p1(stream);
      }
    }
  }

  ctx.kv_tokens = max_used_max_p1(ctx);
};

inline constexpr auto begin_apply = [](const event::apply_ubatch & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  if (ev.kv_tokens_out != nullptr) {
    *ev.kv_tokens_out = 0;
  }
};

inline constexpr auto begin_rollback = [](const event::rollback & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
};

inline constexpr auto run_validate_prepare = [](const event::validate_prepare & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
};

inline constexpr auto run_validate_apply = [](const event::validate_apply & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
};

inline constexpr auto run_validate_rollback = [](const event::validate_rollback & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
};

inline constexpr auto run_prepare_slots = [](const event::prepare_slots & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  std::array<int32_t, MAX_STREAMS> head_sim = {};
  std::array<int32_t, MAX_STREAMS> used_sim = {};
  for (int32_t s = 0; s < ctx.n_stream; ++s) {
    head_sim[s] = ctx.streams[s].head;
    used_sim[s] = ctx.streams[s].used_count;
  }

  for (int32_t i = 0; i < ctx.ubatch_count; ++i) {
    const int32_t size = ctx.ubatch_sizes[i];
    const int32_t stream_id = ctx.ubatch_stream_ids[i];
    const stream_state & stream = ctx.streams[stream_id];

    int32_t head_after = head_sim[stream_id];
    const int32_t slot = find_contiguous_slot(
        ctx,
        stream,
        ctx.kv_size,
        head_sim[stream_id],
        size,
        used_sim[stream_id],
        stream_id,
        i,
        head_after);
    if (slot < 0) {
      *ev.error_out = EMEL_ERR_BACKEND;
      return;
    }

    ctx.slot_offsets[i] = slot;
    head_sim[stream_id] = head_after;
    used_sim[stream_id] = std::min(ctx.kv_size, used_sim[stream_id] + size);
  }

  ctx.planned_ubatch_count = ctx.ubatch_count;
  ctx.applied_ubatches = 0;
  ctx.kv_tokens = max_used_max_p1(ctx);
};

inline constexpr auto run_apply_step = [](const event::apply_step & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  const event::apply_ubatch * request = ev.request;
  const int32_t ubatch_index = request->ubatch_index;
  const int32_t size = ctx.ubatch_sizes[ubatch_index];
  const int32_t start = ctx.slot_offsets[ubatch_index];
  const int32_t stream_id = ctx.ubatch_stream_ids[ubatch_index];
  const int32_t seq_id = ctx.ubatch_seq_ids[ubatch_index];
  stream_state & stream = ctx.streams[stream_id];
  ensure_next_pos_for_seq(ctx, seq_id, stream);
  int32_t next_pos = ctx.next_pos[seq_id];

  for (int32_t i = 0; i < size; ++i) {
    const int32_t idx = start + i;
    if (stream.pos[idx] != POS_NONE) {
      *ev.error_out = EMEL_ERR_BACKEND;
      return;
    }
    set_cell_pos(stream, idx, next_pos);
    add_seq_to_cell(stream, idx, seq_id);
    ++next_pos;
  }

  ctx.next_pos[seq_id] = next_pos;
  stream.head = start + size;
  if (stream.head >= ctx.kv_size) {
    stream.head %= ctx.kv_size;
  }

  ctx.applied_ubatches = ubatch_index + 1;
  ctx.kv_tokens = max_used_max_p1(ctx);
};

inline constexpr auto run_rollback_step = [](const event::rollback_step & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  const event::rollback * request = ev.request;
  const int32_t from_index = request->from_ubatch_index;
  std::array<int32_t, MAX_STREAMS> new_head = {};
  for (int32_t s = 0; s < ctx.n_stream; ++s) {
    new_head[s] = ctx.kv_size;
  }
  for (int32_t i = from_index; i < ctx.applied_ubatches; ++i) {
    const int32_t size = ctx.ubatch_sizes[i];
    const int32_t start = ctx.slot_offsets[i];
    const int32_t stream_id = ctx.ubatch_stream_ids[i];
    const int32_t seq_id = ctx.ubatch_seq_ids[i];
    stream_state & stream = ctx.streams[stream_id];

    for (int32_t j = 0; j < size; ++j) {
      remove_seq_from_cell(stream, start + j, seq_id);
    }
    new_head[stream_id] = std::min(new_head[stream_id], start);
    reset_next_pos_for_seq(ctx, seq_id, stream);
  }

  for (int32_t s = 0; s < ctx.n_stream; ++s) {
    if (new_head[s] < ctx.kv_size && new_head[s] < ctx.streams[s].head) {
      ctx.streams[s].head = new_head[s];
    }
  }

  ctx.applied_ubatches = from_index;
  ctx.kv_tokens = max_used_max_p1(ctx);
};

inline constexpr auto begin_seq_remove = [](const event::seq_remove & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
};

inline constexpr auto begin_seq_copy = [](const event::seq_copy & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
};

inline constexpr auto begin_seq_keep = [](const event::seq_keep & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
};

inline constexpr auto begin_seq_add = [](const event::seq_add & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
};

inline constexpr auto begin_seq_div = [](const event::seq_div & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
};

inline constexpr auto begin_apply_updates = [](const event::apply_updates & ev, context &) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
};

inline constexpr auto run_validate_seq_remove = [](const event::validate_seq_remove & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
};

inline constexpr auto run_validate_seq_copy = [](const event::validate_seq_copy & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
};

inline constexpr auto run_validate_seq_keep = [](const event::validate_seq_keep & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
};

inline constexpr auto run_validate_seq_add = [](const event::validate_seq_add & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
};

inline constexpr auto run_validate_seq_div = [](const event::validate_seq_div & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
};

inline constexpr auto run_validate_updates = [](const event::validate_updates & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
};

inline constexpr auto reject_invalid_prepare = [](const event::validate_prepare & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
};

inline constexpr auto reject_invalid_apply = [](const event::validate_apply & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
};

inline constexpr auto reject_invalid_rollback = [](const event::validate_rollback & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
};

inline constexpr auto reject_invalid_prepare_slots = [](const event::prepare_slots & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
};

inline constexpr auto reject_invalid_apply_step = [](const event::apply_step & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
};

inline constexpr auto reject_invalid_rollback_step = [](const event::rollback_step & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
};

inline constexpr auto reject_invalid_seq_remove = [](const event::validate_seq_remove & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
};

inline constexpr auto reject_invalid_seq_copy = [](const event::validate_seq_copy & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
};

inline constexpr auto reject_invalid_seq_keep = [](const event::validate_seq_keep & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
};

inline constexpr auto reject_invalid_seq_add = [](const event::validate_seq_add & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
};

inline constexpr auto reject_invalid_seq_div = [](const event::validate_seq_div & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
};

inline constexpr auto reject_invalid_updates = [](const event::validate_updates & ev, context &) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
};

inline constexpr auto run_seq_remove_step = [](const event::seq_remove_step & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  const event::seq_remove * request = ev.request;
  const int32_t seq_id = request->seq_id;
  const int32_t stream_id = ctx.seq_to_stream[seq_id];
  stream_state & stream = ctx.streams[stream_id];
  const int32_t pos0 = request->pos_start;
  const int32_t pos1 = request->pos_end;

  for (int32_t i = 0; i < ctx.kv_size; ++i) {
    if (stream.pos[i] == POS_NONE) {
      continue;
    }
    if (!cell_has_seq(stream, i, seq_id)) {
      continue;
    }
    if (!pos_in_range(stream.pos[i], pos0, pos1)) {
      continue;
    }
    remove_seq_from_cell(stream, i, seq_id);
  }

  reset_next_pos_for_seq(ctx, seq_id, stream);
  ctx.kv_tokens = max_used_max_p1(ctx);
};

inline constexpr auto run_seq_copy_step = [](const event::seq_copy_step & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  const event::seq_copy * request = ev.request;
  const int32_t src_seq = request->seq_id_src;
  const int32_t dst_seq = request->seq_id_dst;
  if (src_seq == dst_seq) {
    return;
  }

  const int32_t src_stream_id = ctx.seq_to_stream[src_seq];
  const int32_t dst_stream_id = ctx.seq_to_stream[dst_seq];
  stream_state & src_stream = ctx.streams[src_stream_id];
  stream_state & dst_stream = ctx.streams[dst_stream_id];

  const int32_t pos0 = request->pos_start;
  const int32_t pos1 = request->pos_end;

  if (src_stream_id != dst_stream_id) {
    bool found_pair = false;
    for (int32_t i = 0; i < ctx.pending_copy_count; ++i) {
      if (ctx.pending_copy_src[i] == src_stream_id &&
          ctx.pending_copy_dst[i] == dst_stream_id) {
        found_pair = true;
        break;
      }
    }
    if (!found_pair && ctx.pending_copy_count >= MAX_STREAM_COPY) {
      *ev.error_out = EMEL_ERR_BACKEND;
      return;
    }
    if (!found_pair) {
      add_pending_copy(ctx, src_stream_id, dst_stream_id);
    }
  }

  for (int32_t i = 0; i < ctx.kv_size; ++i) {
    if (src_stream.pos[i] == POS_NONE) {
      continue;
    }
    if (!cell_has_seq(src_stream, i, src_seq)) {
      continue;
    }
    if (!pos_in_range(src_stream.pos[i], pos0, pos1)) {
      continue;
    }

    if (src_stream_id == dst_stream_id) {
      add_seq_to_cell(dst_stream, i, dst_seq);
      continue;
    }

    if (dst_stream.pos[i] != POS_NONE) {
      *ev.error_out = EMEL_ERR_BACKEND;
      return;
    }
    set_cell_pos(dst_stream, i, src_stream.pos[i]);
    add_seq_to_cell(dst_stream, i, dst_seq);
  }

  reset_next_pos_for_seq(ctx, dst_seq, dst_stream);
  ctx.kv_tokens = max_used_max_p1(ctx);
};

inline constexpr auto run_seq_keep_step = [](const event::seq_keep_step & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  const event::seq_keep * request = ev.request;
  const int32_t seq_id = request->seq_id;
  const int32_t stream_id = ctx.seq_to_stream[seq_id];
  stream_state & stream = ctx.streams[stream_id];
  for (int32_t i = 0; i < ctx.kv_size; ++i) {
    if (stream.pos[i] == POS_NONE) {
      continue;
    }
    if (!cell_has_seq(stream, i, seq_id)) {
      set_cell_empty(stream, i);
      continue;
    }
    for (int32_t word = 0; word < SEQ_WORDS; ++word) {
      uint64_t mask = stream.seq_mask[i][word];
      while (mask != 0) {
        const int32_t bit = __builtin_ctzll(mask);
        const int32_t other_seq = word * 64 + bit;
        mask &= (mask - 1);
        if (other_seq == seq_id) {
          continue;
        }
        remove_seq_from_cell(stream, i, other_seq);
      }
    }
  }

  reset_next_pos_for_seq(ctx, seq_id, stream);
  ctx.kv_tokens = max_used_max_p1(ctx);
};

inline constexpr auto run_seq_add_step = [](const event::seq_add_step & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  const event::seq_add * request = ev.request;
  const int32_t seq_id = request->seq_id;
  const int32_t stream_id = ctx.seq_to_stream[seq_id];
  stream_state & stream = ctx.streams[stream_id];
  const int32_t pos0 = request->pos_start;
  const int32_t pos1 = request->pos_end;
  const int32_t delta = request->shift;

  for (int32_t i = 0; i < ctx.kv_size; ++i) {
    if (stream.pos[i] == POS_NONE) {
      continue;
    }
    if (!cell_has_seq(stream, i, seq_id)) {
      continue;
    }
    if (!pos_in_range(stream.pos[i], pos0, pos1)) {
      continue;
    }
    pos_add_cell(stream, i, delta);
  }

  reset_next_pos_for_seq(ctx, seq_id, stream);
  ctx.kv_tokens = max_used_max_p1(ctx);
};

inline constexpr auto run_seq_div_step = [](const event::seq_div_step & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  const event::seq_div * request = ev.request;
  const int32_t seq_id = request->seq_id;
  const int32_t stream_id = ctx.seq_to_stream[seq_id];
  stream_state & stream = ctx.streams[stream_id];
  const int32_t pos0 = request->pos_start;
  const int32_t pos1 = request->pos_end;
  const int32_t divisor = request->divisor;

  for (int32_t i = 0; i < ctx.kv_size; ++i) {
    if (stream.pos[i] == POS_NONE) {
      continue;
    }
    if (!cell_has_seq(stream, i, seq_id)) {
      continue;
    }
    if (!pos_in_range(stream.pos[i], pos0, pos1)) {
      continue;
    }
    pos_div_cell(stream, i, divisor);
  }

  reset_next_pos_for_seq(ctx, seq_id, stream);
  ctx.kv_tokens = max_used_max_p1(ctx);
};

inline constexpr auto run_apply_updates = [](const event::apply_updates_step & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  if (ctx.pending_copy_count > 0) {
    ctx.pending_copy_count = 0;
  }
  for (int32_t s = 0; s < ctx.n_stream; ++s) {
    stream_state & stream = ctx.streams[s];
    if (!stream.has_shift) {
      continue;
    }
    stream.has_shift = false;
    stream.shift.fill(0);
  }
};

inline constexpr auto run_publish = [](const event::publish & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
};

inline constexpr auto on_kv_done = [](const events::kv_done &, context &) {};

inline constexpr auto on_kv_error = [](const events::kv_error &, context &) {};

struct on_unexpected {
  template <class Event>
  void operator()(const Event & ev, context &) const {
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
  }
};

}  // namespace emel::kv::cache::action
