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

struct cell_snapshot {
  int32_t pos = POS_NONE;
  int32_t shift = 0;
  int32_t ext_x = 0;
  int32_t ext_y = 0;
  uint16_t seq_count = 0;
  std::array<uint64_t, SEQ_WORDS> seq_mask = {};
};

struct stream_snapshot {
  int32_t head = 0;
  int32_t used_count = 0;
  int32_t used_max_p1 = 0;
  bool has_shift = false;
  std::array<int32_t, MAX_SEQ> seq_pos_min = {};
  std::array<int32_t, MAX_SEQ> seq_pos_max = {};
  std::array<uint16_t, MAX_SEQ> seq_pos_min_count = {};
  std::array<uint16_t, MAX_SEQ> seq_pos_max_count = {};
};

struct context {
  std::array<int32_t, MAX_UBATCHES> ubatch_sizes = {};
  std::array<int32_t, MAX_UBATCHES> slot_offsets = {};
  std::array<int32_t, MAX_KV_CELLS> slot_indices = {};
  std::array<int32_t, MAX_KV_CELLS> slot_stream_ids = {};
  std::array<cell_snapshot, MAX_KV_CELLS> slot_snapshots = {};
  int32_t slot_index_count = 0;
  std::array<int32_t, MAX_UBATCHES> ubatch_stream_ids = {};
  std::array<int32_t, MAX_UBATCHES> ubatch_seq_ids = {};
  int32_t ubatch_count = 0;
  int32_t planned_ubatch_count = 0;
  int32_t applied_ubatches = 0;

  int32_t kv_size = 0;
  int32_t n_stream = 1;
  int32_t n_pad = 1;
  int32_t n_swa = 0;
  int32_t swa_type = 0;
  std::array<int32_t, MAX_SEQ> seq_to_stream = {};
  std::array<int32_t, MAX_SEQ> next_pos = {};
  std::array<stream_state, MAX_STREAMS> streams = {};
  std::array<stream_snapshot, MAX_STREAMS> prepare_snapshot = {};

  std::array<int32_t, MAX_STREAM_COPY> pending_copy_src = {};
  std::array<int32_t, MAX_STREAM_COPY> pending_copy_dst = {};
  int32_t pending_copy_count = 0;

  int32_t kv_tokens = 0;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;

  event::prepare prepare_request = {};
  event::apply_ubatch apply_request = {};
  event::rollback rollback_request = {};
  event::seq_remove seq_remove_request = {};
  event::seq_copy seq_copy_request = {};
  event::seq_keep seq_keep_request = {};
  event::seq_add seq_add_request = {};
  event::seq_div seq_div_request = {};
  event::apply_updates updates_request = {};

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
  ctx.slot_indices.fill(0);
  ctx.slot_stream_ids.fill(0);
  ctx.slot_index_count = 0;
  ctx.ubatch_stream_ids.fill(0);
  ctx.ubatch_seq_ids.fill(0);
  ctx.ubatch_count = 0;
  ctx.planned_ubatch_count = 0;
  ctx.applied_ubatches = 0;
  ctx.kv_size = 0;
  ctx.n_stream = 1;
  ctx.n_pad = 1;
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
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  ctx.prepare_request = {};
  ctx.apply_request = {};
  ctx.rollback_request = {};
  ctx.seq_remove_request = {};
  ctx.seq_copy_request = {};
  ctx.seq_keep_request = {};
  ctx.seq_add_request = {};
  ctx.seq_div_request = {};
  ctx.updates_request = {};
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

inline void snapshot_cell(const stream_state & s, int32_t idx, cell_snapshot & snap) {
  snap.pos = s.pos[idx];
  snap.shift = s.shift[idx];
  snap.ext_x = s.ext_x[idx];
  snap.ext_y = s.ext_y[idx];
  snap.seq_count = s.seq_count[idx];
  snap.seq_mask = s.seq_mask[idx];
}

inline void restore_cell(stream_state & s, int32_t idx, const cell_snapshot & snap) {
  s.pos[idx] = snap.pos;
  s.shift[idx] = snap.shift;
  s.ext_x[idx] = snap.ext_x;
  s.ext_y[idx] = snap.ext_y;
  s.seq_count[idx] = snap.seq_count;
  s.seq_mask[idx] = snap.seq_mask;
}

inline void snapshot_stream_state(const stream_state & s, stream_snapshot & snap) {
  snap.head = s.head;
  snap.used_count = s.used_count;
  snap.used_max_p1 = s.used_max_p1;
  snap.has_shift = s.has_shift;
  snap.seq_pos_min = s.seq_pos_min;
  snap.seq_pos_max = s.seq_pos_max;
  snap.seq_pos_min_count = s.seq_pos_min_count;
  snap.seq_pos_max_count = s.seq_pos_max_count;
}

inline void restore_stream_state(stream_state & s, const stream_snapshot & snap) {
  s.head = snap.head;
  s.used_count = snap.used_count;
  s.used_max_p1 = snap.used_max_p1;
  s.has_shift = snap.has_shift;
  s.seq_pos_min = snap.seq_pos_min;
  s.seq_pos_max = snap.seq_pos_max;
  s.seq_pos_min_count = snap.seq_pos_min_count;
  s.seq_pos_max_count = snap.seq_pos_max_count;
}

inline int32_t single_seq_id(const stream_state & s, int32_t idx) {
  if (s.seq_count[idx] != 1) {
    return POS_NONE;
  }
  for (int32_t word = 0; word < SEQ_WORDS; ++word) {
    const uint64_t mask = s.seq_mask[idx][word];
    if (mask == 0) {
      continue;
    }
    const int32_t bit = __builtin_ctzll(mask);
    return word * 64 + bit;
  }
  return POS_NONE;
}

inline int32_t pad_to(int32_t value, int32_t pad) {
  if (pad <= 0) {
    return value;
  }
  return ((value + pad - 1) / pad) * pad;
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

inline bool is_full_copy_range(int32_t pos0, int32_t pos1, int32_t kv_size) {
  if (kv_size <= 0) {
    return false;
  }
  bool full = true;
  if (pos0 > 0 && pos0 + 1 < kv_size) {
    full = false;
  }
  if (pos1 > 0 && pos1 + 1 < kv_size) {
    full = false;
  }
  return full;
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

inline int32_t compute_kv_tokens(const context & ctx) {
  if (ctx.kv_size <= 0) {
    return 0;
  }
  const int32_t n_pad_cur = std::max(ctx.n_pad, 256);
  int32_t result = 0;
  for (int32_t s = 0; s < ctx.n_stream; ++s) {
    const int32_t used = ctx.streams[s].used_max_p1;
    int32_t padded = std::max(n_pad_cur, pad_to(used, n_pad_cur));
    if (padded > ctx.kv_size) {
      padded = ctx.kv_size;
    }
    result = std::max(result, padded);
  }
  return result;
}

inline bool is_masked_swa(int32_t n_swa, int32_t swa_type, int32_t pos_cell, int32_t pos_max_p1) {
  if (n_swa <= 0 || swa_type == 0) {
    return false;
  }
  if (pos_max_p1 <= 0) {
    return false;
  }
  return pos_cell + n_swa < pos_max_p1;
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

inline bool find_slot_indices(
    const context & ctx,
    const stream_state & s,
    int32_t kv_size,
    int32_t n_tokens,
    int32_t * indices_out) {
  if (kv_size <= 0 || n_tokens <= 0 || n_tokens > kv_size) {
    return false;
  }

  int32_t head_cur = s.head;
  if (head_cur > s.used_count + 2 * n_tokens) {
    head_cur = 0;
  }

  int32_t n_tested = 0;
  int32_t found = 0;
  while (n_tested < kv_size) {
    if (head_cur + 1 > kv_size) {
      n_tested += kv_size - head_cur;
      head_cur = 0;
      continue;
    }

    bool can_use = s.pos[head_cur] == POS_NONE;
    if (!can_use && s.seq_count[head_cur] == 1) {
      const int32_t seq_id_cell = single_seq_id(s, head_cur);
      if (seq_id_cell != POS_NONE) {
        const int32_t pos_cell = s.pos[head_cur];
        const int32_t pos_max = s.seq_pos_max[seq_id_cell];
        if (pos_max != POS_NONE &&
            is_masked_swa(ctx.n_swa, ctx.swa_type, pos_cell, pos_max + 1)) {
          can_use = true;
        }
      }
    }

    if (can_use) {
      indices_out[found++] = head_cur;
      if (found == n_tokens) {
        return true;
      }
    }

    ++head_cur;
    ++n_tested;
  }

  return false;
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

inline bool apply_slots(
    context & ctx,
    int32_t slot_offset,
    int32_t slot_count,
    int32_t seq_id,
    const int32_t * positions,
    int32_t positions_count,
    bool update_next_pos) {
  if (slot_offset < 0 || slot_count <= 0) {
    return false;
  }
  if (slot_offset + slot_count > ctx.slot_index_count) {
    return false;
  }
  if (positions != nullptr && positions_count < slot_count) {
    return false;
  }

  std::array<int32_t, MAX_SEQ> seq_pos_max_rm = {};
  seq_pos_max_rm.fill(POS_NONE);

  const bool has_positions = positions != nullptr && positions_count >= slot_count;
  const bool has_pos_2d = positions != nullptr && positions_count >= slot_count * 3;

  int32_t next_pos = 0;
  if (!has_positions) {
    const int32_t stream_id = ctx.seq_to_stream[seq_id];
    if (stream_id >= 0 && stream_id < ctx.n_stream) {
      ensure_next_pos_for_seq(ctx, seq_id, ctx.streams[stream_id]);
    }
    next_pos = ctx.next_pos[seq_id];
  }

  std::array<int32_t, MAX_STREAMS> last_idx = {};
  last_idx.fill(POS_NONE);

  for (int32_t i = 0; i < slot_count; ++i) {
    const int32_t slot = slot_offset + i;
    const int32_t idx = ctx.slot_indices[slot];
    const int32_t stream_id = ctx.slot_stream_ids[slot];
    if (stream_id < 0 || stream_id >= ctx.n_stream) {
      return false;
    }
    if (idx < 0 || idx >= ctx.kv_size) {
      return false;
    }
    stream_state & stream = ctx.streams[stream_id];

    if (stream.pos[idx] != POS_NONE) {
      for (int32_t word = 0; word < SEQ_WORDS; ++word) {
        uint64_t mask = stream.seq_mask[idx][word];
        while (mask != 0) {
          const int32_t bit = __builtin_ctzll(mask);
          const int32_t seq_over = word * 64 + bit;
          seq_pos_max_rm[seq_over] =
              std::max(seq_pos_max_rm[seq_over], stream.pos[idx]);
          mask &= (mask - 1);
        }
      }
      set_cell_empty(stream, idx);
    }

    const int32_t pos = has_positions ? positions[i] : next_pos++;
    set_cell_pos(stream, idx, pos);
    if (has_pos_2d) {
      stream.ext_y[idx] = positions[i + slot_count];
      stream.ext_x[idx] = positions[i + slot_count * 2];
    }
    add_seq_to_cell(stream, idx, seq_id);

    if (last_idx[stream_id] == POS_NONE || idx > last_idx[stream_id]) {
      last_idx[stream_id] = idx;
    }
  }

  if (has_positions && update_next_pos) {
    int32_t max_pos = ctx.next_pos[seq_id];
    for (int32_t i = 0; i < slot_count; ++i) {
      if (positions[i] >= max_pos) {
        max_pos = positions[i] + 1;
      }
    }
    ctx.next_pos[seq_id] = max_pos;
  } else if (update_next_pos) {
    ctx.next_pos[seq_id] = next_pos;
  }

  for (int32_t seq = 0; seq < MAX_SEQ; ++seq) {
    if (seq_pos_max_rm[seq] == POS_NONE) {
      continue;
    }
    const int32_t stream_id = ctx.seq_to_stream[seq];
    if (stream_id < 0 || stream_id >= ctx.n_stream) {
      continue;
    }
    stream_state & stream = ctx.streams[stream_id];
    const int32_t min_pos = stream.seq_pos_min[seq];
    if (min_pos == POS_NONE) {
      continue;
    }
    if (min_pos > seq_pos_max_rm[seq]) {
      continue;
    }
    const int32_t end_pos = seq_pos_max_rm[seq] + 1;
    for (int32_t i = 0; i < ctx.kv_size; ++i) {
      if (stream.pos[i] == POS_NONE) {
        continue;
      }
      if (!cell_has_seq(stream, i, seq)) {
        continue;
      }
      if (!pos_in_range(stream.pos[i], min_pos, end_pos)) {
        continue;
      }
      remove_seq_from_cell(stream, i, seq);
    }
    reset_next_pos_for_seq(ctx, seq, stream);
  }

  for (int32_t s = 0; s < ctx.n_stream; ++s) {
    if (last_idx[s] == POS_NONE) {
      continue;
    }
    int32_t head = last_idx[s] + 1;
    if (head >= ctx.kv_size) {
      head = 0;
    }
    ctx.streams[s].head = head;
  }

  return true;
}

inline void store_prepare_request(const event::prepare & ev, context & ctx) noexcept {
  ctx.prepare_request = ev;
  ctx.prepare_request.slot_offsets_out = nullptr;
  ctx.prepare_request.ubatch_count_out = nullptr;
  ctx.prepare_request.error_out = nullptr;
}

inline void store_apply_request(const event::apply_ubatch & ev, context & ctx) noexcept {
  ctx.apply_request = ev;
  ctx.apply_request.kv_tokens_out = nullptr;
  ctx.apply_request.error_out = nullptr;
}

inline void store_rollback_request(const event::rollback & ev, context & ctx) noexcept {
  ctx.rollback_request = ev;
  ctx.rollback_request.error_out = nullptr;
}

inline void store_seq_remove_request(const event::seq_remove & ev, context & ctx) noexcept {
  ctx.seq_remove_request = ev;
  ctx.seq_remove_request.error_out = nullptr;
}

inline void store_seq_copy_request(const event::seq_copy & ev, context & ctx) noexcept {
  ctx.seq_copy_request = ev;
  ctx.seq_copy_request.error_out = nullptr;
}

inline void store_seq_keep_request(const event::seq_keep & ev, context & ctx) noexcept {
  ctx.seq_keep_request = ev;
  ctx.seq_keep_request.error_out = nullptr;
}

inline void store_seq_add_request(const event::seq_add & ev, context & ctx) noexcept {
  ctx.seq_add_request = ev;
  ctx.seq_add_request.error_out = nullptr;
}

inline void store_seq_div_request(const event::seq_div & ev, context & ctx) noexcept {
  ctx.seq_div_request = ev;
  ctx.seq_div_request.error_out = nullptr;
}

inline void store_updates_request(const event::apply_updates & ev, context & ctx) noexcept {
  ctx.updates_request = ev;
  ctx.updates_request.error_out = nullptr;
}

inline void clear_requests(context & ctx) noexcept {
  ctx.prepare_request = {};
  ctx.apply_request = {};
  ctx.rollback_request = {};
  ctx.seq_remove_request = {};
  ctx.seq_copy_request = {};
  ctx.seq_keep_request = {};
  ctx.seq_add_request = {};
  ctx.seq_div_request = {};
  ctx.updates_request = {};
}

inline constexpr auto begin_prepare = [](const event::prepare & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  if (ev.ubatch_count_out != nullptr) {
    *ev.ubatch_count_out = 0;
  }
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  store_prepare_request(ev, ctx);
  ctx.ubatch_count = ev.ubatch_count;
  ctx.planned_ubatch_count = 0;

  ctx.ubatch_sizes.fill(0);
  ctx.slot_offsets.fill(0);
  ctx.slot_index_count = 0;
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

  if (ev.n_pad > 0) {
    ctx.n_pad = ev.n_pad;
  } else if (ctx.n_pad <= 0) {
    ctx.n_pad = 1;
  }
  if (ev.n_swa >= 0) {
    ctx.n_swa = ev.n_swa;
  }
  ctx.swa_type = ev.swa_type;

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

  ctx.kv_tokens = compute_kv_tokens(ctx);
};

inline constexpr auto begin_apply = [](const event::apply_ubatch & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  if (ev.kv_tokens_out != nullptr) {
    *ev.kv_tokens_out = 0;
  }
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  store_apply_request(ev, ctx);
};

inline constexpr auto begin_rollback = [](const event::rollback & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  store_rollback_request(ev, ctx);
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

  ctx.slot_index_count = 0;
  std::array<int32_t, MAX_SEQ> next_pos_snapshot = ctx.next_pos;
  for (int32_t s = 0; s < ctx.n_stream; ++s) {
    snapshot_stream_state(ctx.streams[s], ctx.prepare_snapshot[s]);
  }

  bool success = true;
  for (int32_t i = 0; i < ctx.ubatch_count; ++i) {
    const int32_t size = ctx.ubatch_sizes[i];
    const int32_t stream_id = ctx.ubatch_stream_ids[i];
    const int32_t seq_id = ctx.ubatch_seq_ids[i];
    if (size <= 0 || stream_id < 0 || stream_id >= ctx.n_stream) {
      *ev.error_out = EMEL_ERR_BACKEND;
      success = false;
      break;
    }
    const int32_t slot_offset = ctx.slot_index_count;
    ctx.slot_index_count += size;
    if (ctx.slot_index_count > ctx.kv_size || ctx.slot_index_count > MAX_KV_CELLS) {
      *ev.error_out = EMEL_ERR_BACKEND;
      success = false;
      break;
    }

    ctx.slot_offsets[i] = slot_offset;
    if (!find_slot_indices(
          ctx,
          ctx.streams[stream_id],
          ctx.kv_size,
          size,
          &ctx.slot_indices[slot_offset])) {
      *ev.error_out = EMEL_ERR_BACKEND;
      success = false;
      break;
    }

    for (int32_t j = 0; j < size; ++j) {
      const int32_t slot = slot_offset + j;
      const int32_t idx = ctx.slot_indices[slot];
      ctx.slot_stream_ids[slot] = stream_id;
      snapshot_cell(ctx.streams[stream_id], idx, ctx.slot_snapshots[slot]);
    }

    if (!apply_slots(ctx, slot_offset, size, seq_id, nullptr, 0, true)) {
      *ev.error_out = EMEL_ERR_BACKEND;
      success = false;
      break;
    }
  }

  for (int32_t slot = 0; slot < ctx.slot_index_count; ++slot) {
    const int32_t stream_id = ctx.slot_stream_ids[slot];
    const int32_t idx = ctx.slot_indices[slot];
    if (stream_id < 0 || stream_id >= ctx.n_stream) {
      continue;
    }
    restore_cell(ctx.streams[stream_id], idx, ctx.slot_snapshots[slot]);
  }
  for (int32_t s = 0; s < ctx.n_stream; ++s) {
    restore_stream_state(ctx.streams[s], ctx.prepare_snapshot[s]);
  }
  ctx.next_pos = next_pos_snapshot;

  if (!success) {
    ctx.slot_index_count = 0;
    ctx.planned_ubatch_count = 0;
    ctx.applied_ubatches = 0;
    ctx.slot_offsets.fill(0);
    return;
  }

  ctx.planned_ubatch_count = ctx.ubatch_count;
  ctx.applied_ubatches = 0;
  ctx.kv_tokens = compute_kv_tokens(ctx);
};

inline constexpr auto run_apply_step = [](const event::apply_step & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  const event::apply_ubatch * request = ev.request;
  const int32_t ubatch_index = request->ubatch_index;
  const int32_t size = ctx.ubatch_sizes[ubatch_index];
  const int32_t start = ctx.slot_offsets[ubatch_index];
  const int32_t seq_id = ctx.ubatch_seq_ids[ubatch_index];
  const int32_t * positions = request->positions;
  const int32_t positions_count = request->positions_count;

  if (!apply_slots(ctx, start, size, seq_id, positions, positions_count, true)) {
    *ev.error_out = EMEL_ERR_BACKEND;
    return;
  }

  ctx.applied_ubatches = ubatch_index + 1;
  ctx.kv_tokens = compute_kv_tokens(ctx);
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
    const int32_t seq_id = ctx.ubatch_seq_ids[i];
    const int32_t start = ctx.slot_offsets[i];
    if (start < 0 || start + size > ctx.slot_index_count) {
      *ev.error_out = EMEL_ERR_BACKEND;
      return;
    }
    for (int32_t j = 0; j < size; ++j) {
      const int32_t slot = start + j;
      const int32_t stream_id = ctx.slot_stream_ids[slot];
      const int32_t idx = ctx.slot_indices[slot];
      if (stream_id < 0 || stream_id >= ctx.n_stream) {
        *ev.error_out = EMEL_ERR_BACKEND;
        return;
      }
      if (idx < 0 || idx >= ctx.kv_size) {
        *ev.error_out = EMEL_ERR_BACKEND;
        return;
      }
      stream_state & stream = ctx.streams[stream_id];
      remove_seq_from_cell(stream, idx, seq_id);
      new_head[stream_id] = std::min(new_head[stream_id], idx);
      reset_next_pos_for_seq(ctx, seq_id, stream);
    }
  }

  for (int32_t s = 0; s < ctx.n_stream; ++s) {
    if (new_head[s] < ctx.kv_size && new_head[s] < ctx.streams[s].head) {
      ctx.streams[s].head = new_head[s];
    }
  }

  ctx.applied_ubatches = from_index;
  ctx.kv_tokens = compute_kv_tokens(ctx);
};

inline constexpr auto begin_seq_remove = [](const event::seq_remove & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  store_seq_remove_request(ev, ctx);
};

inline constexpr auto begin_seq_copy = [](const event::seq_copy & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  store_seq_copy_request(ev, ctx);
};

inline constexpr auto begin_seq_keep = [](const event::seq_keep & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  store_seq_keep_request(ev, ctx);
};

inline constexpr auto begin_seq_add = [](const event::seq_add & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  store_seq_add_request(ev, ctx);
};

inline constexpr auto begin_seq_div = [](const event::seq_div & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  store_seq_div_request(ev, ctx);
};

inline constexpr auto begin_apply_updates = [](const event::apply_updates & ev, context & ctx) {
  if (ev.error_out != nullptr) {
    *ev.error_out = EMEL_OK;
  }
  ctx.phase_error = EMEL_OK;
  ctx.last_error = EMEL_OK;
  store_updates_request(ev, ctx);
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
  const int32_t pos0 = request->pos_start;
  const int32_t pos1 = request->pos_end;

  if (seq_id == -1) {
    for (int32_t s = 0; s < ctx.n_stream; ++s) {
      stream_state & stream = ctx.streams[s];
      for (int32_t i = 0; i < ctx.kv_size; ++i) {
        if (stream.pos[i] == POS_NONE) {
          continue;
        }
        if (!pos_in_range(stream.pos[i], pos0, pos1)) {
          continue;
        }
        set_cell_empty(stream, i);
      }
      for (int32_t seq = 0; seq < MAX_SEQ; ++seq) {
        if (ctx.seq_to_stream[seq] != s) {
          continue;
        }
        reset_next_pos_for_seq(ctx, seq, stream);
      }
    }
  } else {
    const int32_t stream_id = ctx.seq_to_stream[seq_id];
    stream_state & stream = ctx.streams[stream_id];
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
  }

  ctx.kv_tokens = compute_kv_tokens(ctx);
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

  if (src_stream_id != dst_stream_id &&
      !is_full_copy_range(pos0, pos1, ctx.kv_size)) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

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
    reset_stream(dst_stream);
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

    set_cell_pos(dst_stream, i, src_stream.pos[i]);
    dst_stream.shift[i] = src_stream.shift[i];
    dst_stream.ext_x[i] = src_stream.ext_x[i];
    dst_stream.ext_y[i] = src_stream.ext_y[i];
    if (dst_stream.shift[i] != 0) {
      dst_stream.has_shift = true;
    }
    add_seq_to_cell(dst_stream, i, dst_seq);
  }

  if (src_stream_id != dst_stream_id) {
    dst_stream.head = src_stream.head;
  }
  reset_next_pos_for_seq(ctx, dst_seq, dst_stream);
  ctx.kv_tokens = compute_kv_tokens(ctx);
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
  ctx.kv_tokens = compute_kv_tokens(ctx);
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

  if (delta == 0) {
    return;
  }
  if (pos0 == pos1) {
    return;
  }

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
  ctx.kv_tokens = compute_kv_tokens(ctx);
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

  if (divisor == 1) {
    return;
  }
  if (pos0 == pos1) {
    return;
  }

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
  ctx.kv_tokens = compute_kv_tokens(ctx);
};

inline constexpr auto run_apply_updates = [](const event::apply_updates_step & ev, context & ctx) {
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;

  const event::apply_updates * request = ev.request;
  if (request == nullptr) {
    *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    return;
  }

  if (ctx.pending_copy_count > 0) {
    if (request->stream_copy == nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    for (int32_t i = 0; i < ctx.pending_copy_count; ++i) {
      int32_t err = EMEL_OK;
      const bool ok = request->stream_copy(
          ctx.pending_copy_src[i],
          ctx.pending_copy_dst[i],
          request->user_data,
          &err);
      if (!ok || err != EMEL_OK) {
        *ev.error_out = err == EMEL_OK ? EMEL_ERR_BACKEND : err;
        return;
      }
    }
    ctx.pending_copy_count = 0;
  }

  bool needs_shift = false;
  for (int32_t s = 0; s < ctx.n_stream; ++s) {
    if (ctx.streams[s].has_shift) {
      needs_shift = true;
      break;
    }
  }
  if (needs_shift) {
    if (request->apply_shift == nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    for (int32_t s = 0; s < ctx.n_stream; ++s) {
      stream_state & stream = ctx.streams[s];
      if (!stream.has_shift) {
        continue;
      }
      int32_t err = EMEL_OK;
      const bool ok = request->apply_shift(
          s,
          stream.shift.data(),
          ctx.kv_size,
          request->user_data,
          &err);
      if (!ok || err != EMEL_OK) {
        *ev.error_out = err == EMEL_OK ? EMEL_ERR_BACKEND : err;
        return;
      }
      stream.has_shift = false;
      stream.shift.fill(0);
    }
  }
};

inline constexpr auto run_publish = [](const event::publish & ev, context & ctx) {
  (void)ctx;
  if (ev.error_out == nullptr) return;
  *ev.error_out = EMEL_OK;
};

struct set_invalid_argument {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
  }
};

struct run_prepare_slots_phase {
  void operator()(context & ctx) const noexcept {
    int32_t err = EMEL_OK;
    event::prepare_slots ev{.error_out = &err};
    run_prepare_slots(ev, ctx);
    ctx.phase_error = err;
  }
};

struct run_apply_step_phase {
  void operator()(context & ctx) const noexcept {
    int32_t err = EMEL_OK;
    event::apply_step ev{.request = &ctx.apply_request, .error_out = &err};
    run_apply_step(ev, ctx);
    ctx.phase_error = err;
  }
};

struct run_rollback_step_phase {
  void operator()(context & ctx) const noexcept {
    int32_t err = EMEL_OK;
    event::rollback_step ev{.request = &ctx.rollback_request, .error_out = &err};
    run_rollback_step(ev, ctx);
    ctx.phase_error = err;
  }
};

struct run_seq_remove_phase {
  void operator()(context & ctx) const noexcept {
    int32_t err = EMEL_OK;
    event::seq_remove_step ev{.request = &ctx.seq_remove_request, .error_out = &err};
    run_seq_remove_step(ev, ctx);
    ctx.phase_error = err;
  }
};

struct run_seq_copy_phase {
  void operator()(context & ctx) const noexcept {
    int32_t err = EMEL_OK;
    event::seq_copy_step ev{.request = &ctx.seq_copy_request, .error_out = &err};
    run_seq_copy_step(ev, ctx);
    ctx.phase_error = err;
  }
};

struct run_seq_keep_phase {
  void operator()(context & ctx) const noexcept {
    int32_t err = EMEL_OK;
    event::seq_keep_step ev{.request = &ctx.seq_keep_request, .error_out = &err};
    run_seq_keep_step(ev, ctx);
    ctx.phase_error = err;
  }
};

struct run_seq_add_phase {
  void operator()(context & ctx) const noexcept {
    int32_t err = EMEL_OK;
    event::seq_add_step ev{.request = &ctx.seq_add_request, .error_out = &err};
    run_seq_add_step(ev, ctx);
    ctx.phase_error = err;
  }
};

struct run_seq_div_phase {
  void operator()(context & ctx) const noexcept {
    int32_t err = EMEL_OK;
    event::seq_div_step ev{.request = &ctx.seq_div_request, .error_out = &err};
    run_seq_div_step(ev, ctx);
    ctx.phase_error = err;
  }
};

struct run_updates_phase {
  void operator()(context & ctx) const noexcept {
    int32_t err = EMEL_OK;
    event::apply_updates_step ev{.request = &ctx.updates_request, .error_out = &err};
    run_apply_updates(ev, ctx);
    ctx.phase_error = err;
  }
};

struct run_publish_phase {
  void operator()(context & ctx) const noexcept {
    int32_t err = EMEL_OK;
    event::publish ev{.error_out = &err};
    run_publish(ev, ctx);
    ctx.phase_error = err;
  }
};

struct mark_done {
  void operator()(context & ctx) const noexcept {
    ctx.last_error = EMEL_OK;
  }
};

struct ensure_last_error {
  void operator()(context & ctx) const noexcept {
    if (ctx.last_error != EMEL_OK) {
      return;
    }
    ctx.last_error = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
  }
};

struct clear_request {
  void operator()(context & ctx) const noexcept {
    clear_requests(ctx);
  }
};

inline constexpr auto on_kv_done = [](const events::kv_done &, context &) {};

inline constexpr auto on_kv_error = [](const events::kv_error &, context &) {};

struct on_unexpected {
  template <class Event>
  void operator()(const Event & ev, context & ctx) const noexcept {
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
    ctx.phase_error = EMEL_ERR_BACKEND;
    ctx.last_error = EMEL_ERR_BACKEND;
  }
};

inline constexpr set_invalid_argument set_invalid_argument{};
inline constexpr run_prepare_slots_phase run_prepare_slots_phase{};
inline constexpr run_apply_step_phase run_apply_step_phase{};
inline constexpr run_rollback_step_phase run_rollback_step_phase{};
inline constexpr run_seq_remove_phase run_seq_remove_phase{};
inline constexpr run_seq_copy_phase run_seq_copy_phase{};
inline constexpr run_seq_keep_phase run_seq_keep_phase{};
inline constexpr run_seq_add_phase run_seq_add_phase{};
inline constexpr run_seq_div_phase run_seq_div_phase{};
inline constexpr run_updates_phase run_updates_phase{};
inline constexpr run_publish_phase run_publish_phase{};
inline constexpr mark_done mark_done{};
inline constexpr ensure_last_error ensure_last_error{};
inline constexpr clear_request clear_request{};

}  // namespace emel::kv::cache::action
