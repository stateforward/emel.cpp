#pragma once

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

}  // namespace emel::kv::cache::action
