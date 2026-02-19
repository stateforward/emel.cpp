#pragma once

#include <algorithm>
#include <cstdint>

#include "emel/kv/cache/actions.hpp"
#include "emel/kv/cache/events.hpp"

namespace emel::kv::cache::guard {

struct phase_ok {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error == EMEL_OK;
  }
};

struct phase_failed {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error != EMEL_OK;
  }
};

inline bool valid_pos_range(int32_t pos0, int32_t pos1) noexcept {
  if (pos0 < 0 && pos1 < 0) {
    return true;
  }
  if (pos0 < 0 || pos1 < 0) {
    return true;
  }
  return pos0 <= pos1;
}

inline bool is_full_copy_range(int32_t pos0, int32_t pos1, int32_t kv_size) noexcept {
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

inline bool valid_stream_id(const action::context & ctx, int32_t stream_id) noexcept {
  return stream_id >= 0 && stream_id < ctx.n_stream;
}

inline bool valid_seq_id(int32_t seq_id) noexcept {
  return seq_id >= 0 && seq_id < action::MAX_SEQ;
}

inline constexpr auto valid_prepare_request = [](
    const event::validate_prepare & ev,
    const action::context & ctx) noexcept {
  const event::prepare * request = ev.request;
  if (request == nullptr) {
    return false;
  }
  if (request->ubatch_count <= 0 || request->ubatch_count > action::MAX_UBATCHES) {
    return false;
  }
  if (request->ubatch_sizes == nullptr) {
    return false;
  }
  if (request->requested_capacity > action::MAX_KV_CELLS) {
    return false;
  }
  if (ctx.n_stream <= 0 || ctx.n_stream > action::MAX_STREAMS) {
    return false;
  }
  if (request->ubatch_stream_ids != nullptr &&
      request->ubatch_stream_ids_count < request->ubatch_count) {
    return false;
  }
  if (request->ubatch_seq_ids != nullptr &&
      request->ubatch_seq_ids_count < request->ubatch_count) {
    return false;
  }
  if (request->seq_to_stream != nullptr && request->seq_to_stream_count > 0) {
    const int32_t count = std::min(request->seq_to_stream_count, action::MAX_SEQ);
    for (int32_t i = 0; i < count; ++i) {
      if (request->seq_to_stream[i] < 0 || request->seq_to_stream[i] >= ctx.n_stream) {
        return false;
      }
    }
  }
  int32_t kv_size = ctx.kv_size;
  if (request->requested_capacity > 0) {
    kv_size = std::max(kv_size, request->requested_capacity);
  }
  if (kv_size <= 0 || kv_size > action::MAX_KV_CELLS) {
    return false;
  }
  for (int32_t i = 0; i < ctx.ubatch_count; ++i) {
    const int32_t size = ctx.ubatch_sizes[i];
    if (size <= 0 || size > kv_size) {
      return false;
    }
    if (!valid_stream_id(ctx, ctx.ubatch_stream_ids[i])) {
      return false;
    }
    if (!valid_seq_id(ctx.ubatch_seq_ids[i])) {
      return false;
    }
    if (ctx.seq_to_stream[ctx.ubatch_seq_ids[i]] != ctx.ubatch_stream_ids[i]) {
      return false;
    }
  }
  int32_t total = 0;
  for (int32_t i = 0; i < ctx.ubatch_count; ++i) {
    total += ctx.ubatch_sizes[i];
    if (total > kv_size) {
      return false;
    }
  }
  return true;
};

inline constexpr auto invalid_prepare_request = [](
    const event::validate_prepare & ev,
    const action::context & ctx) noexcept {
  return !valid_prepare_request(ev, ctx);
};

inline constexpr auto valid_prepare_slots_request = [](
    const event::prepare_slots &,
    const action::context & ctx) noexcept {
  if (ctx.kv_size <= 0 || ctx.kv_size > action::MAX_KV_CELLS) {
    return false;
  }
  if (ctx.ubatch_count <= 0 || ctx.ubatch_count > action::MAX_UBATCHES) {
    return false;
  }
  if (ctx.n_stream <= 0 || ctx.n_stream > action::MAX_STREAMS) {
    return false;
  }
  for (int32_t i = 0; i < ctx.ubatch_count; ++i) {
    const int32_t size = ctx.ubatch_sizes[i];
    if (size <= 0 || size > ctx.kv_size) {
      return false;
    }
    if (!valid_stream_id(ctx, ctx.ubatch_stream_ids[i])) {
      return false;
    }
    if (!valid_seq_id(ctx.ubatch_seq_ids[i])) {
      return false;
    }
    if (ctx.seq_to_stream[ctx.ubatch_seq_ids[i]] != ctx.ubatch_stream_ids[i]) {
      return false;
    }
  }
  return true;
};

inline constexpr auto invalid_prepare_slots_request = [](
    const event::prepare_slots & ev,
    const action::context & ctx) noexcept {
  return !valid_prepare_slots_request(ev, ctx);
};

inline constexpr auto valid_apply_request = [](
    const event::validate_apply & ev,
    const action::context & ctx) noexcept {
  const event::apply_ubatch * request = ev.request;
  if (request == nullptr) {
    return false;
  }
  const int32_t ubatch_index = request->ubatch_index;
  if (ctx.planned_ubatch_count <= 0 ||
      ubatch_index < 0 ||
      ubatch_index >= ctx.planned_ubatch_count) {
    return false;
  }
  if (ubatch_index != ctx.applied_ubatches) {
    return false;
  }
  return true;
};

inline constexpr auto invalid_apply_request = [](
    const event::validate_apply & ev,
    const action::context & ctx) noexcept {
  return !valid_apply_request(ev, ctx);
};

inline constexpr auto valid_apply_step_request = [](
    const event::apply_step & ev,
    const action::context & ctx) noexcept {
  const event::apply_ubatch * request = ev.request;
  if (request == nullptr) {
    return false;
  }
  const int32_t ubatch_index = request->ubatch_index;
  if (ctx.planned_ubatch_count <= 0 ||
      ubatch_index < 0 ||
      ubatch_index >= ctx.planned_ubatch_count) {
    return false;
  }
  if (ubatch_index != ctx.applied_ubatches) {
    return false;
  }
  const int32_t size = ctx.ubatch_sizes[ubatch_index];
  const int32_t start = ctx.slot_offsets[ubatch_index];
  if (ctx.kv_size <= 0 || size <= 0 || start < 0 ||
      start + size > ctx.slot_index_count) {
    return false;
  }
  if (!valid_stream_id(ctx, ctx.ubatch_stream_ids[ubatch_index])) {
    return false;
  }
  if (!valid_seq_id(ctx.ubatch_seq_ids[ubatch_index])) {
    return false;
  }
  if (request->positions != nullptr && request->positions_count < size) {
    return false;
  }
  return true;
};

inline constexpr auto invalid_apply_step_request = [](
    const event::apply_step & ev,
    const action::context & ctx) noexcept {
  return !valid_apply_step_request(ev, ctx);
};

inline constexpr auto valid_rollback_request = [](
    const event::validate_rollback & ev,
    const action::context & ctx) noexcept {
  const event::rollback * request = ev.request;
  if (request == nullptr) {
    return false;
  }
  const int32_t from_index = request->from_ubatch_index;
  if (from_index < 0 ||
      from_index > ctx.applied_ubatches ||
      from_index > ctx.planned_ubatch_count) {
    return false;
  }
  return true;
};

inline constexpr auto invalid_rollback_request = [](
    const event::validate_rollback & ev,
    const action::context & ctx) noexcept {
  return !valid_rollback_request(ev, ctx);
};

inline constexpr auto valid_rollback_step_request = [](
    const event::rollback_step & ev,
    const action::context & ctx) noexcept {
  const event::rollback * request = ev.request;
  if (request == nullptr) {
    return false;
  }
  const int32_t from_index = request->from_ubatch_index;
  if (from_index < 0 ||
      from_index > ctx.applied_ubatches ||
      from_index > ctx.planned_ubatch_count) {
    return false;
  }
  for (int32_t i = from_index; i < ctx.applied_ubatches; ++i) {
    const int32_t size = ctx.ubatch_sizes[i];
    const int32_t start = ctx.slot_offsets[i];
    if (size <= 0 || start < 0 || start + size > ctx.slot_index_count) {
      return false;
    }
    if (!valid_stream_id(ctx, ctx.ubatch_stream_ids[i])) {
      return false;
    }
    if (!valid_seq_id(ctx.ubatch_seq_ids[i])) {
      return false;
    }
  }
  return true;
};

inline constexpr auto invalid_rollback_step_request = [](
    const event::rollback_step & ev,
    const action::context & ctx) noexcept {
  return !valid_rollback_step_request(ev, ctx);
};

inline constexpr auto valid_seq_remove_request = [](
    const event::validate_seq_remove & ev,
    const action::context & ctx) noexcept {
  const event::seq_remove * request = ev.request;
  if (request == nullptr) {
    return false;
  }
  if (request->seq_id != -1) {
    if (!valid_seq_id(request->seq_id)) {
      return false;
    }
    const int32_t stream_id = ctx.seq_to_stream[request->seq_id];
    if (!valid_stream_id(ctx, stream_id)) {
      return false;
    }
  }
  if (!valid_pos_range(request->pos_start, request->pos_end)) {
    return false;
  }
  return true;
};

inline constexpr auto invalid_seq_remove_request = [](
    const event::validate_seq_remove & ev,
    const action::context & ctx) noexcept {
  return !valid_seq_remove_request(ev, ctx);
};

inline constexpr auto valid_seq_remove_step_request = [](
    const event::seq_remove_step & ev,
    const action::context & ctx) noexcept {
  const event::seq_remove * request = ev.request;
  if (request == nullptr) {
    return false;
  }
  if (request->seq_id != -1) {
    if (!valid_seq_id(request->seq_id)) {
      return false;
    }
    const int32_t stream_id = ctx.seq_to_stream[request->seq_id];
    if (!valid_stream_id(ctx, stream_id)) {
      return false;
    }
  }
  if (!valid_pos_range(request->pos_start, request->pos_end)) {
    return false;
  }
  return true;
};

inline constexpr auto invalid_seq_remove_step_request = [](
    const event::seq_remove_step & ev,
    const action::context & ctx) noexcept {
  return !valid_seq_remove_step_request(ev, ctx);
};

inline constexpr auto valid_seq_copy_request = [](
    const event::validate_seq_copy & ev,
    const action::context & ctx) noexcept {
  const event::seq_copy * request = ev.request;
  if (request == nullptr) {
    return false;
  }
  if (!valid_seq_id(request->seq_id_src) || !valid_seq_id(request->seq_id_dst)) {
    return false;
  }
  const int32_t src_stream = ctx.seq_to_stream[request->seq_id_src];
  const int32_t dst_stream = ctx.seq_to_stream[request->seq_id_dst];
  if (!valid_stream_id(ctx, src_stream) || !valid_stream_id(ctx, dst_stream)) {
    return false;
  }
  if (!valid_pos_range(request->pos_start, request->pos_end)) {
    return false;
  }
  if (src_stream == dst_stream) {
    return true;
  }
  if (!is_full_copy_range(request->pos_start, request->pos_end, ctx.kv_size)) {
    return false;
  }
  bool has_pair = false;
  for (int32_t i = 0; i < ctx.pending_copy_count; ++i) {
    if (ctx.pending_copy_src[i] == src_stream && ctx.pending_copy_dst[i] == dst_stream) {
      has_pair = true;
      break;
    }
  }
  if (!has_pair && ctx.pending_copy_count >= action::MAX_STREAM_COPY) {
    return false;
  }
  return true;
};

inline constexpr auto invalid_seq_copy_request = [](
    const event::validate_seq_copy & ev,
    const action::context & ctx) noexcept {
  return !valid_seq_copy_request(ev, ctx);
};

inline constexpr auto valid_seq_copy_step_request = [](
    const event::seq_copy_step & ev,
    const action::context & ctx) noexcept {
  const event::seq_copy * request = ev.request;
  if (request == nullptr) {
    return false;
  }
  if (!valid_seq_id(request->seq_id_src) || !valid_seq_id(request->seq_id_dst)) {
    return false;
  }
  const int32_t src_stream = ctx.seq_to_stream[request->seq_id_src];
  const int32_t dst_stream = ctx.seq_to_stream[request->seq_id_dst];
  if (!valid_stream_id(ctx, src_stream) || !valid_stream_id(ctx, dst_stream)) {
    return false;
  }
  if (!valid_pos_range(request->pos_start, request->pos_end)) {
    return false;
  }
  if (src_stream == dst_stream) {
    return true;
  }
  if (!is_full_copy_range(request->pos_start, request->pos_end, ctx.kv_size)) {
    return false;
  }
  bool has_pair = false;
  for (int32_t i = 0; i < ctx.pending_copy_count; ++i) {
    if (ctx.pending_copy_src[i] == src_stream && ctx.pending_copy_dst[i] == dst_stream) {
      has_pair = true;
      break;
    }
  }
  if (!has_pair && ctx.pending_copy_count >= action::MAX_STREAM_COPY) {
    return false;
  }
  return true;
};

inline constexpr auto invalid_seq_copy_step_request = [](
    const event::seq_copy_step & ev,
    const action::context & ctx) noexcept {
  return !valid_seq_copy_step_request(ev, ctx);
};

inline constexpr auto valid_seq_keep_request = [](
    const event::validate_seq_keep & ev,
    const action::context & ctx) noexcept {
  const event::seq_keep * request = ev.request;
  if (request == nullptr) {
    return false;
  }
  if (!valid_seq_id(request->seq_id)) {
    return false;
  }
  const int32_t stream_id = ctx.seq_to_stream[request->seq_id];
  if (!valid_stream_id(ctx, stream_id)) {
    return false;
  }
  return true;
};

inline constexpr auto invalid_seq_keep_request = [](
    const event::validate_seq_keep & ev,
    const action::context & ctx) noexcept {
  return !valid_seq_keep_request(ev, ctx);
};

inline constexpr auto valid_seq_keep_step_request = [](
    const event::seq_keep_step & ev,
    const action::context & ctx) noexcept {
  const event::seq_keep * request = ev.request;
  if (request == nullptr) {
    return false;
  }
  if (!valid_seq_id(request->seq_id)) {
    return false;
  }
  const int32_t stream_id = ctx.seq_to_stream[request->seq_id];
  if (!valid_stream_id(ctx, stream_id)) {
    return false;
  }
  return true;
};

inline constexpr auto invalid_seq_keep_step_request = [](
    const event::seq_keep_step & ev,
    const action::context & ctx) noexcept {
  return !valid_seq_keep_step_request(ev, ctx);
};

inline constexpr auto valid_seq_add_request = [](
    const event::validate_seq_add & ev,
    const action::context & ctx) noexcept {
  const event::seq_add * request = ev.request;
  if (request == nullptr) {
    return false;
  }
  if (!valid_seq_id(request->seq_id)) {
    return false;
  }
  const int32_t stream_id = ctx.seq_to_stream[request->seq_id];
  if (!valid_stream_id(ctx, stream_id)) {
    return false;
  }
  if (!valid_pos_range(request->pos_start, request->pos_end)) {
    return false;
  }
  return true;
};

inline constexpr auto invalid_seq_add_request = [](
    const event::validate_seq_add & ev,
    const action::context & ctx) noexcept {
  return !valid_seq_add_request(ev, ctx);
};

inline constexpr auto valid_seq_add_step_request = [](
    const event::seq_add_step & ev,
    const action::context & ctx) noexcept {
  const event::seq_add * request = ev.request;
  if (request == nullptr) {
    return false;
  }
  if (!valid_seq_id(request->seq_id)) {
    return false;
  }
  const int32_t stream_id = ctx.seq_to_stream[request->seq_id];
  if (!valid_stream_id(ctx, stream_id)) {
    return false;
  }
  if (!valid_pos_range(request->pos_start, request->pos_end)) {
    return false;
  }
  return true;
};

inline constexpr auto invalid_seq_add_step_request = [](
    const event::seq_add_step & ev,
    const action::context & ctx) noexcept {
  return !valid_seq_add_step_request(ev, ctx);
};

inline constexpr auto valid_seq_div_request = [](
    const event::validate_seq_div & ev,
    const action::context & ctx) noexcept {
  const event::seq_div * request = ev.request;
  if (request == nullptr) {
    return false;
  }
  if (!valid_seq_id(request->seq_id)) {
    return false;
  }
  const int32_t stream_id = ctx.seq_to_stream[request->seq_id];
  if (!valid_stream_id(ctx, stream_id)) {
    return false;
  }
  if (request->divisor <= 0) {
    return false;
  }
  if (!valid_pos_range(request->pos_start, request->pos_end)) {
    return false;
  }
  return true;
};

inline constexpr auto invalid_seq_div_request = [](
    const event::validate_seq_div & ev,
    const action::context & ctx) noexcept {
  return !valid_seq_div_request(ev, ctx);
};

inline constexpr auto valid_seq_div_step_request = [](
    const event::seq_div_step & ev,
    const action::context & ctx) noexcept {
  const event::seq_div * request = ev.request;
  if (request == nullptr) {
    return false;
  }
  if (!valid_seq_id(request->seq_id)) {
    return false;
  }
  const int32_t stream_id = ctx.seq_to_stream[request->seq_id];
  if (!valid_stream_id(ctx, stream_id)) {
    return false;
  }
  if (request->divisor <= 0) {
    return false;
  }
  if (!valid_pos_range(request->pos_start, request->pos_end)) {
    return false;
  }
  return true;
};

inline constexpr auto invalid_seq_div_step_request = [](
    const event::seq_div_step & ev,
    const action::context & ctx) noexcept {
  return !valid_seq_div_step_request(ev, ctx);
};

inline constexpr auto valid_updates_request = [](
    const event::validate_updates & ev,
    const action::context & ctx) noexcept {
  const event::apply_updates * request = ev.request;
  if (request == nullptr) {
    return false;
  }
  if (ctx.pending_copy_count > 0 && request->stream_copy == nullptr) {
    return false;
  }
  for (int32_t s = 0; s < ctx.n_stream; ++s) {
    if (ctx.streams[s].has_shift && request->apply_shift == nullptr) {
      return false;
    }
  }
  return true;
};

inline constexpr auto valid_updates_step_request = [](
    const event::apply_updates_step & ev,
    const action::context & ctx) noexcept {
  const event::apply_updates * request = ev.request;
  if (request == nullptr) {
    return false;
  }
  if (ctx.pending_copy_count > 0 && request->stream_copy == nullptr) {
    return false;
  }
  for (int32_t s = 0; s < ctx.n_stream; ++s) {
    if (ctx.streams[s].has_shift && request->apply_shift == nullptr) {
      return false;
    }
  }
  return true;
};

inline constexpr auto invalid_updates_request = [](
    const event::validate_updates & ev,
    const action::context & ctx) noexcept {
  return !valid_updates_request(ev, ctx);
};

inline constexpr auto invalid_updates_step_request = [](
    const event::apply_updates_step & ev,
    const action::context & ctx) noexcept {
  return !valid_updates_step_request(ev, ctx);
};

struct valid_prepare_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::validate_prepare validate{.request = &ctx.prepare_request};
    return valid_prepare_request(validate, ctx);
  }
};

struct invalid_prepare_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_prepare_context{}(ctx);
  }
};

struct valid_prepare_slots_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::prepare_slots slots{};
    return valid_prepare_slots_request(slots, ctx);
  }
};

struct invalid_prepare_slots_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_prepare_slots_context{}(ctx);
  }
};

struct valid_apply_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::validate_apply validate{.request = &ctx.apply_request};
    return valid_apply_request(validate, ctx);
  }
};

struct invalid_apply_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_apply_context{}(ctx);
  }
};

struct valid_apply_step_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::apply_step step{.request = &ctx.apply_request};
    return valid_apply_step_request(step, ctx);
  }
};

struct invalid_apply_step_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_apply_step_context{}(ctx);
  }
};

struct valid_rollback_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::validate_rollback validate{.request = &ctx.rollback_request};
    return valid_rollback_request(validate, ctx);
  }
};

struct invalid_rollback_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_rollback_context{}(ctx);
  }
};

struct valid_rollback_step_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::rollback_step step{.request = &ctx.rollback_request};
    return valid_rollback_step_request(step, ctx);
  }
};

struct invalid_rollback_step_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_rollback_step_context{}(ctx);
  }
};

struct valid_seq_remove_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::validate_seq_remove validate{.request = &ctx.seq_remove_request};
    return valid_seq_remove_request(validate, ctx);
  }
};

struct invalid_seq_remove_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_seq_remove_context{}(ctx);
  }
};

struct valid_seq_remove_step_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::seq_remove_step step{.request = &ctx.seq_remove_request};
    return valid_seq_remove_step_request(step, ctx);
  }
};

struct invalid_seq_remove_step_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_seq_remove_step_context{}(ctx);
  }
};

struct valid_seq_copy_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::validate_seq_copy validate{.request = &ctx.seq_copy_request};
    return valid_seq_copy_request(validate, ctx);
  }
};

struct invalid_seq_copy_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_seq_copy_context{}(ctx);
  }
};

struct valid_seq_copy_step_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::seq_copy_step step{.request = &ctx.seq_copy_request};
    return valid_seq_copy_step_request(step, ctx);
  }
};

struct invalid_seq_copy_step_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_seq_copy_step_context{}(ctx);
  }
};

struct valid_seq_keep_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::validate_seq_keep validate{.request = &ctx.seq_keep_request};
    return valid_seq_keep_request(validate, ctx);
  }
};

struct invalid_seq_keep_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_seq_keep_context{}(ctx);
  }
};

struct valid_seq_keep_step_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::seq_keep_step step{.request = &ctx.seq_keep_request};
    return valid_seq_keep_step_request(step, ctx);
  }
};

struct invalid_seq_keep_step_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_seq_keep_step_context{}(ctx);
  }
};

struct valid_seq_add_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::validate_seq_add validate{.request = &ctx.seq_add_request};
    return valid_seq_add_request(validate, ctx);
  }
};

struct invalid_seq_add_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_seq_add_context{}(ctx);
  }
};

struct valid_seq_add_step_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::seq_add_step step{.request = &ctx.seq_add_request};
    return valid_seq_add_step_request(step, ctx);
  }
};

struct invalid_seq_add_step_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_seq_add_step_context{}(ctx);
  }
};

struct valid_seq_div_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::validate_seq_div validate{.request = &ctx.seq_div_request};
    return valid_seq_div_request(validate, ctx);
  }
};

struct invalid_seq_div_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_seq_div_context{}(ctx);
  }
};

struct valid_seq_div_step_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::seq_div_step step{.request = &ctx.seq_div_request};
    return valid_seq_div_step_request(step, ctx);
  }
};

struct invalid_seq_div_step_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_seq_div_step_context{}(ctx);
  }
};

struct valid_updates_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::validate_updates validate{.request = &ctx.updates_request};
    return valid_updates_request(validate, ctx);
  }
};

struct invalid_updates_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_updates_context{}(ctx);
  }
};

struct valid_updates_step_context {
  bool operator()(const action::context & ctx) const noexcept {
    event::apply_updates_step step{.request = &ctx.updates_request};
    return valid_updates_step_request(step, ctx);
  }
};

struct invalid_updates_step_context {
  bool operator()(const action::context & ctx) const noexcept {
    return !valid_updates_step_context{}(ctx);
  }
};

}  // namespace emel::kv::cache::guard
