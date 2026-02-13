#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include "emel/buffer_planner/events.hpp"
#include "emel/emel.h"

namespace emel::buffer_planner::action {
struct context;
}

namespace emel::buffer_planner {

struct strategy {
  void (*seed_leafs)(action::context &) noexcept = nullptr;
  void (*count_references)(action::context &) noexcept = nullptr;
  void (*alloc_explicit_inputs)(action::context &) noexcept = nullptr;
  void (*plan_nodes)(action::context &) noexcept = nullptr;
  void (*release_expired)(action::context &) noexcept = nullptr;
  void (*finalize)(action::context &) noexcept = nullptr;
};

}  // namespace emel::buffer_planner

namespace emel::buffer_planner::action {

inline constexpr int32_t k_max_buffers = 16;
inline constexpr int32_t k_max_tensors = 2048;
inline constexpr int32_t k_max_free_blocks = 256;

struct free_block {
  int32_t offset = 0;
  int32_t size = 0;
};

struct buffer_layout {
  std::array<free_block, k_max_free_blocks> free_blocks = {};
  int32_t free_block_count = 0;
  int32_t high_watermark = 0;
};

struct tensor_record {
  int32_t tensor_id = -1;
  int32_t alloc_size = 0;
  int32_t buffer_id = 0;
  int32_t alloc_offset = -1;
  int32_t alloc_reserved = 0;
  int32_t n_children = 0;
  int32_t n_views = 0;
  int32_t view_src_id = -1;
  bool is_view = false;
  bool is_input = false;
  bool is_output = false;
  bool allocatable = false;
  bool allocated = false;
};

struct context {
  event::graph_view graph = {};
  const int32_t * node_buffer_ids = nullptr;
  const int32_t * leaf_buffer_ids = nullptr;
  int32_t buffer_count = 0;
  bool size_only = false;
  int32_t * sizes_out = nullptr;
  int32_t sizes_out_count = 0;
  int32_t * error_out = nullptr;
  const emel::buffer_planner::strategy * strategy = nullptr;

  std::array<int32_t, k_max_buffers> current_bytes_by_buffer = {};
  std::array<int32_t, k_max_buffers> bytes_by_buffer = {};
  std::array<buffer_layout, k_max_buffers> buffer_layouts = {};
  std::array<tensor_record, k_max_tensors> tensors = {};
  int32_t tensor_count = 0;
  int32_t total_bytes = 0;
  int32_t planned_nodes = 0;
  int32_t planned_leafs = 0;
  int32_t reference_edges = 0;
  int32_t pending_error = 0;
  int32_t last_error = 0;
};

namespace detail {

inline int32_t normalize_error(const int32_t err) noexcept {
  return err == 0 ? EMEL_ERR_BACKEND : err;
}

inline int32_t sat_add(const int32_t lhs, const int32_t rhs) noexcept {
  const int64_t sum = static_cast<int64_t>(lhs) + static_cast<int64_t>(rhs);
  if (sum > std::numeric_limits<int32_t>::max()) {
    return std::numeric_limits<int32_t>::max();
  }
  if (sum < std::numeric_limits<int32_t>::min()) {
    return std::numeric_limits<int32_t>::min();
  }
  return static_cast<int32_t>(sum);
}

inline int32_t sat_sub_floor_zero(const int32_t lhs, const int32_t rhs) noexcept {
  if (rhs <= 0) {
    return lhs;
  }
  if (lhs <= rhs) {
    return 0;
  }
  return lhs - rhs;
}

inline int32_t align_up(const int32_t value) noexcept {
  if (value <= 0) {
    return 0;
  }
  const int64_t aligned = (static_cast<int64_t>(value) + 15LL) & ~15LL;
  if (aligned > std::numeric_limits<int32_t>::max()) {
    return std::numeric_limits<int32_t>::max();
  }
  return static_cast<int32_t>(aligned);
}

inline int32_t buffer_id_for(
    const int32_t * ids, const int32_t index, const int32_t fallback = 0) noexcept {
  return ids == nullptr ? fallback : ids[index];
}

inline bool valid_buffer_id(const int32_t buffer_id, const int32_t buffer_count) noexcept {
  return buffer_id >= 0 && buffer_id < buffer_count;
}

inline void reset_layouts(context & ctx) noexcept {
  for (int32_t i = 0; i < k_max_buffers; ++i) {
    ctx.buffer_layouts[i] = {};
  }
}

inline void remove_free_block(buffer_layout & layout, const int32_t idx) noexcept {
  if (idx < 0 || idx >= layout.free_block_count) {
    return;
  }
  for (int32_t i = idx; i + 1 < layout.free_block_count; ++i) {
    layout.free_blocks[i] = layout.free_blocks[i + 1];
  }
  layout.free_block_count -= 1;
}

inline bool insert_free_block(
    buffer_layout & layout, const int32_t offset, const int32_t size) noexcept {
  if (size <= 0 || offset < 0) {
    return true;
  }
  if (layout.free_block_count >= k_max_free_blocks) {
    return false;
  }

  int32_t pos = 0;
  while (pos < layout.free_block_count && layout.free_blocks[pos].offset < offset) {
    pos += 1;
  }
  for (int32_t i = layout.free_block_count; i > pos; --i) {
    layout.free_blocks[i] = layout.free_blocks[i - 1];
  }
  layout.free_blocks[pos] = free_block{
    .offset = offset,
    .size = size,
  };
  layout.free_block_count += 1;

  if (pos > 0) {
    auto & prev = layout.free_blocks[pos - 1];
    auto & cur = layout.free_blocks[pos];
    if (sat_add(prev.offset, prev.size) >= cur.offset) {
      const int32_t prev_end = sat_add(prev.offset, prev.size);
      const int32_t cur_end = sat_add(cur.offset, cur.size);
      prev.size = (cur_end > prev_end ? cur_end : prev_end) - prev.offset;
      remove_free_block(layout, pos);
      pos -= 1;
    }
  }

  if (pos + 1 < layout.free_block_count) {
    auto & cur = layout.free_blocks[pos];
    auto & next = layout.free_blocks[pos + 1];
    if (sat_add(cur.offset, cur.size) >= next.offset) {
      const int32_t cur_end = sat_add(cur.offset, cur.size);
      const int32_t next_end = sat_add(next.offset, next.size);
      cur.size = (next_end > cur_end ? next_end : cur_end) - cur.offset;
      remove_free_block(layout, pos + 1);
    }
  }

  return true;
}

inline bool alloc_bytes_from_layout(
    context & ctx, const int32_t buffer_id, const int32_t size, int32_t & out_offset) noexcept {
  if (!valid_buffer_id(buffer_id, ctx.buffer_count)) {
    return false;
  }
  const int32_t alloc_size = align_up(size);
  auto & layout = ctx.buffer_layouts[buffer_id];

  int32_t best_idx = -1;
  int32_t best_waste = std::numeric_limits<int32_t>::max();
  for (int32_t i = 0; i < layout.free_block_count; ++i) {
    const auto & block = layout.free_blocks[i];
    if (block.size < alloc_size) {
      continue;
    }
    const int32_t waste = block.size - alloc_size;
    if (waste < best_waste) {
      best_waste = waste;
      best_idx = i;
    }
  }

  if (best_idx >= 0) {
    auto & block = layout.free_blocks[best_idx];
    out_offset = block.offset;
    block.offset = sat_add(block.offset, alloc_size);
    block.size = sat_sub_floor_zero(block.size, alloc_size);
    if (block.size == 0) {
      remove_free_block(layout, best_idx);
    }
    return true;
  }

  const int32_t offset = layout.high_watermark;
  const int32_t end = sat_add(offset, alloc_size);
  if (end < offset) {
    return false;
  }
  out_offset = offset;
  layout.high_watermark = end;
  if (layout.high_watermark > ctx.bytes_by_buffer[buffer_id]) {
    ctx.bytes_by_buffer[buffer_id] = layout.high_watermark;
  }
  return true;
}

inline bool free_bytes_to_layout(
    context & ctx, const int32_t buffer_id, const int32_t offset, const int32_t size) noexcept {
  if (!valid_buffer_id(buffer_id, ctx.buffer_count)) {
    return false;
  }
  return insert_free_block(ctx.buffer_layouts[buffer_id], offset, align_up(size));
}

inline int32_t find_record_index(const context & ctx, const int32_t tensor_id) noexcept {
  for (int32_t i = 0; i < ctx.tensor_count; ++i) {
    if (ctx.tensors[i].tensor_id == tensor_id) {
      return i;
    }
  }
  return -1;
}

inline tensor_record * find_record(context & ctx, const int32_t tensor_id) noexcept {
  const int32_t idx = find_record_index(ctx, tensor_id);
  return idx < 0 ? nullptr : &ctx.tensors[idx];
}

inline const tensor_record * find_record(const context & ctx, const int32_t tensor_id) noexcept {
  const int32_t idx = find_record_index(ctx, tensor_id);
  return idx < 0 ? nullptr : &ctx.tensors[idx];
}

inline bool can_allocate_tensor(
    const emel::buffer_allocator::event::tensor_desc & tensor) noexcept {
  return !tensor.has_external_data && !tensor.is_view && tensor.alloc_size > 0;
}

inline bool register_tensor(
    context & ctx, const emel::buffer_allocator::event::tensor_desc & tensor) noexcept {
  if (tensor.tensor_id < 0 || tensor.alloc_size < 0) {
    return false;
  }
  if (find_record(ctx, tensor.tensor_id) != nullptr) {
    return false;
  }
  if (ctx.tensor_count >= k_max_tensors) {
    return false;
  }

  auto & rec = ctx.tensors[ctx.tensor_count++];
  rec = tensor_record{};
  rec.tensor_id = tensor.tensor_id;
  rec.alloc_size = align_up(tensor.alloc_size);
  rec.view_src_id = tensor.view_src_id;
  rec.is_view = tensor.is_view;
  rec.is_input = tensor.is_input;
  rec.is_output = tensor.is_output;
  rec.allocatable = can_allocate_tensor(tensor);
  return true;
}

inline bool register_graph_tensors(context & ctx) noexcept {
  for (int32_t i = 0; i < ctx.graph.n_leafs; ++i) {
    if (!register_tensor(ctx, ctx.graph.leafs[i])) {
      return false;
    }
  }
  for (int32_t i = 0; i < ctx.graph.n_nodes; ++i) {
    if (!register_tensor(ctx, ctx.graph.nodes[i])) {
      return false;
    }
  }
  return true;
}

inline bool allocate_record(context & ctx, tensor_record & rec, const int32_t buffer_id) noexcept {
  if (!rec.allocatable || rec.allocated) {
    return true;
  }
  if (!valid_buffer_id(buffer_id, ctx.buffer_count)) {
    return false;
  }

  int32_t offset = -1;
  if (!alloc_bytes_from_layout(ctx, buffer_id, rec.alloc_size, offset)) {
    return false;
  }

  rec.buffer_id = buffer_id;
  rec.alloc_offset = offset;
  rec.alloc_reserved = rec.alloc_size;
  rec.allocated = true;
  ctx.current_bytes_by_buffer[buffer_id] =
    sat_add(ctx.current_bytes_by_buffer[buffer_id], rec.alloc_reserved);
  return true;
}

inline bool free_record(context & ctx, tensor_record & rec) noexcept {
  if (!rec.allocatable || !rec.allocated) {
    return true;
  }
  if (!valid_buffer_id(rec.buffer_id, ctx.buffer_count)) {
    rec.allocated = false;
    return false;
  }

  if (!free_bytes_to_layout(ctx, rec.buffer_id, rec.alloc_offset, rec.alloc_reserved)) {
    return false;
  }
  ctx.current_bytes_by_buffer[rec.buffer_id] =
    sat_sub_floor_zero(ctx.current_bytes_by_buffer[rec.buffer_id], rec.alloc_reserved);
  rec.allocated = false;
  rec.alloc_offset = -1;
  rec.alloc_reserved = 0;
  return true;
}

inline bool valid_plan_event(const event::plan & ev) noexcept {
  if (ev.buffer_count <= 0 || ev.buffer_count > k_max_buffers) {
    return false;
  }
  if (ev.graph.n_nodes < 0 || ev.graph.n_leafs < 0) {
    return false;
  }
  if ((ev.graph.n_nodes > 0 && ev.graph.nodes == nullptr) ||
      (ev.graph.n_leafs > 0 && ev.graph.leafs == nullptr)) {
    return false;
  }
  const int64_t total_tensors = static_cast<int64_t>(ev.graph.n_nodes) + ev.graph.n_leafs;
  if (total_tensors > k_max_tensors) {
    return false;
  }
  if (ev.sizes_out != nullptr && ev.sizes_out_count < ev.buffer_count) {
    return false;
  }
  return true;
}

inline bool valid_strategy(const emel::buffer_planner::strategy * strategy) noexcept {
  if (strategy == nullptr) {
    return true;
  }
  return strategy->seed_leafs != nullptr && strategy->count_references != nullptr &&
         strategy->alloc_explicit_inputs != nullptr && strategy->plan_nodes != nullptr &&
         strategy->release_expired != nullptr && strategy->finalize != nullptr;
}

inline void default_seed_leafs(context & ctx) noexcept {
  if (ctx.pending_error != EMEL_OK) {
    return;
  }
  for (int32_t i = 0; i < ctx.graph.n_leafs; ++i) {
    const auto & leaf = ctx.graph.leafs[i];
    auto * rec = detail::find_record(ctx, leaf.tensor_id);
    if (rec == nullptr) {
      ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    const int32_t buffer_id = detail::buffer_id_for(ctx.leaf_buffer_ids, i);
    const bool was_allocated = rec->allocated;
    if (!detail::allocate_record(ctx, *rec, buffer_id)) {
      ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    if (!was_allocated && rec->allocated) {
      ctx.planned_leafs = detail::sat_add(ctx.planned_leafs, 1);
    }
  }
}

inline void default_count_references(context & ctx) noexcept {
  if (ctx.pending_error != EMEL_OK) {
    return;
  }
  for (int32_t i = 0; i < ctx.graph.n_nodes; ++i) {
    const auto & node = ctx.graph.nodes[i];
    auto * node_rec = detail::find_record(ctx, node.tensor_id);
    if (node_rec == nullptr) {
      ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    if (node.is_view) {
      if (node.view_src_id < 0) {
        ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
        return;
      }
      auto * view_src = detail::find_record(ctx, node.view_src_id);
      if (view_src == nullptr) {
        ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
        return;
      }
      view_src->n_views = detail::sat_add(view_src->n_views, 1);
    }
    for (int32_t j = 0; j < emel::buffer_allocator::event::k_max_sources; ++j) {
      const int32_t src_id = node.src_ids[j];
      if (src_id < 0) {
        continue;
      }
      auto * src = detail::find_record(ctx, src_id);
      if (src == nullptr) {
        ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
        return;
      }
      src->n_children = detail::sat_add(src->n_children, 1);
      ctx.reference_edges = detail::sat_add(ctx.reference_edges, 1);
    }
  }
}

inline void default_alloc_explicit_inputs(context & ctx) noexcept {
  if (ctx.pending_error != EMEL_OK) {
    return;
  }
  for (int32_t i = 0; i < ctx.graph.n_nodes; ++i) {
    const auto & node = ctx.graph.nodes[i];
    const int32_t buffer_id = detail::buffer_id_for(ctx.node_buffer_ids, i);
    auto * node_rec = detail::find_record(ctx, node.tensor_id);
    if (node_rec == nullptr) {
      ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }

    if (node.is_input) {
      const bool was_allocated = node_rec->allocated;
      if (!detail::allocate_record(ctx, *node_rec, buffer_id)) {
        ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
        return;
      }
      if (!was_allocated && node_rec->allocated) {
        ctx.planned_nodes = detail::sat_add(ctx.planned_nodes, 1);
      }
    }

    for (int32_t j = 0; j < emel::buffer_allocator::event::k_max_sources; ++j) {
      const int32_t src_id = node.src_ids[j];
      if (src_id < 0) {
        continue;
      }
      auto * src = detail::find_record(ctx, src_id);
      if (src == nullptr) {
        ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
        return;
      }
      if (!src->is_input) {
        continue;
      }
      const bool was_allocated = src->allocated;
      if (!detail::allocate_record(ctx, *src, buffer_id)) {
        ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
        return;
      }
      if (!was_allocated && src->allocated) {
        ctx.planned_nodes = detail::sat_add(ctx.planned_nodes, 1);
      }
    }
  }
}

inline void default_plan_nodes(context & ctx) noexcept {
  if (ctx.pending_error != EMEL_OK) {
    return;
  }
  for (int32_t i = 0; i < ctx.graph.n_nodes; ++i) {
    const auto & node = ctx.graph.nodes[i];
    const int32_t buffer_id = detail::buffer_id_for(ctx.node_buffer_ids, i);
    auto * node_rec = detail::find_record(ctx, node.tensor_id);
    if (node_rec == nullptr) {
      ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }

    for (int32_t j = 0; j < emel::buffer_allocator::event::k_max_sources; ++j) {
      const int32_t src_id = node.src_ids[j];
      if (src_id < 0) {
        continue;
      }
      auto * parent = detail::find_record(ctx, src_id);
      if (parent == nullptr) {
        ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
        return;
      }
      if (!detail::allocate_record(ctx, *parent, buffer_id)) {
        ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
        return;
      }
    }

    if (!node.is_input && node_rec->allocatable && !node_rec->allocated) {
      bool reused_parent = false;
      for (int32_t j = 0; j < emel::buffer_allocator::event::k_max_sources; ++j) {
        const int32_t src_id = node.src_ids[j];
        if (src_id < 0) {
          continue;
        }
        auto * parent = detail::find_record(ctx, src_id);
        if (parent == nullptr) {
          ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
          return;
        }
        tensor_record * reuse_owner = parent;
        bool reuse_from_view_src = false;
        if (parent->is_view) {
          if (parent->view_src_id < 0) {
            continue;
          }
          auto * view_src = detail::find_record(ctx, parent->view_src_id);
          if (view_src == nullptr) {
            ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
            return;
          }
          if (!view_src->allocatable || !view_src->allocated || view_src->is_output) {
            continue;
          }
          if (view_src->n_views != 1 || view_src->n_children != 0) {
            continue;
          }
          reuse_owner = view_src;
          reuse_from_view_src = true;
        }

        if (!reuse_owner->allocatable || !reuse_owner->allocated || reuse_owner->is_output) {
          continue;
        }
        if (!detail::valid_buffer_id(reuse_owner->buffer_id, ctx.buffer_count) ||
            reuse_owner->buffer_id != buffer_id) {
          continue;
        }
        if (reuse_from_view_src) {
          if (reuse_owner->n_views != 1 || reuse_owner->n_children != 0) {
            continue;
          }
        } else {
          if (reuse_owner->n_children != 1 || reuse_owner->n_views != 0) {
            continue;
          }
        }
        if (reuse_owner->alloc_size < node_rec->alloc_size) {
          continue;
        }

        node_rec->buffer_id = reuse_owner->buffer_id;
        node_rec->alloc_offset = reuse_owner->alloc_offset;
        node_rec->alloc_reserved = node_rec->alloc_size;
        node_rec->allocated = true;
        const int32_t extra = reuse_owner->alloc_reserved - node_rec->alloc_reserved;
        if (extra > 0) {
          if (!detail::free_bytes_to_layout(
                ctx,
                reuse_owner->buffer_id,
                detail::sat_add(reuse_owner->alloc_offset, node_rec->alloc_reserved),
                extra)) {
            ctx.pending_error = EMEL_ERR_BACKEND;
            return;
          }
          ctx.current_bytes_by_buffer[reuse_owner->buffer_id] =
            detail::sat_sub_floor_zero(ctx.current_bytes_by_buffer[reuse_owner->buffer_id], extra);
        }
        reuse_owner->allocated = false;
        reuse_owner->alloc_offset = -1;
        reuse_owner->alloc_reserved = 0;
        reused_parent = true;
        break;
      }

      if (!reused_parent && !detail::allocate_record(ctx, *node_rec, buffer_id)) {
        ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
        return;
      }

      if (node_rec->allocated) {
        ctx.planned_nodes = detail::sat_add(ctx.planned_nodes, 1);
      }
    }

    for (int32_t j = 0; j < emel::buffer_allocator::event::k_max_sources; ++j) {
      const int32_t src_id = node.src_ids[j];
      if (src_id < 0) {
        continue;
      }
      auto * parent = detail::find_record(ctx, src_id);
      if (parent == nullptr) {
        ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
        return;
      }
      parent->n_children = parent->n_children <= 0 ? 0 : parent->n_children - 1;

      if (parent->n_children == 0 && parent->n_views == 0) {
        if (parent->is_view) {
          if (parent->view_src_id < 0) {
            ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
            return;
          }
          auto * view_src = detail::find_record(ctx, parent->view_src_id);
          if (view_src == nullptr) {
            ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
            return;
          }
          view_src->n_views = view_src->n_views <= 0 ? 0 : view_src->n_views - 1;
          if (view_src->n_views == 0 && view_src->n_children == 0 && view_src->allocated &&
              !view_src->is_output) {
            if (!detail::free_record(ctx, *view_src)) {
              ctx.pending_error = EMEL_ERR_BACKEND;
              return;
            }
          }
        } else if (parent->allocated && !parent->is_output) {
          if (!detail::free_record(ctx, *parent)) {
            ctx.pending_error = EMEL_ERR_BACKEND;
            return;
          }
        }
      }
    }
  }
}

inline void default_release_expired(context &) noexcept {}

inline void default_finalize(context & ctx) noexcept {
  if (ctx.pending_error != EMEL_OK) {
    return;
  }
  ctx.total_bytes = 0;
  for (int32_t i = 0; i < ctx.buffer_count; ++i) {
    ctx.total_bytes = detail::sat_add(ctx.total_bytes, ctx.bytes_by_buffer[i]);
    if (ctx.sizes_out != nullptr && i < ctx.sizes_out_count) {
      ctx.sizes_out[i] = ctx.bytes_by_buffer[i];
    }
  }
}

inline constexpr emel::buffer_planner::strategy make_gallocr_strategy() noexcept {
  return emel::buffer_planner::strategy{
    .seed_leafs = default_seed_leafs,
    .count_references = default_count_references,
    .alloc_explicit_inputs = default_alloc_explicit_inputs,
    .plan_nodes = default_plan_nodes,
    .release_expired = default_release_expired,
    .finalize = default_finalize,
  };
}

}  // namespace detail

}  // namespace emel::buffer_planner::action

namespace emel::buffer_planner::default_strategies {

inline constexpr strategy gallocr_parity = action::detail::make_gallocr_strategy();
inline constexpr strategy reserve_n_size = gallocr_parity;
inline constexpr strategy reserve_n = gallocr_parity;
inline constexpr strategy reserve = gallocr_parity;
inline constexpr strategy alloc_graph = gallocr_parity;

}  // namespace emel::buffer_planner::default_strategies

namespace emel::buffer_planner::action::detail {

inline const emel::buffer_planner::strategy * resolve_strategy(const context & ctx) noexcept {
  return ctx.strategy == nullptr ? &emel::buffer_planner::default_strategies::gallocr_parity
                                 : ctx.strategy;
}

}  // namespace detail

namespace emel::buffer_planner::action {

struct no_op {
  template <class Event>
  void operator()(const Event &, context &) const noexcept {}
};

struct begin_plan {
  void operator()(const event::plan & ev, context & ctx) const noexcept {
    ctx = {};
    ctx.graph = ev.graph;
    ctx.node_buffer_ids = ev.node_buffer_ids;
    ctx.leaf_buffer_ids = ev.leaf_buffer_ids;
    ctx.buffer_count = ev.buffer_count;
    ctx.size_only = ev.size_only;
    ctx.sizes_out = ev.sizes_out;
    ctx.sizes_out_count = ev.sizes_out_count;
    ctx.error_out = ev.error_out;
    ctx.strategy = ev.strategy;
    ctx.pending_error = detail::valid_plan_event(ev) && detail::valid_strategy(ev.strategy)
                        ? EMEL_OK
                        : EMEL_ERR_INVALID_ARGUMENT;
    if (ctx.error_out != nullptr) *ctx.error_out = ctx.pending_error;
  }
};

struct on_reset_done {
  void operator()(const event::reset_done &, context & ctx) const noexcept {
    ctx.current_bytes_by_buffer.fill(0);
    ctx.bytes_by_buffer.fill(0);
    detail::reset_layouts(ctx);
    ctx.tensors.fill(tensor_record{});
    ctx.tensor_count = 0;
    ctx.total_bytes = 0;
    ctx.planned_nodes = 0;
    ctx.planned_leafs = 0;
    ctx.reference_edges = 0;
    if (!detail::register_graph_tensors(ctx)) {
      ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
    }
  }
};

struct on_seed_leafs_done {
  void operator()(const event::seed_leafs_done &, context & ctx) const noexcept {
    const auto * strategy = detail::resolve_strategy(ctx);
    if (strategy == nullptr || strategy->seed_leafs == nullptr) {
      ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    strategy->seed_leafs(ctx);
  }
};

struct on_count_references_done {
  void operator()(const event::count_references_done &, context & ctx) const noexcept {
    const auto * strategy = detail::resolve_strategy(ctx);
    if (strategy == nullptr || strategy->count_references == nullptr) {
      ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    strategy->count_references(ctx);
  }
};

struct on_alloc_explicit_inputs_done {
  void operator()(const event::alloc_explicit_inputs_done &, context & ctx) const noexcept {
    const auto * strategy = detail::resolve_strategy(ctx);
    if (strategy == nullptr || strategy->alloc_explicit_inputs == nullptr) {
      ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    strategy->alloc_explicit_inputs(ctx);
  }
};

struct on_plan_nodes_done {
  void operator()(const event::plan_nodes_done &, context & ctx) const noexcept {
    const auto * strategy = detail::resolve_strategy(ctx);
    if (strategy == nullptr || strategy->plan_nodes == nullptr) {
      ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    strategy->plan_nodes(ctx);
  }
};

struct on_release_expired_done {
  void operator()(const event::release_expired_done &, context & ctx) const noexcept {
    const auto * strategy = detail::resolve_strategy(ctx);
    if (strategy == nullptr || strategy->release_expired == nullptr) {
      ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    strategy->release_expired(ctx);
  }
};

struct on_finalize_done {
  void operator()(const event::finalize_done &, context & ctx) const noexcept {
    const auto * strategy = detail::resolve_strategy(ctx);
    if (strategy == nullptr || strategy->finalize == nullptr) {
      ctx.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      return;
    }
    strategy->finalize(ctx);
  }
};

struct record_phase_error {
  template <class ErrorEvent>
  void operator()(const ErrorEvent & ev, context & ctx) const noexcept {
    ctx.pending_error = detail::normalize_error(ev.err);
  }
};

struct on_plan_done {
  void operator()(const events::plan_done &, context & ctx) const noexcept {
    ctx.last_error = EMEL_OK;
    if (ctx.error_out != nullptr) *ctx.error_out = EMEL_OK;
  }
};

struct on_plan_error {
  void operator()(const events::plan_error & ev, context & ctx) const noexcept {
    ctx.last_error = detail::normalize_error(ev.err);
    if (ctx.error_out != nullptr) *ctx.error_out = ctx.last_error;
  }
};

inline constexpr no_op no_op{};
inline constexpr begin_plan begin_plan{};
inline constexpr on_reset_done on_reset_done{};
inline constexpr on_seed_leafs_done on_seed_leafs_done{};
inline constexpr on_count_references_done on_count_references_done{};
inline constexpr on_alloc_explicit_inputs_done on_alloc_explicit_inputs_done{};
inline constexpr on_plan_nodes_done on_plan_nodes_done{};
inline constexpr on_release_expired_done on_release_expired_done{};
inline constexpr on_finalize_done on_finalize_done{};
inline constexpr record_phase_error record_phase_error{};
inline constexpr on_plan_done on_plan_done{};
inline constexpr on_plan_error on_plan_error{};

}  // namespace emel::buffer_planner::action
