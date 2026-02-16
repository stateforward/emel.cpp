#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include "emel/buffer/planner/events.hpp"
#include "emel/emel.h"

namespace emel::buffer::planner::action {
struct context;
}

namespace emel::buffer::planner {

struct strategy {
  int32_t (*seed_leafs)(action::context &) noexcept = nullptr;
  int32_t (*count_references)(action::context &) noexcept = nullptr;
  int32_t (*alloc_explicit_inputs)(action::context &) noexcept = nullptr;
  int32_t (*plan_nodes)(action::context &) noexcept = nullptr;
  int32_t (*release_expired)(action::context &) noexcept = nullptr;
  int32_t (*finalize)(action::context &) noexcept = nullptr;
};

}  // namespace emel::buffer::planner

namespace emel::buffer::planner::action {

inline constexpr int32_t k_max_buffers = 16;
inline constexpr int32_t k_max_tensors = 2048;
inline constexpr int32_t k_max_free_blocks = 256;
inline constexpr int32_t k_max_chunks_per_buffer = 16;
inline constexpr int32_t k_max_chunk_plan_entries = k_max_buffers * k_max_chunks_per_buffer;
inline constexpr int32_t k_default_alignment = 16;
inline constexpr int32_t k_default_max_size = std::numeric_limits<int32_t>::max();

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
  int32_t buffer_count = 0;
  int32_t node_count = 0;
  int32_t leaf_count = 0;
  std::array<emel::buffer::allocator::event::tensor_desc, k_max_tensors> nodes = {};
  std::array<emel::buffer::allocator::event::tensor_desc, k_max_tensors> leafs = {};
  std::array<int32_t, k_max_tensors> node_buffer_ids = {};
  std::array<int32_t, k_max_tensors> leaf_buffer_ids = {};
  emel::buffer::planner::strategy strategy = {};
  std::array<int32_t, k_max_buffers> buffer_alignments = {};
  std::array<int32_t, k_max_buffers> buffer_max_sizes = {};
  std::array<int32_t, k_max_buffers> current_bytes_by_buffer = {};
  std::array<int32_t, k_max_buffers> bytes_by_buffer = {};
  std::array<buffer_layout, k_max_buffers> buffer_layouts = {};
  std::array<int32_t, k_max_buffers> chunk_counts = {};
  std::array<int32_t, k_max_chunk_plan_entries> chunk_sizes = {};
  int32_t total_chunk_count = 0;
  std::array<tensor_record, k_max_tensors> tensors = {};
  int32_t tensor_count = 0;
  int32_t total_bytes = 0;
  int32_t planned_nodes = 0;
  int32_t planned_leafs = 0;
  int32_t reference_edges = 0;
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

inline bool valid_alignment(const int32_t alignment) noexcept {
  return alignment > 0 && (alignment & (alignment - 1)) == 0;
}

inline int32_t sanitize_alignment(const int32_t alignment) noexcept {
  return valid_alignment(alignment) ? alignment : k_default_alignment;
}

inline int32_t sanitize_max_size(const int32_t max_size) noexcept {
  return max_size <= 0 ? k_default_max_size : max_size;
}

inline int32_t alignment_for_buffer(const context & ctx, const int32_t buffer_id) noexcept {
  if (buffer_id < 0 || buffer_id >= ctx.buffer_count) {
    return k_default_alignment;
  }
  return sanitize_alignment(ctx.buffer_alignments[buffer_id]);
}

inline int32_t max_size_for_buffer(const context & ctx, const int32_t buffer_id) noexcept {
  if (buffer_id < 0 || buffer_id >= ctx.buffer_count) {
    return k_default_max_size;
  }
  return sanitize_max_size(ctx.buffer_max_sizes[buffer_id]);
}

inline int32_t align_up(const int32_t value, const int32_t alignment) noexcept {
  if (value <= 0) {
    return 0;
  }
  const int32_t align = sanitize_alignment(alignment);
  const int64_t aligned =
    (static_cast<int64_t>(value) + static_cast<int64_t>(align) - 1) &
    ~static_cast<int64_t>(align - 1);
  if (aligned > std::numeric_limits<int32_t>::max()) {
    return std::numeric_limits<int32_t>::max();
  }
  return static_cast<int32_t>(aligned);
}

inline int32_t aligned_alloc_size(
    const context & ctx, const int32_t buffer_id, const int32_t size) noexcept {
  return align_up(size, alignment_for_buffer(ctx, buffer_id));
}

inline int32_t chunk_plan_index(const int32_t buffer_id, const int32_t chunk_index) noexcept {
  return buffer_id * k_max_chunks_per_buffer + chunk_index;
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
  const int32_t alloc_size = align_up(size, alignment_for_buffer(ctx, buffer_id));
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
  return insert_free_block(
    ctx.buffer_layouts[buffer_id], offset, align_up(size, alignment_for_buffer(ctx, buffer_id)));
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
    const emel::buffer::allocator::event::tensor_desc & tensor) noexcept {
  return !tensor.has_external_data && !tensor.is_view && tensor.alloc_size > 0;
}

inline bool register_tensor(
    context & ctx, const emel::buffer::allocator::event::tensor_desc & tensor) noexcept {
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
  rec.alloc_size = tensor.alloc_size;
  rec.view_src_id = tensor.view_src_id;
  rec.is_view = tensor.is_view;
  rec.is_input = tensor.is_input;
  rec.is_output = tensor.is_output;
  rec.allocatable = can_allocate_tensor(tensor);
  return true;
}

inline bool register_graph_tensors(context & ctx) noexcept {
  for (int32_t i = 0; i < ctx.leaf_count; ++i) {
    if (!register_tensor(ctx, ctx.leafs[i])) {
      return false;
    }
  }
  for (int32_t i = 0; i < ctx.node_count; ++i) {
    if (!register_tensor(ctx, ctx.nodes[i])) {
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
  const int32_t aligned_size = aligned_alloc_size(ctx, buffer_id, rec.alloc_size);
  if (!alloc_bytes_from_layout(ctx, buffer_id, rec.alloc_size, offset)) {
    return false;
  }

  rec.buffer_id = buffer_id;
  rec.alloc_offset = offset;
  rec.alloc_reserved = aligned_size;
  rec.allocated = true;
  ctx.current_bytes_by_buffer[buffer_id] =
    sat_add(ctx.current_bytes_by_buffer[buffer_id], aligned_size);
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
  if (ev.chunk_counts_out != nullptr && ev.chunk_counts_out_count < ev.buffer_count) {
    return false;
  }
  if (ev.buffer_alignments != nullptr) {
    for (int32_t i = 0; i < ev.buffer_count; ++i) {
      if (!valid_alignment(ev.buffer_alignments[i])) {
        return false;
      }
    }
  }
  if (ev.buffer_max_sizes != nullptr) {
    for (int32_t i = 0; i < ev.buffer_count; ++i) {
      if (ev.buffer_max_sizes[i] < 0) {
        return false;
      }
      if (ev.buffer_max_sizes[i] == 0 || ev.buffer_max_sizes[i] == k_default_max_size) {
        continue;
      }
      const int32_t alignment =
        ev.buffer_alignments != nullptr ? ev.buffer_alignments[i] : k_default_alignment;
      if (!valid_alignment(alignment)) {
        return false;
      }
      if (ev.buffer_max_sizes[i] < alignment) {
        return false;
      }
      if ((ev.buffer_max_sizes[i] % alignment) != 0) {
        return false;
      }
    }
  }
  if (ev.owner_sm == nullptr || ev.dispatch_done == nullptr || ev.dispatch_error == nullptr) {
    return false;
  }
  return true;
}

inline bool valid_strategy(const emel::buffer::planner::strategy * strategy) noexcept {
  if (strategy == nullptr) {
    return true;
  }
  return strategy->seed_leafs != nullptr && strategy->count_references != nullptr &&
         strategy->alloc_explicit_inputs != nullptr && strategy->plan_nodes != nullptr &&
         strategy->release_expired != nullptr && strategy->finalize != nullptr;
}

inline int32_t default_seed_leafs(context & ctx) noexcept {
  for (int32_t i = 0; i < ctx.leaf_count; ++i) {
    const auto & leaf = ctx.leafs[i];
    auto * rec = detail::find_record(ctx, leaf.tensor_id);
    if (rec == nullptr) {
      return EMEL_ERR_INVALID_ARGUMENT;
    }
    const int32_t buffer_id = detail::buffer_id_for(ctx.leaf_buffer_ids.data(), i);
    const bool was_allocated = rec->allocated;
    if (!detail::allocate_record(ctx, *rec, buffer_id)) {
      return EMEL_ERR_INVALID_ARGUMENT;
    }
    if (!was_allocated && rec->allocated) {
      ctx.planned_leafs = detail::sat_add(ctx.planned_leafs, 1);
    }
  }
  return EMEL_OK;
}

inline int32_t default_count_references(context & ctx) noexcept {
  for (int32_t i = 0; i < ctx.node_count; ++i) {
    const auto & node = ctx.nodes[i];
    auto * node_rec = detail::find_record(ctx, node.tensor_id);
    if (node_rec == nullptr) {
      return EMEL_ERR_INVALID_ARGUMENT;
    }
    if (node.is_view) {
      if (node.view_src_id < 0) {
        return EMEL_ERR_INVALID_ARGUMENT;
      }
      auto * view_src = detail::find_record(ctx, node.view_src_id);
      if (view_src == nullptr) {
        return EMEL_ERR_INVALID_ARGUMENT;
      }
      view_src->n_views = detail::sat_add(view_src->n_views, 1);
    }
    for (int32_t j = 0; j < emel::buffer::allocator::event::k_max_sources; ++j) {
      const int32_t src_id = node.src_ids[j];
      if (src_id < 0) {
        continue;
      }
      auto * src = detail::find_record(ctx, src_id);
      if (src == nullptr) {
        return EMEL_ERR_INVALID_ARGUMENT;
      }
      src->n_children = detail::sat_add(src->n_children, 1);
      ctx.reference_edges = detail::sat_add(ctx.reference_edges, 1);
    }
  }
  return EMEL_OK;
}

inline int32_t default_alloc_explicit_inputs(context & ctx) noexcept {
  for (int32_t i = 0; i < ctx.leaf_count; ++i) {
    const auto & leaf = ctx.leafs[i];
    if (!leaf.is_input) {
      continue;
    }
    const int32_t buffer_id = detail::buffer_id_for(ctx.leaf_buffer_ids.data(), i);
    auto * leaf_rec = detail::find_record(ctx, leaf.tensor_id);
    if (leaf_rec == nullptr) {
      return EMEL_ERR_INVALID_ARGUMENT;
    }
    const bool was_allocated = leaf_rec->allocated;
    if (!detail::allocate_record(ctx, *leaf_rec, buffer_id)) {
      return EMEL_ERR_INVALID_ARGUMENT;
    }
    if (!was_allocated && leaf_rec->allocated) {
      ctx.planned_leafs = detail::sat_add(ctx.planned_leafs, 1);
    }
  }

  for (int32_t i = 0; i < ctx.node_count; ++i) {
    const auto & node = ctx.nodes[i];
    const int32_t buffer_id = detail::buffer_id_for(ctx.node_buffer_ids.data(), i);
    auto * node_rec = detail::find_record(ctx, node.tensor_id);
    if (node_rec == nullptr) {
      return EMEL_ERR_INVALID_ARGUMENT;
    }

    if (node.is_input) {
      const bool was_allocated = node_rec->allocated;
      if (!detail::allocate_record(ctx, *node_rec, buffer_id)) {
        return EMEL_ERR_INVALID_ARGUMENT;
      }
      if (!was_allocated && node_rec->allocated) {
        ctx.planned_nodes = detail::sat_add(ctx.planned_nodes, 1);
      }
    }

    for (int32_t j = 0; j < emel::buffer::allocator::event::k_max_sources; ++j) {
      const int32_t src_id = node.src_ids[j];
      if (src_id < 0) {
        continue;
      }
      auto * src = detail::find_record(ctx, src_id);
      if (src == nullptr) {
        return EMEL_ERR_INVALID_ARGUMENT;
      }
      if (!src->is_input) {
        continue;
      }
      const bool was_allocated = src->allocated;
      if (!detail::allocate_record(ctx, *src, buffer_id)) {
        return EMEL_ERR_INVALID_ARGUMENT;
      }
      if (!was_allocated && src->allocated) {
        ctx.planned_nodes = detail::sat_add(ctx.planned_nodes, 1);
      }
    }
  }
  return EMEL_OK;
}

inline int32_t default_plan_nodes(context & ctx) noexcept {
  for (int32_t i = 0; i < ctx.node_count; ++i) {
    const auto & node = ctx.nodes[i];
    const int32_t buffer_id = detail::buffer_id_for(ctx.node_buffer_ids.data(), i);
    auto * node_rec = detail::find_record(ctx, node.tensor_id);
    if (node_rec == nullptr) {
      return EMEL_ERR_INVALID_ARGUMENT;
    }

    for (int32_t j = 0; j < emel::buffer::allocator::event::k_max_sources; ++j) {
      const int32_t src_id = node.src_ids[j];
      if (src_id < 0) {
        continue;
      }
      auto * parent = detail::find_record(ctx, src_id);
      if (parent == nullptr) {
        return EMEL_ERR_INVALID_ARGUMENT;
      }
      if (!detail::allocate_record(ctx, *parent, buffer_id)) {
        return EMEL_ERR_INVALID_ARGUMENT;
      }
    }

    if (!node.is_input && node_rec->allocatable && !node_rec->allocated) {
      bool reused_parent = false;
      if (node.can_inplace) {
      for (int32_t j = 0; j < emel::buffer::allocator::event::k_max_sources; ++j) {
        const int32_t src_id = node.src_ids[j];
        if (src_id < 0) {
          continue;
        }
        auto * parent = detail::find_record(ctx, src_id);
        if (parent == nullptr) {
          return EMEL_ERR_INVALID_ARGUMENT;
        }
        tensor_record * reuse_owner = parent;
        bool reuse_from_view_src = false;
        if (parent->is_view) {
          if (parent->view_src_id < 0) {
            continue;
          }
          auto * view_src = detail::find_record(ctx, parent->view_src_id);
          if (view_src == nullptr) {
            return EMEL_ERR_INVALID_ARGUMENT;
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
        const int32_t node_aligned =
          detail::aligned_alloc_size(ctx, buffer_id, node_rec->alloc_size);
        const int32_t reuse_aligned =
          detail::aligned_alloc_size(ctx, reuse_owner->buffer_id, reuse_owner->alloc_size);
        if (reuse_aligned < node_aligned) {
          continue;
        }

        node_rec->buffer_id = reuse_owner->buffer_id;
        node_rec->alloc_offset = reuse_owner->alloc_offset;
        node_rec->alloc_reserved = node_aligned;
        node_rec->allocated = true;
        const int32_t extra = reuse_owner->alloc_reserved - node_aligned;
        if (extra > 0) {
          if (!detail::free_bytes_to_layout(
                ctx,
                reuse_owner->buffer_id,
                detail::sat_add(reuse_owner->alloc_offset, node_rec->alloc_reserved),
                extra)) {
            return EMEL_ERR_BACKEND;
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
      }

      if (!reused_parent && !detail::allocate_record(ctx, *node_rec, buffer_id)) {
        return EMEL_ERR_INVALID_ARGUMENT;
      }

      if (node_rec->allocated) {
        ctx.planned_nodes = detail::sat_add(ctx.planned_nodes, 1);
      }
    }

    for (int32_t j = 0; j < emel::buffer::allocator::event::k_max_sources; ++j) {
      const int32_t src_id = node.src_ids[j];
      if (src_id < 0) {
        continue;
      }
      auto * parent = detail::find_record(ctx, src_id);
      if (parent == nullptr) {
        return EMEL_ERR_INVALID_ARGUMENT;
      }
      parent->n_children = parent->n_children <= 0 ? 0 : parent->n_children - 1;

      if (parent->n_children == 0 && parent->n_views == 0) {
        if (parent->is_view) {
        if (parent->view_src_id < 0) {
          return EMEL_ERR_INVALID_ARGUMENT;
        }
        auto * view_src = detail::find_record(ctx, parent->view_src_id);
        if (view_src == nullptr) {
          return EMEL_ERR_INVALID_ARGUMENT;
        }
          view_src->n_views = view_src->n_views <= 0 ? 0 : view_src->n_views - 1;
          if (view_src->n_views == 0 && view_src->n_children == 0 && view_src->allocated &&
              !view_src->is_output) {
          if (!detail::free_record(ctx, *view_src)) {
              return EMEL_ERR_BACKEND;
          }
          }
        } else if (parent->allocated && !parent->is_output) {
          if (!detail::free_record(ctx, *parent)) {
            return EMEL_ERR_BACKEND;
          }
        }
      }
    }
  }
  return EMEL_OK;
}

inline int32_t default_release_expired(context &) noexcept { return EMEL_OK; }

inline int32_t default_finalize(context & ctx) noexcept {
  ctx.total_bytes = 0;
  for (int32_t i = 0; i < ctx.buffer_count; ++i) {
    ctx.total_bytes = detail::sat_add(ctx.total_bytes, ctx.bytes_by_buffer[i]);
  }
  return EMEL_OK;
}

inline constexpr emel::buffer::planner::strategy make_gallocr_strategy() noexcept {
  return emel::buffer::planner::strategy{
    .seed_leafs = default_seed_leafs,
    .count_references = default_count_references,
    .alloc_explicit_inputs = default_alloc_explicit_inputs,
    .plan_nodes = default_plan_nodes,
    .release_expired = default_release_expired,
    .finalize = default_finalize,
  };
}

}  // namespace detail

}  // namespace emel::buffer::planner::action

namespace emel::buffer::planner::default_strategies {

inline constexpr strategy gallocr_parity = action::detail::make_gallocr_strategy();
inline constexpr strategy reserve_n_size = gallocr_parity;
inline constexpr strategy reserve_n = gallocr_parity;
inline constexpr strategy reserve = gallocr_parity;
inline constexpr strategy alloc_graph = gallocr_parity;

}  // namespace emel::buffer::planner::default_strategies

namespace emel::buffer::planner::action::detail {

inline const emel::buffer::planner::strategy * resolve_strategy(const context & ctx) noexcept {
  return &ctx.strategy;
}

inline int32_t run_reset(context & ctx) noexcept {
  ctx.current_bytes_by_buffer.fill(0);
  ctx.bytes_by_buffer.fill(0);
  reset_layouts(ctx);
  ctx.chunk_counts.fill(0);
  ctx.chunk_sizes.fill(0);
  ctx.total_chunk_count = 0;
  ctx.tensors.fill(tensor_record{});
  ctx.tensor_count = 0;
  ctx.total_bytes = 0;
  ctx.planned_nodes = 0;
  ctx.planned_leafs = 0;
  ctx.reference_edges = 0;
  if (!register_graph_tensors(ctx)) {
    return EMEL_ERR_INVALID_ARGUMENT;
  }
  return EMEL_OK;
}

inline int32_t run_seed_leafs(context & ctx) noexcept {
  const auto * strategy = resolve_strategy(ctx);
  if (strategy == nullptr || strategy->seed_leafs == nullptr) {
    return EMEL_ERR_INVALID_ARGUMENT;
  }
  return strategy->seed_leafs(ctx);
}

inline int32_t run_count_references(context & ctx) noexcept {
  const auto * strategy = resolve_strategy(ctx);
  if (strategy == nullptr || strategy->count_references == nullptr) {
    return EMEL_ERR_INVALID_ARGUMENT;
  }
  return strategy->count_references(ctx);
}

inline int32_t run_alloc_explicit_inputs(context & ctx) noexcept {
  const auto * strategy = resolve_strategy(ctx);
  if (strategy == nullptr || strategy->alloc_explicit_inputs == nullptr) {
    return EMEL_ERR_INVALID_ARGUMENT;
  }
  return strategy->alloc_explicit_inputs(ctx);
}

inline int32_t run_plan_nodes(context & ctx) noexcept {
  const auto * strategy = resolve_strategy(ctx);
  if (strategy == nullptr || strategy->plan_nodes == nullptr) {
    return EMEL_ERR_INVALID_ARGUMENT;
  }
  return strategy->plan_nodes(ctx);
}

inline int32_t run_release_expired(context & ctx) noexcept {
  const auto * strategy = resolve_strategy(ctx);
  if (strategy == nullptr || strategy->release_expired == nullptr) {
    return EMEL_ERR_INVALID_ARGUMENT;
  }
  return strategy->release_expired(ctx);
}

inline int32_t run_finalize(context & ctx, const event::plan * request) noexcept {
  const auto * strategy = resolve_strategy(ctx);
  if (strategy == nullptr || strategy->finalize == nullptr) {
    return EMEL_ERR_INVALID_ARGUMENT;
  }
  const int32_t err = strategy->finalize(ctx);
  if (err == EMEL_OK && request != nullptr && request->sizes_out != nullptr) {
    for (int32_t i = 0; i < ctx.buffer_count && i < request->sizes_out_count; ++i) {
      request->sizes_out[i] = ctx.bytes_by_buffer[i];
    }
  }
  return err;
}

inline int32_t run_split_required(context & ctx, const event::plan * request) noexcept {
  ctx.chunk_counts.fill(0);
  ctx.chunk_sizes.fill(0);
  ctx.total_chunk_count = 0;

  for (int32_t i = 0; i < ctx.buffer_count; ++i) {
    int32_t remaining = ctx.bytes_by_buffer[i];
    if (remaining <= 0) {
      continue;
    }

    const int32_t alignment = alignment_for_buffer(ctx, i);
    const int32_t max_size = max_size_for_buffer(ctx, i);
    if (max_size <= 0 || max_size == k_default_max_size || max_size >= remaining) {
      const int32_t aligned = align_up(remaining, alignment);
      ctx.chunk_sizes[chunk_plan_index(i, 0)] = aligned;
      ctx.chunk_counts[i] = 1;
      ctx.total_chunk_count = sat_add(ctx.total_chunk_count, 1);
      continue;
    }

    int32_t count = 0;
    while (remaining > 0) {
      if (count >= k_max_chunks_per_buffer) {
        return EMEL_ERR_INVALID_ARGUMENT;
      }
      const int32_t chunk_size = remaining > max_size ? max_size : remaining;
      const int32_t aligned = align_up(chunk_size, alignment);
      if (aligned > max_size) {
        return EMEL_ERR_INVALID_ARGUMENT;
      }
      ctx.chunk_sizes[chunk_plan_index(i, count)] = aligned;
      remaining = sat_sub_floor_zero(remaining, aligned);
      count += 1;
    }
    ctx.chunk_counts[i] = count;
    ctx.total_chunk_count = sat_add(ctx.total_chunk_count, count);
    if (ctx.total_chunk_count > k_max_chunk_plan_entries) {
      return EMEL_ERR_INVALID_ARGUMENT;
    }
  }

  if (request != nullptr) {
    if (request->chunk_sizes_out != nullptr &&
        request->chunk_sizes_out_count >= k_max_chunk_plan_entries) {
      for (int32_t i = 0; i < k_max_chunk_plan_entries; ++i) {
        request->chunk_sizes_out[i] = ctx.chunk_sizes[i];
      }
    }
    if (request->chunk_counts_out != nullptr &&
        request->chunk_counts_out_count >= ctx.buffer_count) {
      for (int32_t i = 0; i < ctx.buffer_count; ++i) {
        request->chunk_counts_out[i] = ctx.chunk_counts[i];
      }
    }
  }

  return EMEL_OK;
}

}  // namespace detail

namespace emel::buffer::planner::action {

struct no_op {
  template <class Event>
  void operator()(const Event &, context &) const noexcept {}
};

struct begin_plan {
  void operator()(const event::plan & ev, context & ctx) const noexcept {
    ctx = {};
    ctx.buffer_count = ev.buffer_count;
    const bool valid = detail::valid_plan_event(ev) && detail::valid_strategy(ev.strategy);
    const int32_t err = valid ? EMEL_OK : EMEL_ERR_INVALID_ARGUMENT;
    if (valid) {
      for (int32_t i = 0; i < ctx.buffer_count; ++i) {
        const int32_t alignment =
          ev.buffer_alignments != nullptr ? ev.buffer_alignments[i] : k_default_alignment;
        const int32_t max_size =
          ev.buffer_max_sizes != nullptr ? ev.buffer_max_sizes[i] : k_default_max_size;
        ctx.buffer_alignments[i] = detail::sanitize_alignment(alignment);
        ctx.buffer_max_sizes[i] = detail::sanitize_max_size(max_size);
      }
      ctx.node_count = ev.graph.n_nodes;
      ctx.leaf_count = ev.graph.n_leafs;
      for (int32_t i = 0; i < ctx.leaf_count; ++i) {
        ctx.leafs[i] = ev.graph.leafs[i];
        ctx.leaf_buffer_ids[i] = detail::buffer_id_for(ev.leaf_buffer_ids, i);
      }
      for (int32_t i = 0; i < ctx.node_count; ++i) {
        ctx.nodes[i] = ev.graph.nodes[i];
        ctx.node_buffer_ids[i] = detail::buffer_id_for(ev.node_buffer_ids, i);
      }
      ctx.strategy = ev.strategy == nullptr
          ? emel::buffer::planner::default_strategies::gallocr_parity
          : *ev.strategy;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
  }
};

struct reject_plan {
  void operator()(const event::plan & ev, context &) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
  }
};

struct on_reset_done {
  void operator()(const event::reset_done & ev, context & ctx) const noexcept {
    const int32_t err = detail::run_reset(ctx);
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
  }
};

struct on_seed_leafs_done {
  void operator()(const event::seed_leafs_done & ev, context & ctx) const noexcept {
    const int32_t err = detail::run_seed_leafs(ctx);
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
  }
};

struct on_count_references_done {
  void operator()(const event::count_references_done & ev, context & ctx) const noexcept {
    const int32_t err = detail::run_count_references(ctx);
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
  }
};

struct on_alloc_explicit_inputs_done {
  void operator()(const event::alloc_explicit_inputs_done & ev, context & ctx) const noexcept {
    const int32_t err = detail::run_alloc_explicit_inputs(ctx);
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
  }
};

struct on_plan_nodes_done {
  void operator()(const event::plan_nodes_done & ev, context & ctx) const noexcept {
    const int32_t err = detail::run_plan_nodes(ctx);
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
  }
};

struct on_release_expired_done {
  void operator()(const event::release_expired_done & ev, context & ctx) const noexcept {
    const int32_t err = detail::run_release_expired(ctx);
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
  }
};

struct on_finalize_done {
  void operator()(const event::finalize_done & ev, context & ctx) const noexcept {
    const int32_t err = detail::run_finalize(ctx, ev.request);
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
  }
};

struct on_split_required_done {
  void operator()(const event::split_required_done & ev, context & ctx) const noexcept {
    const int32_t err = detail::run_split_required(ctx, ev.request);
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
  }
};

struct record_phase_error {
  template <class ErrorEvent>
  void operator()(const ErrorEvent & ev, context &) const noexcept {
    (void)ev;
  }
};

struct on_plan_done {
  void operator()(const events::plan_done & ev, context &) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
  }
};

struct on_plan_error {
  void operator()(const events::plan_error & ev, context &) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = detail::normalize_error(ev.err);
    }
  }
};

struct on_unexpected {
  template <class Event>
  void operator()(const Event & ev, context &) const noexcept {
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      }
    }
  }
};

inline constexpr no_op no_op{};
inline constexpr begin_plan begin_plan{};
inline constexpr reject_plan reject_plan{};
inline constexpr on_reset_done on_reset_done{};
inline constexpr on_seed_leafs_done on_seed_leafs_done{};
inline constexpr on_count_references_done on_count_references_done{};
inline constexpr on_alloc_explicit_inputs_done on_alloc_explicit_inputs_done{};
inline constexpr on_plan_nodes_done on_plan_nodes_done{};
inline constexpr on_release_expired_done on_release_expired_done{};
inline constexpr on_finalize_done on_finalize_done{};
inline constexpr on_split_required_done on_split_required_done{};
inline constexpr record_phase_error record_phase_error{};
inline constexpr on_plan_done on_plan_done{};
inline constexpr on_plan_error on_plan_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::buffer::planner::action
