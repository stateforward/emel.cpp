#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include "emel/buffer/chunk_allocator/events.hpp"
#include "emel/buffer/chunk_allocator/sm.hpp"
#include "emel/buffer/allocator/events.hpp"
#include "emel/buffer/planner/events.hpp"
#include "emel/buffer/planner/sm.hpp"
#include "emel/buffer/realloc_analyzer/events.hpp"
#include "emel/buffer/realloc_analyzer/sm.hpp"
#include "emel/emel.h"

namespace emel::buffer::allocator::action {

inline constexpr int32_t k_max_buffers = 16;
inline constexpr int32_t k_max_graph_tensors = 2048;
inline constexpr int32_t k_max_chunks_per_buffer = 16;
inline constexpr int32_t k_max_chunk_bindings = k_max_buffers * k_max_chunks_per_buffer;
inline constexpr int32_t k_default_alignment = 16;
inline constexpr int32_t k_default_max_size = std::numeric_limits<int32_t>::max();

using tensor_alloc = emel::buffer::realloc_analyzer::event::tensor_alloc;
using node_alloc = emel::buffer::realloc_analyzer::event::node_alloc;
using leaf_alloc = emel::buffer::realloc_analyzer::event::leaf_alloc;

struct context {
  int32_t buffer_count = 0;
  int32_t init_epoch = 0;
  int32_t reserve_epoch = 0;
  int32_t alloc_epoch = 0;
  int32_t release_epoch = 0;
  uint32_t step = 0;
  int32_t phase_error = EMEL_OK;
  int32_t last_n_nodes = 0;
  int32_t last_n_leafs = 0;
  bool has_reserve_snapshot = false;
  std::array<int32_t, k_max_buffers> buffer_alignments = {};
  std::array<int32_t, k_max_buffers> buffer_max_sizes = {};
  std::array<int32_t, k_max_buffers> committed_sizes = {};
  std::array<int32_t, k_max_buffers> committed_chunk_counts = {};
  std::array<int32_t, k_max_chunk_bindings> committed_chunk_ids = {};
  std::array<uint64_t, k_max_chunk_bindings> committed_chunk_offsets = {};
  std::array<uint64_t, k_max_chunk_bindings> committed_chunk_sizes = {};
  std::array<int32_t, k_max_graph_tensors> last_node_buffer_ids = {};
  std::array<int32_t, k_max_graph_tensors> last_leaf_buffer_ids = {};
  std::array<node_alloc, k_max_graph_tensors> node_allocs = {};
  std::array<leaf_alloc, k_max_graph_tensors> leaf_allocs = {};
};

namespace detail {

inline int32_t normalize_error(const int32_t err, const int32_t fallback) noexcept {
  if (err != 0) return err;
  if (fallback != 0) return fallback;
  return EMEL_ERR_BACKEND;
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

inline int32_t alignment_for_buffer(const context & c, const int32_t buffer_id) noexcept {
  if (buffer_id < 0 || buffer_id >= c.buffer_count) {
    return k_default_alignment;
  }
  return sanitize_alignment(c.buffer_alignments[buffer_id]);
}

inline int32_t max_size_for_buffer(const context & c, const int32_t buffer_id) noexcept {
  if (buffer_id < 0 || buffer_id >= c.buffer_count) {
    return k_default_max_size;
  }
  return sanitize_max_size(c.buffer_max_sizes[buffer_id]);
}

inline bool dispatch_plan_done(
    void *, const emel::buffer::planner::events::plan_done &) noexcept {
  return true;
}

inline bool dispatch_plan_error(
    void *, const emel::buffer::planner::events::plan_error &) noexcept {
  return true;
}

inline bool valid_graph_tensors(const event::graph_view & g) noexcept {
  if (g.n_nodes < 0 || g.n_leafs < 0) return false;
  if (g.n_nodes > k_max_graph_tensors || g.n_leafs > k_max_graph_tensors) return false;
  if ((g.n_nodes > 0 && g.nodes == nullptr) || (g.n_leafs > 0 && g.leafs == nullptr)) return false;
  for (int32_t i = 0; i < g.n_nodes; ++i) {
    if (g.nodes[i].tensor_id < 0 || g.nodes[i].alloc_size < 0) return false;
  }
  for (int32_t i = 0; i < g.n_leafs; ++i) {
    if (g.leafs[i].tensor_id < 0 || g.leafs[i].alloc_size < 0) return false;
  }
  return true;
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

inline bool align_up_checked(
    const int32_t value, const int32_t alignment, int32_t & out) noexcept {
  if (value <= 0) {
    out = 0;
    return true;
  }
  const int32_t align = sanitize_alignment(alignment);
  const int64_t sum = static_cast<int64_t>(value) + static_cast<int64_t>(align) - 1;
  if (sum > std::numeric_limits<int32_t>::max()) {
    return false;  // GCOVR_EXCL_LINE
  }
  const int64_t aligned = sum & ~static_cast<int64_t>(align - 1);
  if (aligned > std::numeric_limits<int32_t>::max()) {
    return false;  // GCOVR_EXCL_LINE
  }
  out = static_cast<int32_t>(aligned);
  return true;
}

inline int32_t chunk_binding_index(const int32_t buffer_id, const int32_t chunk_index) noexcept {
  return buffer_id * k_max_chunks_per_buffer + chunk_index;
}

inline int32_t get_buffer_id(const int32_t * ids, const int32_t index) noexcept {
  return ids == nullptr ? 0 : ids[index];
}

inline const event::tensor_desc * find_tensor(
    const event::graph_view & g, const int32_t tensor_id, bool & is_node, int32_t & index) noexcept {
  for (int32_t i = 0; i < g.n_nodes; ++i) {
    if (g.nodes[i].tensor_id == tensor_id) {
      is_node = true;
      index = i;
      return &g.nodes[i];
    }
  }
  for (int32_t i = 0; i < g.n_leafs; ++i) {
    if (g.leafs[i].tensor_id == tensor_id) {
      is_node = false;
      index = i;
      return &g.leafs[i];
    }
  }
  return nullptr;
}

inline bool build_tensor_alloc(
    tensor_alloc & out, const context & c, const event::tensor_desc & tensor, const int32_t buffer_id,
    const int32_t buffer_count) noexcept {
  out.tensor_id = tensor.tensor_id;
  if (tensor.has_external_data || tensor.is_view) {
    out.buffer_id = -1;
    out.size_max = 0;
    out.alignment = 0;
    return true;
  }
  if (buffer_id < 0 || buffer_id >= buffer_count) {
    return false;  // GCOVR_EXCL_LINE
  }
  out.buffer_id = buffer_id;
  out.alignment = alignment_for_buffer(c, buffer_id);
  int32_t aligned = 0;
  if (!align_up_checked(tensor.alloc_size, out.alignment, aligned)) {
    return false;  // GCOVR_EXCL_LINE
  }
  out.size_max = aligned;
  return true;
}

inline bool capture_alloc_snapshot(
    context & c, const event::graph_view & graph, const int32_t * node_buffer_ids,
    const int32_t * leaf_buffer_ids) noexcept {
  if (!valid_graph_tensors(graph)) {
    return false;  // GCOVR_EXCL_LINE
  }
  if (graph.n_nodes > static_cast<int32_t>(c.node_allocs.size()) ||
      graph.n_leafs > static_cast<int32_t>(c.leaf_allocs.size())) {
    return false;  // GCOVR_EXCL_LINE
  }

  for (int32_t i = 0; i < graph.n_nodes; ++i) {
    auto & dst = c.node_allocs[i];
    dst = {};
    const auto & node = graph.nodes[i];
    const int32_t node_buffer_id = get_buffer_id(node_buffer_ids, i);
    if (!build_tensor_alloc(dst.dst, c, node, node_buffer_id, c.buffer_count)) {
      return false;  // GCOVR_EXCL_LINE
    }

    for (int32_t j = 0; j < event::k_max_sources; ++j) {
      const int32_t src_id = node.src_ids[j];
      if (src_id < 0) {
        dst.src[j] = {};
        continue;
      }
      bool src_is_node = false;
      int32_t src_index = -1;
      const auto * src = find_tensor(graph, src_id, src_is_node, src_index);
      if (src == nullptr || src_index < 0) {
        return false;
      }
      const int32_t src_buffer_id = src_is_node ? get_buffer_id(node_buffer_ids, src_index)
                                                : get_buffer_id(leaf_buffer_ids, src_index);
      if (!build_tensor_alloc(dst.src[j], c, *src, src_buffer_id, c.buffer_count)) {
        return false;
      }
    }
  }

  for (int32_t i = 0; i < graph.n_leafs; ++i) {
    auto & dst = c.leaf_allocs[i];
    dst = {};
    const auto & leaf = graph.leafs[i];
    const int32_t leaf_buffer_id = get_buffer_id(leaf_buffer_ids, i);
    if (!build_tensor_alloc(dst.leaf, c, leaf, leaf_buffer_id, c.buffer_count)) {
      return false;
    }
  }

  for (int32_t i = graph.n_nodes; i < static_cast<int32_t>(c.node_allocs.size()); ++i) {
    c.node_allocs[i] = {};
  }
  for (int32_t i = graph.n_leafs; i < static_cast<int32_t>(c.leaf_allocs.size()); ++i) {
    c.leaf_allocs[i] = {};
  }
  c.has_reserve_snapshot = true;
  return true;
}

inline bool tensor_needs_realloc(
    const context & c, const event::tensor_desc & tensor, const tensor_alloc & alloc) noexcept {
  if (tensor.has_external_data || tensor.is_view) {
    return false;
  }
  if (alloc.buffer_id < 0) {
    return true;
  }
  const int32_t alignment =
    alloc.alignment > 0 ? alloc.alignment : alignment_for_buffer(c, alloc.buffer_id);
  int32_t aligned = 0;
  if (!align_up_checked(tensor.alloc_size, alignment, aligned)) {
    return true;
  }
  return alloc.size_max < aligned;
}

inline bool graph_needs_realloc(const event::graph_view & graph, const context & c) noexcept {
  if (!c.has_reserve_snapshot) {
    return true;
  }
  if (graph.n_nodes != c.last_n_nodes || graph.n_leafs != c.last_n_leafs) {
    return true;
  }

  for (int32_t i = 0; i < graph.n_nodes; ++i) {
    const auto & node = graph.nodes[i];
    const auto & node_alloc = c.node_allocs[i];
    if (tensor_needs_realloc(c, node, node_alloc.dst)) {
      return true;
    }

    for (int32_t j = 0; j < event::k_max_sources; ++j) {
      const int32_t src_id = node.src_ids[j];
      const auto & src_alloc = node_alloc.src[j];
      if (src_id < 0) {
        continue;
      }
      bool src_is_node = false;
      int32_t src_index = -1;
      const auto * src = find_tensor(graph, src_id, src_is_node, src_index);
      if (src == nullptr || src_index < 0 || tensor_needs_realloc(c, *src, src_alloc)) {
        return true;
      }
    }
  }

  return false;
}

inline bool valid_buffer_ids_for_graph(
    const event::graph_view & g, const int32_t * node_ids, const int32_t * leaf_ids,
    const int32_t buffer_count) noexcept {
  if (!valid_graph_tensors(g) || buffer_count <= 0) return false;
  for (int32_t i = 0; i < g.n_nodes; ++i) {
    const int32_t id = get_buffer_id(node_ids, i);
    if (id < 0 || id >= buffer_count) return false;
  }
  for (int32_t i = 0; i < g.n_leafs; ++i) {
    const int32_t id = get_buffer_id(leaf_ids, i);
    if (id < 0 || id >= buffer_count) return false;
  }
  return true;
}

inline bool run_planner(
    emel::buffer::planner::sm * planner, const event::graph_view & graph,
    const int32_t * node_buffer_ids, const int32_t * leaf_buffer_ids, const int32_t buffer_count,
    const bool size_only, int32_t * sizes_out, const int32_t sizes_out_count,
    int32_t * chunk_counts_out, const int32_t chunk_counts_out_count,
    int32_t * chunk_sizes_out, const int32_t chunk_sizes_out_count,
    const emel::buffer::planner::strategy * planner_strategy, const context & c,
    int32_t & out_error) noexcept {
  const int32_t * buffer_alignments = nullptr;
  const int32_t * buffer_max_sizes = nullptr;
  for (int32_t i = 0; i < buffer_count; ++i) {
    if (c.buffer_alignments[i] != 0) {
      buffer_alignments = c.buffer_alignments.data();
      break;
    }
  }
  for (int32_t i = 0; i < buffer_count; ++i) {
    if (c.buffer_max_sizes[i] != 0) {
      buffer_max_sizes = c.buffer_max_sizes.data();
      break;
    }
  }

  int32_t planner_error = EMEL_OK;
  const bool ok = planner->process_event(emel::buffer::planner::event::plan{
    .graph = graph,
    .node_buffer_ids = node_buffer_ids,
    .leaf_buffer_ids = leaf_buffer_ids,
    .buffer_count = buffer_count,
    .buffer_alignments = buffer_alignments,
    .buffer_max_sizes = buffer_max_sizes,
    .size_only = size_only,
    .sizes_out = sizes_out,
    .sizes_out_count = sizes_out_count,
    .chunk_sizes_out = chunk_sizes_out,
    .chunk_sizes_out_count = chunk_sizes_out_count,
    .chunk_counts_out = chunk_counts_out,
    .chunk_counts_out_count = chunk_counts_out_count,
    .error_out = &planner_error,
    .owner_sm = planner,
    .dispatch_done = &dispatch_plan_done,
    .dispatch_error = &dispatch_plan_error,
    .strategy = planner_strategy,
  });

  if (!ok || planner_error != EMEL_OK) {
    out_error = normalize_error(planner_error, EMEL_ERR_BACKEND);
    return false;
  }

  out_error = EMEL_OK;
  return true;
}

inline bool run_realloc_analyzer(
    emel::buffer::realloc_analyzer::sm * analyzer, const event::graph_view & graph,
    const context & c, bool & out_needs_realloc, int32_t & out_error) noexcept {
  int32_t analyze_error = EMEL_OK;
  int32_t needs_realloc = 0;
  const bool ok = analyzer->process_event(emel::buffer::realloc_analyzer::event::analyze{
    .graph = graph,
    .node_allocs = c.node_allocs.data(),
    .node_alloc_count = c.last_n_nodes,
    .leaf_allocs = c.leaf_allocs.data(),
    .leaf_alloc_count = c.last_n_leafs,
    .needs_realloc_out = &needs_realloc,
    .error_out = &analyze_error,
  });

  if (!ok || analyze_error != EMEL_OK) {
    out_error = normalize_error(analyze_error, EMEL_ERR_BACKEND);
    return false;
  }

  out_needs_realloc = needs_realloc != 0;
  out_error = EMEL_OK;
  return true;
}

inline void commit_sizes(
    std::array<int32_t, k_max_buffers> & committed, const std::array<int32_t, k_max_buffers> & required,
    const int32_t buffer_count) noexcept {
  for (int32_t i = 0; i < buffer_count; ++i) {
    if (required[i] > committed[i]) committed[i] = required[i];
  }
}

inline bool requires_growth(
    const std::array<int32_t, k_max_buffers> & committed, const std::array<int32_t, k_max_buffers> & required,
    const int32_t buffer_count) noexcept {
  for (int32_t i = 0; i < buffer_count; ++i) {
    if (required[i] > committed[i]) return true;
  }
  return false;
}

inline void capture_buffer_map(
    context & c, const event::graph_view & graph, const int32_t * node_buffer_ids,
    const int32_t * leaf_buffer_ids) noexcept {
  c.last_n_nodes = graph.n_nodes;
  c.last_n_leafs = graph.n_leafs;
  for (int32_t i = 0; i < graph.n_nodes && i < static_cast<int32_t>(c.last_node_buffer_ids.size()); ++i) {
    c.last_node_buffer_ids[i] = get_buffer_id(node_buffer_ids, i);
  }
  for (int32_t i = 0; i < graph.n_leafs && i < static_cast<int32_t>(c.last_leaf_buffer_ids.size()); ++i) {
    c.last_leaf_buffer_ids[i] = get_buffer_id(leaf_buffer_ids, i);
  }
  for (int32_t i = graph.n_nodes; i < static_cast<int32_t>(c.last_node_buffer_ids.size()); ++i) {
    c.last_node_buffer_ids[i] = 0;
  }
  for (int32_t i = graph.n_leafs; i < static_cast<int32_t>(c.last_leaf_buffer_ids.size()); ++i) {
    c.last_leaf_buffer_ids[i] = 0;
  }
}

inline void reset_chunk_bindings(context & c) noexcept {
  for (int32_t i = 0; i < k_max_buffers; ++i) {
    c.committed_chunk_counts[i] = 0;
  }
  for (int32_t i = 0; i < k_max_chunk_bindings; ++i) {
    c.committed_chunk_ids[i] = -1;
    c.committed_chunk_offsets[i] = 0;
    c.committed_chunk_sizes[i] = 0;
  }
}

inline bool chunk_bindings_cover_required(
    const context & c, const std::array<int32_t, k_max_buffers> & required,
    const int32_t buffer_count) noexcept {
  for (int32_t i = 0; i < buffer_count; ++i) {
    if (required[i] <= 0) {
      continue;
    }
    if (c.committed_chunk_counts[i] <= 0) {
      return false;
    }
    uint64_t total = 0;
    const int32_t count = c.committed_chunk_counts[i];
    for (int32_t j = 0; j < count; ++j) {
      const int32_t idx = chunk_binding_index(i, j);
      if (c.committed_chunk_ids[idx] < 0) {
        return false;
      }
      total += c.committed_chunk_sizes[idx];
    }
    if (total < static_cast<uint64_t>(required[i])) {
      return false;
    }
  }
  return true;
}

inline bool configure_chunk_allocator(
    emel::buffer::chunk_allocator::sm * chunk_allocator_sm, int32_t & out_error) noexcept {
  int32_t reset_error = EMEL_OK;
  if (!chunk_allocator_sm->process_event(emel::buffer::chunk_allocator::event::reset{
        .error_out = &reset_error,
      }) ||
      reset_error != EMEL_OK) {
    out_error = normalize_error(reset_error, EMEL_ERR_BACKEND);  // GCOVR_EXCL_LINE
    return false;  // GCOVR_EXCL_LINE
  }

  int32_t configure_error = EMEL_OK;
  if (!chunk_allocator_sm->process_event(emel::buffer::chunk_allocator::event::configure{
        .alignment = static_cast<uint64_t>(k_default_alignment),
        .max_chunk_size = static_cast<uint64_t>(k_default_max_size),
        .error_out = &configure_error,
      }) ||
      configure_error != EMEL_OK) {
    out_error = normalize_error(configure_error, EMEL_ERR_BACKEND);  // GCOVR_EXCL_LINE
    return false;  // GCOVR_EXCL_LINE
  }

  out_error = EMEL_OK;
  return true;
}

inline bool apply_required_sizes_to_chunks(
    context & c, const std::array<int32_t, k_max_buffers> & required,
    const int32_t * planned_chunk_counts, const int32_t * planned_chunk_sizes,
    emel::buffer::chunk_allocator::sm * chunk_allocator_sm, int32_t & out_error) noexcept {
  for (int32_t i = 0; i < c.buffer_count; ++i) {
    const int32_t target_size = required[i] > c.committed_sizes[i] ? required[i] : c.committed_sizes[i];
    if (target_size <= 0) {
      continue;
    }

    int32_t planned_count = 0;
    if (planned_chunk_counts != nullptr) {
      planned_count = planned_chunk_counts[i];
    }
    if (planned_count < 0 || planned_count > k_max_chunks_per_buffer) {
      out_error = EMEL_ERR_INVALID_ARGUMENT;
      return false;
    }

    std::array<int32_t, k_max_chunks_per_buffer> planned_sizes = {};
    if (planned_count == 0) {
      planned_count = 1;
      planned_sizes[0] = target_size;
    } else {
      if (planned_chunk_sizes == nullptr) {
        out_error = EMEL_ERR_INVALID_ARGUMENT;
        return false;
      }
      for (int32_t j = 0; j < planned_count; ++j) {
        planned_sizes[j] = planned_chunk_sizes[chunk_binding_index(i, j)];
      }
    }

    uint64_t planned_total = 0;
    for (int32_t j = 0; j < planned_count; ++j) {
      if (planned_sizes[j] <= 0) {
        out_error = EMEL_ERR_INVALID_ARGUMENT;
        return false;
      }
      planned_total += static_cast<uint64_t>(planned_sizes[j]);
    }
    if (planned_total < static_cast<uint64_t>(target_size)) {
      out_error = EMEL_ERR_INVALID_ARGUMENT;
      return false;
    }

    bool can_reuse = planned_count == c.committed_chunk_counts[i];
    if (can_reuse) {
      for (int32_t j = 0; j < planned_count; ++j) {
        const int32_t idx = chunk_binding_index(i, j);
        if (c.committed_chunk_ids[idx] < 0 ||
            c.committed_chunk_sizes[idx] < static_cast<uint64_t>(planned_sizes[j])) {
          can_reuse = false;
          break;
        }
      }
    }

    if (can_reuse) {
      continue;
    }

    const int32_t alignment = alignment_for_buffer(c, i);
    const int32_t max_size = max_size_for_buffer(c, i);
    std::array<int32_t, k_max_chunks_per_buffer> new_chunk_ids = {};
    std::array<uint64_t, k_max_chunks_per_buffer> new_offsets = {};
    std::array<uint64_t, k_max_chunks_per_buffer> new_sizes = {};

    for (int32_t j = 0; j < planned_count; ++j) {
      int32_t alloc_error = EMEL_OK;
      if (!chunk_allocator_sm->process_event(emel::buffer::chunk_allocator::event::allocate{
            .size = static_cast<uint64_t>(planned_sizes[j]),
            .alignment = static_cast<uint64_t>(alignment),
            .max_chunk_size = static_cast<uint64_t>(max_size),
            .chunk_out = &new_chunk_ids[j],
            .offset_out = &new_offsets[j],
            .aligned_size_out = &new_sizes[j],
            .error_out = &alloc_error,
          }) ||
          alloc_error != EMEL_OK) {
        for (int32_t r = 0; r < j; ++r) {
          int32_t release_error = EMEL_OK;
          (void)chunk_allocator_sm->process_event(emel::buffer::chunk_allocator::event::release{
            .chunk = new_chunk_ids[r],
            .offset = new_offsets[r],
            .size = new_sizes[r],
            .alignment = static_cast<uint64_t>(alignment),
            .error_out = &release_error,
          });
        }
        out_error = normalize_error(alloc_error, EMEL_ERR_BACKEND);  // GCOVR_EXCL_LINE
        return false;  // GCOVR_EXCL_LINE
      }
    }

    if (c.committed_chunk_counts[i] > 0) {
      for (int32_t j = 0; j < c.committed_chunk_counts[i]; ++j) {
        const int32_t idx = chunk_binding_index(i, j);
        if (c.committed_chunk_ids[idx] < 0 || c.committed_chunk_sizes[idx] == 0) {
          continue;
        }
        int32_t release_error = EMEL_OK;
        if (!chunk_allocator_sm->process_event(emel::buffer::chunk_allocator::event::release{
              .chunk = c.committed_chunk_ids[idx],
              .offset = c.committed_chunk_offsets[idx],
              .size = c.committed_chunk_sizes[idx],
              .alignment = static_cast<uint64_t>(alignment),
              .error_out = &release_error,
            }) ||
            release_error != EMEL_OK) {
          out_error = normalize_error(release_error, EMEL_ERR_BACKEND);  // GCOVR_EXCL_LINE
          return false;  // GCOVR_EXCL_LINE
        }
      }
    }

    const int32_t base = chunk_binding_index(i, 0);
    for (int32_t j = 0; j < k_max_chunks_per_buffer; ++j) {
      const int32_t idx = base + j;
      if (j < planned_count) {
        c.committed_chunk_ids[idx] = new_chunk_ids[j];
        c.committed_chunk_offsets[idx] = new_offsets[j];
        c.committed_chunk_sizes[idx] = new_sizes[j];
      } else {
        c.committed_chunk_ids[idx] = -1;
        c.committed_chunk_offsets[idx] = 0;
        c.committed_chunk_sizes[idx] = 0;
      }
    }
    c.committed_chunk_counts[i] = planned_count;
  }

  commit_sizes(c.committed_sizes, required, c.buffer_count);
  out_error = EMEL_OK;
  return true;
}

inline bool release_all_chunk_bindings(
    context & c, emel::buffer::chunk_allocator::sm * chunk_allocator_sm, int32_t & out_error) noexcept {
  for (int32_t i = 0; i < c.buffer_count; ++i) {
    if (c.committed_chunk_counts[i] <= 0) {
      continue;
    }
    const int32_t alignment = alignment_for_buffer(c, i);
    for (int32_t j = 0; j < c.committed_chunk_counts[i]; ++j) {
      const int32_t idx = chunk_binding_index(i, j);
      if (c.committed_chunk_ids[idx] < 0 || c.committed_chunk_sizes[idx] == 0) {
        continue;
      }
      int32_t release_error = EMEL_OK;
      if (!chunk_allocator_sm->process_event(emel::buffer::chunk_allocator::event::release{
            .chunk = c.committed_chunk_ids[idx],
            .offset = c.committed_chunk_offsets[idx],
            .size = c.committed_chunk_sizes[idx],
            .alignment = static_cast<uint64_t>(alignment),
            .error_out = &release_error,
          }) ||
          release_error != EMEL_OK) {
        out_error = normalize_error(release_error, EMEL_ERR_BACKEND);
        return false;
      }
      c.committed_chunk_ids[idx] = -1;
      c.committed_chunk_offsets[idx] = 0;
      c.committed_chunk_sizes[idx] = 0;
    }
    c.committed_chunk_counts[i] = 0;
  }

  int32_t reset_error = EMEL_OK;
  if (!chunk_allocator_sm->process_event(emel::buffer::chunk_allocator::event::reset{
        .error_out = &reset_error,
      }) ||
      reset_error != EMEL_OK) {
    out_error = normalize_error(reset_error, EMEL_ERR_BACKEND);  // GCOVR_EXCL_LINE
    return false;  // GCOVR_EXCL_LINE
  }

  out_error = EMEL_OK;
  return true;
}

}  // namespace detail

struct begin_initialize {
  void operator()(
      const event::initialize & ev,
      context & c,
      emel::buffer::chunk_allocator::sm & chunk_allocator) const noexcept {
    int32_t err = EMEL_OK;
    c.phase_error = EMEL_OK;
    c = {};
    c.buffer_count = ev.buffer_count;
    for (int32_t i = 0; i < k_max_buffers; ++i) {
      c.buffer_alignments[i] = k_default_alignment;
      c.buffer_max_sizes[i] = k_default_max_size;
    }
    for (int32_t i = 0; i < c.buffer_count && i < k_max_buffers; ++i) {
      const int32_t alignment =
        ev.buffer_alignments != nullptr ? ev.buffer_alignments[i] : k_default_alignment;
      const int32_t max_size =
        ev.buffer_max_sizes != nullptr ? ev.buffer_max_sizes[i] : k_default_max_size;
      c.buffer_alignments[i] = detail::sanitize_alignment(alignment);
      c.buffer_max_sizes[i] = detail::sanitize_max_size(max_size);
    }
    detail::reset_chunk_bindings(c);
    int32_t chunk_error = EMEL_OK;
    emel::buffer::chunk_allocator::sm * chunk_allocator_sm =
        ev.chunk_allocator_sm != nullptr ? ev.chunk_allocator_sm : &chunk_allocator;
    if (!detail::configure_chunk_allocator(chunk_allocator_sm, chunk_error)) {
      err = chunk_error;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    c.phase_error = err;
    c.step += 1;
  }
};

struct on_initialize_done {
  void operator()(const events::initialize_done & ev, context & c) const noexcept {
    c.init_epoch += 1;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    c.step += 1;
  }

  template <class Ev>
  void operator()(const Ev &, context & c) const noexcept {
    c.init_epoch += 1;
    c.step += 1;
  }
};

struct on_initialize_error {
  void operator()(const events::initialize_error & ev, context & c) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = detail::normalize_error(ev.err, EMEL_ERR_BACKEND);
    }
    c.step += 1;
  }

  template <class Ev>
  void operator()(const Ev &, context & c) const noexcept {
    c.step += 1;
  }
};

struct begin_reserve_n_size {
  void operator()(
      const event::reserve_n_size & ev,
      context & c,
      emel::buffer::planner::sm & planner) const noexcept {
    int32_t err = EMEL_OK;
    c.phase_error = EMEL_OK;
    emel::buffer::planner::sm * planner_sm =
        ev.buffer_planner_sm != nullptr ? ev.buffer_planner_sm : &planner;
    const emel::buffer::planner::strategy * strategy =
        ev.strategy != nullptr ? ev.strategy : &emel::buffer::planner::default_strategies::reserve_n_size;
    int32_t planner_err = EMEL_OK;
    if (!detail::run_planner(
          planner_sm, ev.graph, ev.node_buffer_ids, ev.leaf_buffer_ids, c.buffer_count, true,
          ev.sizes_out, ev.sizes_out_count, nullptr, 0, nullptr, 0, strategy, c, planner_err)) {
      err = planner_err;  // GCOVR_EXCL_LINE
    } else {
      detail::capture_buffer_map(c, ev.graph, ev.node_buffer_ids, ev.leaf_buffer_ids);
      if (!detail::capture_alloc_snapshot(c, ev.graph, ev.node_buffer_ids, ev.leaf_buffer_ids)) {
        err = EMEL_ERR_INVALID_ARGUMENT;  // GCOVR_EXCL_LINE
      }
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    c.phase_error = err;
    c.step += 1;
  }
};

struct on_reserve_n_size_done {
  void operator()(const events::reserve_n_size_done & ev, context & c) const noexcept {
    c.reserve_epoch += 1;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    c.step += 1;
  }

  template <class Ev>
  void operator()(const Ev &, context & c) const noexcept {
    c.reserve_epoch += 1;
    c.step += 1;
  }
};

struct on_reserve_n_size_error {
  void operator()(const events::reserve_n_size_error & ev, context & c) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = detail::normalize_error(ev.err, EMEL_ERR_BACKEND);
    }
    c.step += 1;
  }

  template <class Ev>
  void operator()(const Ev &, context & c) const noexcept {
    c.step += 1;
  }
};

struct begin_reserve_n {
  void operator()(
      const event::reserve_n & ev,
      context & c,
      emel::buffer::planner::sm & planner,
      emel::buffer::chunk_allocator::sm & chunk_allocator) const noexcept {
    int32_t err = EMEL_OK;
    c.phase_error = EMEL_OK;
    emel::buffer::planner::sm * planner_sm =
        ev.buffer_planner_sm != nullptr ? ev.buffer_planner_sm : &planner;
    emel::buffer::chunk_allocator::sm * chunk_allocator_sm =
        ev.chunk_allocator_sm != nullptr ? ev.chunk_allocator_sm : &chunk_allocator;
    const emel::buffer::planner::strategy * strategy =
        ev.strategy != nullptr ? ev.strategy : &emel::buffer::planner::default_strategies::reserve_n;
    std::array<int32_t, k_max_buffers> required = {};
    std::array<int32_t, k_max_buffers> chunk_counts = {};
    std::array<int32_t, k_max_chunk_bindings> chunk_sizes = {};
    int32_t planner_err = EMEL_OK;
    if (!detail::run_planner(
          planner_sm, ev.graph, ev.node_buffer_ids, ev.leaf_buffer_ids, c.buffer_count, false,
          required.data(), c.buffer_count,
          chunk_counts.data(), c.buffer_count,
          chunk_sizes.data(), static_cast<int32_t>(chunk_sizes.size()),
          strategy, c, planner_err)) {
      err = planner_err;
    } else {
      detail::capture_buffer_map(c, ev.graph, ev.node_buffer_ids, ev.leaf_buffer_ids);
      if (!detail::capture_alloc_snapshot(c, ev.graph, ev.node_buffer_ids, ev.leaf_buffer_ids)) {
        err = EMEL_ERR_INVALID_ARGUMENT;  // GCOVR_EXCL_LINE
      } else if (!detail::apply_required_sizes_to_chunks(
                   c, required, chunk_counts.data(), chunk_sizes.data(), chunk_allocator_sm, planner_err)) {
        err = planner_err;  // GCOVR_EXCL_LINE
      }
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    c.phase_error = err;
    c.step += 1;
  }
};

struct begin_reserve {
  void operator()(
      const event::reserve & ev,
      context & c,
      emel::buffer::planner::sm & planner,
      emel::buffer::chunk_allocator::sm & chunk_allocator) const noexcept {
    int32_t err = EMEL_OK;
    c.phase_error = EMEL_OK;
    emel::buffer::planner::sm * planner_sm =
        ev.buffer_planner_sm != nullptr ? ev.buffer_planner_sm : &planner;
    emel::buffer::chunk_allocator::sm * chunk_allocator_sm =
        ev.chunk_allocator_sm != nullptr ? ev.chunk_allocator_sm : &chunk_allocator;
    const emel::buffer::planner::strategy * strategy =
        ev.strategy != nullptr ? ev.strategy : &emel::buffer::planner::default_strategies::reserve;
    std::array<int32_t, k_max_buffers> required = {};
    std::array<int32_t, k_max_buffers> chunk_counts = {};
    std::array<int32_t, k_max_chunk_bindings> chunk_sizes = {};
    int32_t planner_err = EMEL_OK;
    if (!detail::run_planner(
          planner_sm, ev.graph, nullptr, nullptr, c.buffer_count, false, required.data(),
          c.buffer_count,
          chunk_counts.data(), c.buffer_count,
          chunk_sizes.data(), static_cast<int32_t>(chunk_sizes.size()),
          strategy, c, planner_err)) {
      err = planner_err;  // GCOVR_EXCL_LINE
    } else {
      detail::capture_buffer_map(c, ev.graph, nullptr, nullptr);
      if (!detail::capture_alloc_snapshot(c, ev.graph, nullptr, nullptr)) {
        err = EMEL_ERR_INVALID_ARGUMENT;  // GCOVR_EXCL_LINE
      } else if (!detail::apply_required_sizes_to_chunks(
                   c, required, chunk_counts.data(), chunk_sizes.data(), chunk_allocator_sm, planner_err)) {
        err = planner_err;  // GCOVR_EXCL_LINE
      }
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    c.phase_error = err;
    c.step += 1;
  }
};

struct on_reserve_done {
  void operator()(const events::reserve_done & ev, context & c) const noexcept {
    c.reserve_epoch += 1;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    c.step += 1;
  }

  template <class Ev>
  void operator()(const Ev &, context & c) const noexcept {
    c.reserve_epoch += 1;
    c.step += 1;
  }
};

struct on_reserve_error {
  void operator()(const events::reserve_error & ev, context & c) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = detail::normalize_error(ev.err, EMEL_ERR_BACKEND);
    }
    c.step += 1;
  }

  template <class Ev>
  void operator()(const Ev &, context & c) const noexcept {
    c.step += 1;
  }
};

struct begin_alloc_graph {
  void operator()(
      const event::alloc_graph & ev,
      context & c,
      emel::buffer::planner::sm & planner,
      emel::buffer::chunk_allocator::sm & chunk_allocator,
      emel::buffer::realloc_analyzer::sm & realloc_analyzer) const noexcept {
    int32_t err = EMEL_OK;
    c.phase_error = EMEL_OK;
    emel::buffer::planner::sm * planner_sm =
        ev.buffer_planner_sm != nullptr ? ev.buffer_planner_sm : &planner;
    emel::buffer::chunk_allocator::sm * chunk_allocator_sm =
        ev.chunk_allocator_sm != nullptr ? ev.chunk_allocator_sm : &chunk_allocator;
    emel::buffer::realloc_analyzer::sm * realloc_analyzer_sm =
        ev.buffer_realloc_analyzer_sm != nullptr ? ev.buffer_realloc_analyzer_sm : &realloc_analyzer;
    const emel::buffer::planner::strategy * strategy =
        ev.strategy != nullptr ? ev.strategy : &emel::buffer::planner::default_strategies::alloc_graph;
    int32_t analyze_err = EMEL_OK;
    bool needs_realloc = false;
    if (!detail::run_realloc_analyzer(
          realloc_analyzer_sm, ev.graph, c, needs_realloc, analyze_err)) {
      err = analyze_err;  // GCOVR_EXCL_LINE
    } else if (needs_realloc && c.buffer_count > 1) {
      err = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
    } else {
      const int32_t * node_ids = nullptr;
      const int32_t * leaf_ids = nullptr;
      if (!needs_realloc) {
        node_ids = c.last_node_buffer_ids.data();
        leaf_ids = c.last_leaf_buffer_ids.data();
      }

      std::array<int32_t, k_max_buffers> required = {};
      std::array<int32_t, k_max_buffers> chunk_counts = {};
      std::array<int32_t, k_max_chunk_bindings> chunk_sizes = {};
      int32_t planner_err = EMEL_OK;
      if (!detail::run_planner(
            planner_sm, ev.graph, node_ids, leaf_ids, c.buffer_count, true, required.data(),
            c.buffer_count,
            chunk_counts.data(), c.buffer_count,
            chunk_sizes.data(), static_cast<int32_t>(chunk_sizes.size()),
            strategy, c, planner_err)) {
        err = planner_err;  // GCOVR_EXCL_LINE
      } else {
        detail::capture_buffer_map(c, ev.graph, node_ids, leaf_ids);
        if (!detail::capture_alloc_snapshot(c, ev.graph, node_ids, leaf_ids)) {
          err = EMEL_ERR_INVALID_ARGUMENT;  // GCOVR_EXCL_LINE
        } else {
          const bool needs_growth = detail::requires_growth(c.committed_sizes, required, c.buffer_count);
          const bool missing_chunk_bindings =
              !detail::chunk_bindings_cover_required(c, required, c.buffer_count);
          if (needs_growth || missing_chunk_bindings) {
            if (c.buffer_count > 1) {
              err = EMEL_ERR_BACKEND;  // GCOVR_EXCL_LINE
            } else if (!detail::apply_required_sizes_to_chunks(
                         c, required, chunk_counts.data(), chunk_sizes.data(),
                         chunk_allocator_sm, planner_err)) {
              err = planner_err;  // GCOVR_EXCL_LINE
            }
          }
        }
      }
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    c.phase_error = err;
    c.step += 1;
  }
};

struct on_alloc_graph_done {
  void operator()(const events::alloc_graph_done & ev, context & c) const noexcept {
    c.alloc_epoch += 1;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    c.step += 1;
  }

  template <class Ev>
  void operator()(const Ev &, context & c) const noexcept {
    c.alloc_epoch += 1;
    c.step += 1;
  }
};

struct on_alloc_graph_error {
  void operator()(const events::alloc_graph_error & ev, context & c) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = detail::normalize_error(ev.err, EMEL_ERR_BACKEND);
    }
    c.step += 1;
  }

  template <class Ev>
  void operator()(const Ev &, context & c) const noexcept {
    c.step += 1;
  }
};

struct begin_release {
  void operator()(
      const event::release & ev,
      context & c,
      emel::buffer::chunk_allocator::sm & chunk_allocator) const noexcept {
    int32_t err = EMEL_OK;
    c.phase_error = EMEL_OK;
    emel::buffer::chunk_allocator::sm * chunk_allocator_sm =
        ev.chunk_allocator_sm != nullptr ? ev.chunk_allocator_sm : &chunk_allocator;
    if (!detail::release_all_chunk_bindings(c, chunk_allocator_sm, err)) {
      if (ev.error_out != nullptr) {
        *ev.error_out = err;
      }
      c.phase_error = err;
      c.step += 1;
      return;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    c.phase_error = EMEL_OK;
    c.step += 1;
  }
};

struct on_release_done {
  void operator()(const events::release_done & ev, context & c) const noexcept {
    const int32_t releases = c.release_epoch + 1;
    c = {};
    c.release_epoch = releases;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    c.step = 1;
  }

  template <class Ev>
  void operator()(const Ev &, context & c) const noexcept {
    const int32_t releases = c.release_epoch + 1;
    c = {};
    c.release_epoch = releases;
    c.step = 1;
  }
};

struct on_release_error {
  void operator()(const events::release_error & ev, context & c) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = detail::normalize_error(ev.err, EMEL_ERR_BACKEND);
    }
    c.step += 1;
  }

  template <class Ev>
  void operator()(const Ev &, context & c) const noexcept {
    c.step += 1;
  }
};

struct reject_invalid {
  template <class Event>
  void operator()(const Event & ev, context & c) const noexcept {
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      }
    }
    c.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    c.step += 1;
  }
};

struct on_unexpected {
  template <class Event>
  void operator()(const Event & ev, context & c) const noexcept {
    if constexpr (requires { ev.error_out; }) {
      if (ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
      }
    }
    c.phase_error = EMEL_ERR_INVALID_ARGUMENT;
    c.step += 1;
  }
};

inline constexpr begin_initialize begin_initialize{};
inline constexpr on_initialize_done on_initialize_done{};
inline constexpr on_initialize_error on_initialize_error{};
inline constexpr begin_reserve_n_size begin_reserve_n_size{};
inline constexpr on_reserve_n_size_done on_reserve_n_size_done{};
inline constexpr on_reserve_n_size_error on_reserve_n_size_error{};
inline constexpr begin_reserve_n begin_reserve_n{};
inline constexpr begin_reserve begin_reserve{};
inline constexpr on_reserve_done on_reserve_done{};
inline constexpr on_reserve_error on_reserve_error{};
inline constexpr begin_alloc_graph begin_alloc_graph{};
inline constexpr on_alloc_graph_done on_alloc_graph_done{};
inline constexpr on_alloc_graph_error on_alloc_graph_error{};
inline constexpr begin_release begin_release{};
inline constexpr on_release_done on_release_done{};
inline constexpr on_release_error on_release_error{};
inline constexpr reject_invalid reject_invalid{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::buffer::allocator::action
