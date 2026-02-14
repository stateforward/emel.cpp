#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include "emel/buffer/allocator/events.hpp"
#include "emel/buffer/planner/events.hpp"
#include "emel/buffer/planner/sm.hpp"
#include "emel/emel.h"

namespace emel::buffer::allocator::action {

inline constexpr int32_t k_max_buffers = 16;
inline constexpr int32_t k_max_graph_tensors = 1024;

struct tensor_alloc {
  int32_t tensor_id = -1;
  int32_t buffer_id = -1;
  int32_t size_max = 0;
};

struct node_alloc {
  tensor_alloc dst = {};
  std::array<tensor_alloc, event::k_max_sources> src = {};
};

struct leaf_alloc {
  tensor_alloc leaf = {};
};

struct context {
  int32_t buffer_count = 0;
  int32_t pending_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
  int32_t init_epoch = 0;
  int32_t reserve_epoch = 0;
  int32_t alloc_epoch = 0;
  int32_t release_epoch = 0;
  uint32_t step = 0;
  int32_t last_n_nodes = 0;
  int32_t last_n_leafs = 0;
  bool has_reserve_snapshot = false;
  std::array<int32_t, k_max_buffers> committed_sizes = {};
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

inline bool valid_graph_tensors(const event::graph_view & g) noexcept {
  if (g.n_nodes < 0 || g.n_leafs < 0) return false;
  if ((g.n_nodes > 0 && g.nodes == nullptr) || (g.n_leafs > 0 && g.leafs == nullptr)) return false;
  for (int32_t i = 0; i < g.n_nodes; ++i) {
    if (g.nodes[i].tensor_id < 0 || g.nodes[i].alloc_size < 0) return false;
  }
  for (int32_t i = 0; i < g.n_leafs; ++i) {
    if (g.leafs[i].tensor_id < 0 || g.leafs[i].alloc_size < 0) return false;
  }
  return true;
}

inline int32_t align_up_16(const int32_t value) noexcept {
  if (value <= 0) {
    return 0;
  }
  const int64_t aligned = (static_cast<int64_t>(value) + 15LL) & ~15LL;
  if (aligned > std::numeric_limits<int32_t>::max()) {
    return std::numeric_limits<int32_t>::max();
  }
  return static_cast<int32_t>(aligned);
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
    tensor_alloc & out, const event::tensor_desc & tensor, const int32_t buffer_id,
    const int32_t buffer_count) noexcept {
  out.tensor_id = tensor.tensor_id;
  if (tensor.has_external_data || tensor.is_view) {
    out.buffer_id = -1;
    out.size_max = 0;
    return true;
  }
  if (buffer_id < 0 || buffer_id >= buffer_count) {
    return false;
  }
  out.buffer_id = buffer_id;
  out.size_max = align_up_16(tensor.alloc_size);
  return true;
}

inline bool capture_alloc_snapshot(
    context & c, const event::graph_view & graph, const int32_t * node_buffer_ids,
    const int32_t * leaf_buffer_ids) noexcept {
  if (!valid_graph_tensors(graph)) {
    return false;
  }
  if (graph.n_nodes > static_cast<int32_t>(c.node_allocs.size()) ||
      graph.n_leafs > static_cast<int32_t>(c.leaf_allocs.size())) {
    return false;
  }

  for (int32_t i = 0; i < graph.n_nodes; ++i) {
    auto & dst = c.node_allocs[i];
    dst = {};
    const auto & node = graph.nodes[i];
    const int32_t node_buffer_id = get_buffer_id(node_buffer_ids, i);
    if (!build_tensor_alloc(dst.dst, node, node_buffer_id, c.buffer_count)) {
      return false;
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
      if (!build_tensor_alloc(dst.src[j], *src, src_buffer_id, c.buffer_count)) {
        return false;
      }
    }
  }

  for (int32_t i = 0; i < graph.n_leafs; ++i) {
    auto & dst = c.leaf_allocs[i];
    dst = {};
    const auto & leaf = graph.leafs[i];
    const int32_t leaf_buffer_id = get_buffer_id(leaf_buffer_ids, i);
    if (!build_tensor_alloc(dst.leaf, leaf, leaf_buffer_id, c.buffer_count)) {
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
    const event::tensor_desc & tensor, const tensor_alloc & alloc) noexcept {
  if (tensor.has_external_data || tensor.is_view) {
    return false;
  }
  if (alloc.buffer_id < 0) {
    return true;
  }
  return alloc.size_max < align_up_16(tensor.alloc_size);
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
    if (node_alloc.dst.tensor_id != node.tensor_id ||
        tensor_needs_realloc(node, node_alloc.dst)) {
      return true;
    }

    for (int32_t j = 0; j < event::k_max_sources; ++j) {
      const int32_t src_id = node.src_ids[j];
      const auto & src_alloc = node_alloc.src[j];
      if (src_id < 0) {
        if (src_alloc.tensor_id != -1) {
          return true;
        }
        continue;
      }
      if (src_alloc.tensor_id != src_id) {
        return true;
      }
      bool src_is_node = false;
      int32_t src_index = -1;
      const auto * src = find_tensor(graph, src_id, src_is_node, src_index);
      if (src == nullptr || src_index < 0 || tensor_needs_realloc(*src, src_alloc)) {
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
    const emel::buffer::planner::strategy * planner_strategy,
    int32_t & out_error) noexcept {
  if (planner == nullptr) {
    out_error = EMEL_ERR_INVALID_ARGUMENT;
    return false;
  }

  int32_t planner_error = EMEL_OK;
  const bool ok = planner->process_event(emel::buffer::planner::event::plan{
    .graph = graph,
    .node_buffer_ids = node_buffer_ids,
    .leaf_buffer_ids = leaf_buffer_ids,
    .buffer_count = buffer_count,
    .size_only = size_only,
    .sizes_out = sizes_out,
    .sizes_out_count = sizes_out_count,
    .error_out = &planner_error,
    .strategy = planner_strategy,
  });

  if (!ok || planner_error != EMEL_OK) {
    out_error = normalize_error(planner_error, EMEL_ERR_BACKEND);
    return false;
  }

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

}  // namespace detail

struct begin_initialize {
  void operator()(const event::initialize & ev, context & c) const noexcept {
    c.pending_error = (ev.buffer_count > 0 && ev.buffer_count <= k_max_buffers)
                        ? EMEL_OK
                        : EMEL_ERR_INVALID_ARGUMENT;
    if (c.pending_error == EMEL_OK) {
      c = {};
      c.buffer_count = ev.buffer_count;
    }
    c.step += 1;
  }
};

struct on_initialize_done {
  void operator()(const events::initialize_done &, context & c) const noexcept {
    c.init_epoch += 1;
    c.last_error = EMEL_OK;
    c.step += 1;
  }
};

struct on_initialize_error {
  void operator()(const events::initialize_error & ev, context & c) const noexcept {
    c.last_error = detail::normalize_error(ev.err, c.pending_error);
    c.step += 1;
  }
};

struct begin_reserve_n_size {
  void operator()(const event::reserve_n_size & ev, context & c) const noexcept {
    if (!detail::valid_graph_tensors(ev.graph)) {
      c.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      c.step += 1;
      return;
    }
    int32_t err = EMEL_OK;
    if (!detail::run_planner(
          ev.buffer_planner_sm, ev.graph, ev.node_buffer_ids, ev.leaf_buffer_ids, c.buffer_count, true,
          ev.sizes_out, ev.sizes_out_count, ev.strategy, err)) {
      c.pending_error = err;
      c.step += 1;
      return;
    }
    detail::capture_buffer_map(c, ev.graph, ev.node_buffer_ids, ev.leaf_buffer_ids);
    if (!detail::capture_alloc_snapshot(c, ev.graph, ev.node_buffer_ids, ev.leaf_buffer_ids)) {
      c.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      c.step += 1;
      return;
    }
    c.pending_error = EMEL_OK;
    c.step += 1;
  }
};

struct on_reserve_n_size_done {
  void operator()(const events::reserve_n_size_done &, context & c) const noexcept {
    c.reserve_epoch += 1;
    c.last_error = EMEL_OK;
    c.step += 1;
  }
};

struct on_reserve_n_size_error {
  void operator()(const events::reserve_n_size_error & ev, context & c) const noexcept {
    c.last_error = detail::normalize_error(ev.err, c.pending_error);
    c.step += 1;
  }
};

struct begin_reserve_n {
  void operator()(const event::reserve_n & ev, context & c) const noexcept {
    if (!detail::valid_graph_tensors(ev.graph)) {
      c.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      c.step += 1;
      return;
    }
    std::array<int32_t, k_max_buffers> required = {};
    int32_t err = EMEL_OK;
    if (!detail::run_planner(
          ev.buffer_planner_sm, ev.graph, ev.node_buffer_ids, ev.leaf_buffer_ids, c.buffer_count, false,
          required.data(), c.buffer_count, ev.strategy, err)) {
      c.pending_error = err;
      c.step += 1;
      return;
    }
    detail::capture_buffer_map(c, ev.graph, ev.node_buffer_ids, ev.leaf_buffer_ids);
    if (!detail::capture_alloc_snapshot(c, ev.graph, ev.node_buffer_ids, ev.leaf_buffer_ids)) {
      c.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      c.step += 1;
      return;
    }
    detail::commit_sizes(c.committed_sizes, required, c.buffer_count);
    c.pending_error = EMEL_OK;
    c.step += 1;
  }
};

struct begin_reserve {
  void operator()(const event::reserve & ev, context & c) const noexcept {
    if (!detail::valid_graph_tensors(ev.graph)) {
      c.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      c.step += 1;
      return;
    }
    std::array<int32_t, k_max_buffers> required = {};
    int32_t err = EMEL_OK;
    if (!detail::run_planner(
          ev.buffer_planner_sm, ev.graph, nullptr, nullptr, c.buffer_count, false, required.data(),
          c.buffer_count, ev.strategy, err)) {
      c.pending_error = err;
      c.step += 1;
      return;
    }
    detail::capture_buffer_map(c, ev.graph, nullptr, nullptr);
    if (!detail::capture_alloc_snapshot(c, ev.graph, nullptr, nullptr)) {
      c.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      c.step += 1;
      return;
    }
    detail::commit_sizes(c.committed_sizes, required, c.buffer_count);
    c.pending_error = EMEL_OK;
    c.step += 1;
  }
};

struct on_reserve_done {
  void operator()(const events::reserve_done &, context & c) const noexcept {
    c.reserve_epoch += 1;
    c.last_error = EMEL_OK;
    c.step += 1;
  }
};

struct on_reserve_error {
  void operator()(const events::reserve_error & ev, context & c) const noexcept {
    c.last_error = detail::normalize_error(ev.err, c.pending_error);
    c.step += 1;
  }
};

struct begin_alloc_graph {
  void operator()(const event::alloc_graph & ev, context & c) const noexcept {
    if (!detail::valid_graph_tensors(ev.graph)) {
      c.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      c.step += 1;
      return;
    }

    const bool needs_realloc = detail::graph_needs_realloc(ev.graph, c);

    if (needs_realloc && c.buffer_count > 1) {
      c.pending_error = EMEL_ERR_BACKEND;
      c.step += 1;
      return;
    }

    const int32_t * node_ids = nullptr;
    const int32_t * leaf_ids = nullptr;
    if (!needs_realloc) {
      node_ids = c.last_node_buffer_ids.data();
      leaf_ids = c.last_leaf_buffer_ids.data();
    }

    std::array<int32_t, k_max_buffers> required = {};
    int32_t err = EMEL_OK;
    if (!detail::run_planner(
          ev.buffer_planner_sm, ev.graph, node_ids, leaf_ids, c.buffer_count, true, required.data(),
          c.buffer_count, ev.strategy, err)) {
      c.pending_error = err;
      c.step += 1;
      return;
    }

    detail::capture_buffer_map(c, ev.graph, node_ids, leaf_ids);
    if (!detail::capture_alloc_snapshot(c, ev.graph, node_ids, leaf_ids)) {
      c.pending_error = EMEL_ERR_INVALID_ARGUMENT;
      c.step += 1;
      return;
    }

    if (detail::requires_growth(c.committed_sizes, required, c.buffer_count)) {
      if (c.buffer_count > 1) {
        c.pending_error = EMEL_ERR_BACKEND;
        c.step += 1;
        return;
      }
      detail::commit_sizes(c.committed_sizes, required, c.buffer_count);
    }

    c.pending_error = EMEL_OK;
    c.step += 1;
  }
};

struct on_alloc_graph_done {
  void operator()(const events::alloc_graph_done &, context & c) const noexcept {
    c.alloc_epoch += 1;
    c.last_error = EMEL_OK;
    c.step += 1;
  }
};

struct on_alloc_graph_error {
  void operator()(const events::alloc_graph_error & ev, context & c) const noexcept {
    c.last_error = detail::normalize_error(ev.err, c.pending_error);
    c.step += 1;
  }
};

struct begin_release {
  void operator()(const event::release &, context & c) const noexcept {
    c.pending_error = EMEL_OK;
    c.step += 1;
  }
};

struct on_release_done {
  void operator()(const events::release_done &, context & c) const noexcept {
    const int32_t releases = c.release_epoch + 1;
    c = {};
    c.release_epoch = releases;
    c.last_error = EMEL_OK;
    c.step = 1;
  }
};

struct on_release_error {
  void operator()(const events::release_error & ev, context & c) const noexcept {
    c.last_error = detail::normalize_error(ev.err, c.pending_error);
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

}  // namespace emel::buffer::allocator::action
