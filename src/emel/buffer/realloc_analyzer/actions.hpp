#pragma once

#include <cstdint>
#include <limits>

#include "emel/buffer/allocator/events.hpp"
#include "emel/buffer/realloc_analyzer/events.hpp"
#include "emel/emel.h"

namespace emel::buffer::realloc_analyzer::action {

struct context {
  event::graph_view graph = {};
  const event::node_alloc * node_allocs = nullptr;
  int32_t node_alloc_count = 0;
  const event::leaf_alloc * leaf_allocs = nullptr;
  int32_t leaf_alloc_count = 0;
  int32_t * needs_realloc_out = nullptr;
  int32_t * error_out = nullptr;
  bool needs_realloc = false;
  uint32_t step = 0;
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
  if (value <= 0) return 0;
  const int64_t aligned = (static_cast<int64_t>(value) + 15LL) & ~15LL;
  if (aligned > std::numeric_limits<int32_t>::max()) {
    return std::numeric_limits<int32_t>::max();
  }
  return static_cast<int32_t>(aligned);
}

inline const emel::buffer::allocator::event::tensor_desc * find_tensor(
    const event::graph_view & graph, const int32_t tensor_id) noexcept {
  for (int32_t i = 0; i < graph.n_nodes; ++i) {
    if (graph.nodes[i].tensor_id == tensor_id) return &graph.nodes[i];
  }
  for (int32_t i = 0; i < graph.n_leafs; ++i) {
    if (graph.leafs[i].tensor_id == tensor_id) return &graph.leafs[i];
  }
  return nullptr;
}

inline bool tensor_snapshot_valid(
    const emel::buffer::allocator::event::tensor_desc & tensor, const event::tensor_alloc & alloc) noexcept {
  int32_t tensor_size = 0;
  if (!tensor.has_external_data && !tensor.is_view) {
    if (alloc.buffer_id < 0) {
      return false;
    }
    tensor_size = align_up_16(tensor.alloc_size);
  }
  return alloc.size_max >= tensor_size;
}

inline bool graph_needs_realloc(const context & c) noexcept {
  if (c.node_alloc_count != c.graph.n_nodes) {
    return true;
  }
  if (c.leaf_alloc_count != c.graph.n_leafs) {
    return true;
  }

  for (int32_t i = 0; i < c.graph.n_nodes; ++i) {
    const auto & node = c.graph.nodes[i];
    const auto & node_alloc = c.node_allocs[i];

    if (!tensor_snapshot_valid(node, node_alloc.dst)) {
      return true;
    }

    for (int32_t j = 0; j < event::k_max_sources; ++j) {
      const int32_t src_id = node.src_ids[j];
      if (src_id < 0) {
        continue;
      }
      const auto * src = find_tensor(c.graph, src_id);
      if (src == nullptr) {
        return true;
      }
      if (!tensor_snapshot_valid(*src, node_alloc.src[j])) {
        return true;
      }
    }
  }

  return false;
}

}  // namespace detail

struct begin_analyze {
  void operator()(const event::analyze & ev, context & c) const noexcept {
    c = {};
    c.graph = ev.graph;
    c.node_allocs = ev.node_allocs;
    c.node_alloc_count = ev.node_alloc_count;
    c.leaf_allocs = ev.leaf_allocs;
    c.leaf_alloc_count = ev.leaf_alloc_count;
    c.needs_realloc_out = ev.needs_realloc_out;
    c.error_out = ev.error_out;
    c.needs_realloc = false;
    if (c.needs_realloc_out != nullptr) *c.needs_realloc_out = 0;
    if (c.error_out != nullptr) *c.error_out = EMEL_OK;
    c.step += 1;
  }
};

struct run_validate {
  void operator()(const event::validate & ev, context & c) const noexcept {
    int32_t err = EMEL_OK;
    if (!detail::valid_graph_tensors(c.graph)) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else if (c.node_alloc_count < 0 || c.leaf_alloc_count < 0) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else if (c.graph.n_nodes > 0 && c.node_allocs == nullptr) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    } else if (c.graph.n_leafs > 0 && c.leaf_allocs == nullptr) {
      err = EMEL_ERR_INVALID_ARGUMENT;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    c.step += 1;
  }
};

struct run_evaluate {
  void operator()(const event::evaluate & ev, context & c) const noexcept {
    c.needs_realloc = detail::graph_needs_realloc(c);
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    c.step += 1;
  }
};

struct run_publish {
  void operator()(const event::publish & ev, context & c) const noexcept {
    if (c.needs_realloc_out != nullptr) {
      *c.needs_realloc_out = c.needs_realloc ? 1 : 0;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    c.step += 1;
  }
};

struct begin_reset {
  void operator()(const event::reset & ev, context & c) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    c.step += 1;
  }
};

struct on_analyze_done {
  void operator()(const events::analyze_done & ev, context & c) const noexcept {
    c.needs_realloc = ev.needs_realloc != 0;
    if (c.needs_realloc_out != nullptr) {
      *c.needs_realloc_out = ev.needs_realloc;
    }
    if (c.error_out != nullptr) *c.error_out = EMEL_OK;
    c.step += 1;
  }
};

struct on_analyze_error {
  void operator()(const events::analyze_error & ev, context & c) const noexcept {
    if (c.error_out != nullptr) *c.error_out = detail::normalize_error(ev.err, EMEL_ERR_BACKEND);
    c.step += 1;
  }
};

struct on_reset_done {
  void operator()(const events::reset_done &, context & c) const noexcept {
    c = {};
    c.step = 1;
  }
};

struct on_reset_error {
  void operator()(const events::reset_error & ev, context & c) const noexcept {
    if (c.error_out != nullptr) *c.error_out = detail::normalize_error(ev.err, EMEL_ERR_BACKEND);
    c.step += 1;
  }
};

inline constexpr begin_analyze begin_analyze{};
inline constexpr run_validate run_validate{};
inline constexpr run_evaluate run_evaluate{};
inline constexpr run_publish run_publish{};
inline constexpr begin_reset begin_reset{};
inline constexpr on_analyze_done on_analyze_done{};
inline constexpr on_analyze_error on_analyze_error{};
inline constexpr on_reset_done on_reset_done{};
inline constexpr on_reset_error on_reset_error{};

}  // namespace emel::buffer::realloc_analyzer::action
