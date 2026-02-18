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
  bool needs_realloc = false;
  uint32_t step = 0;
  int32_t phase_error = EMEL_OK;
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

inline bool valid_analyze_payload(
    const event::graph_view & graph,
    const event::node_alloc * node_allocs,
    const int32_t node_alloc_count,
    const event::leaf_alloc * leaf_allocs,
    const int32_t leaf_alloc_count) noexcept {
  if (!valid_graph_tensors(graph)) {
    return false;
  }
  if (node_alloc_count < 0 || leaf_alloc_count < 0) {
    return false;
  }
  if (graph.n_nodes > 0 && node_allocs == nullptr) {
    return false;
  }
  if (graph.n_leafs > 0 && leaf_allocs == nullptr) {
    return false;
  }
  return true;
}

inline int32_t align_up(const int32_t value, const int32_t alignment) noexcept {
  if (value <= 0) return 0;
  const int32_t align = alignment > 0 ? alignment : 1;
  const int64_t aligned =
    (static_cast<int64_t>(value) + static_cast<int64_t>(align) - 1) &
    ~static_cast<int64_t>(align - 1);
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
    const int32_t alignment = alloc.alignment > 0 ? alloc.alignment : 16;
    tensor_size = align_up(tensor.alloc_size, alignment);
  }
  return alloc.size_max >= tensor_size;
}

inline bool graph_needs_realloc(
    const event::graph_view & graph,
    const event::node_alloc * node_allocs,
    const int32_t node_alloc_count,
    const event::leaf_alloc * leaf_allocs,
    const int32_t leaf_alloc_count) noexcept {
  (void)leaf_allocs;
  if (node_alloc_count != graph.n_nodes) {
    return true;
  }
  if (leaf_alloc_count != graph.n_leafs) {
    return true;
  }

  for (int32_t i = 0; i < graph.n_nodes; ++i) {
    const auto & node = graph.nodes[i];
    const auto & node_alloc = node_allocs[i];

    if (node_alloc.dst.tensor_id >= 0 && node_alloc.dst.tensor_id != node.tensor_id) {
      return true;
    }

    if (!tensor_snapshot_valid(node, node_alloc.dst)) {
      return true;
    }

    for (int32_t j = 0; j < event::k_max_sources; ++j) {
      const int32_t src_id = node.src_ids[j];
      const auto & src_alloc = node_alloc.src[j];
      if (src_id < 0) {
        if (src_alloc.tensor_id >= 0) {
          return true;
        }
        continue;
      }
      if (src_alloc.tensor_id >= 0 && src_alloc.tensor_id != src_id) {
        return true;
      }
      const auto * src = find_tensor(graph, src_id);
      if (src == nullptr) {
        return true;
      }
      if (!tensor_snapshot_valid(*src, src_alloc)) {
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
    c.needs_realloc = false;
    c.phase_error = EMEL_OK;
    if (ev.needs_realloc_out != nullptr) *ev.needs_realloc_out = 0;
    if (ev.error_out != nullptr) *ev.error_out = EMEL_OK;
    c.step += 1;
  }
};

struct run_validate {
  void operator()(const event::validate & ev, context & c) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    c.phase_error = detail::valid_analyze_payload(
        ev.graph, ev.node_allocs, ev.node_alloc_count, ev.leaf_allocs, ev.leaf_alloc_count)
      ? EMEL_OK
      : EMEL_ERR_INVALID_ARGUMENT;
    c.step += 1;
  }

  template <class Ev>
  void operator()(const Ev &, context & c) const noexcept {
    c.phase_error = detail::valid_analyze_payload(
        c.graph, c.node_allocs, c.node_alloc_count, c.leaf_allocs, c.leaf_alloc_count)
      ? EMEL_OK
      : EMEL_ERR_INVALID_ARGUMENT;
    c.step += 1;
  }
};

struct run_evaluate {
  void operator()(const event::evaluate & ev, context & c) const noexcept {
    c.needs_realloc = detail::graph_needs_realloc(
        ev.graph, ev.node_allocs, ev.node_alloc_count, ev.leaf_allocs, ev.leaf_alloc_count);
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    c.phase_error = EMEL_OK;
    c.step += 1;
  }

  template <class Ev>
  void operator()(const Ev &, context & c) const noexcept {
    c.needs_realloc = detail::graph_needs_realloc(
        c.graph, c.node_allocs, c.node_alloc_count, c.leaf_allocs, c.leaf_alloc_count);
    c.phase_error = EMEL_OK;
    c.step += 1;
  }
};

struct run_publish {
  void operator()(const event::publish & ev, context & c) const noexcept {
    if (ev.needs_realloc_out != nullptr) {
      *ev.needs_realloc_out = c.needs_realloc ? 1 : 0;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    c.phase_error = EMEL_OK;
    c.step += 1;
  }

  template <class Ev>
  void operator()(const Ev &, context & c) const noexcept {
    c.phase_error = EMEL_OK;
    c.step += 1;
  }
};

struct begin_reset {
  void operator()(const event::reset & ev, context & c) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
    c.phase_error = EMEL_OK;
    c.step += 1;
  }

  template <class Ev>
  void operator()(const Ev &, context & c) const noexcept {
    c.phase_error = EMEL_OK;
    c.step += 1;
  }
};

struct on_analyze_done {
  void operator()(const events::analyze_done & ev, context & c) const noexcept {
    c.needs_realloc = ev.needs_realloc != 0;
    if (ev.needs_realloc_out != nullptr) {
      *ev.needs_realloc_out = ev.needs_realloc;
    }
    if (ev.error_out != nullptr) *ev.error_out = EMEL_OK;
    c.step += 1;
  }

  template <class Ev>
  void operator()(const Ev &, context & c) const noexcept {
    c.step += 1;
  }
};

struct on_analyze_error {
  void operator()(const events::analyze_error & ev, context & c) const noexcept {
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

struct on_reset_done {
  void operator()(const events::reset_done &, context & c) const noexcept {
    c = {};
    c.step = 1;
  }

  template <class Ev>
  void operator()(const Ev &, context & c) const noexcept {
    c = {};
    c.step = 1;
  }
};

struct on_reset_error {
  void operator()(const events::reset_error & ev, context & c) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = detail::normalize_error(ev.err, EMEL_ERR_BACKEND);
    }
    c.phase_error = detail::normalize_error(ev.err, EMEL_ERR_BACKEND);
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
    if constexpr (requires { ev.needs_realloc_out; }) {
      if (ev.needs_realloc_out != nullptr) {
        *ev.needs_realloc_out = 0;
      }
    }
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

inline constexpr begin_analyze begin_analyze{};
inline constexpr run_validate run_validate{};
inline constexpr run_evaluate run_evaluate{};
inline constexpr run_publish run_publish{};
inline constexpr begin_reset begin_reset{};
inline constexpr on_analyze_done on_analyze_done{};
inline constexpr on_analyze_error on_analyze_error{};
inline constexpr on_reset_done on_reset_done{};
inline constexpr on_reset_error on_reset_error{};
inline constexpr reject_invalid reject_invalid{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::buffer::realloc_analyzer::action
