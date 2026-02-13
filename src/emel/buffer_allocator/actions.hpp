#pragma once

#include <array>
#include <cstdint>
#include <limits>

#include "emel/buffer_allocator/events.hpp"
#include "emel/buffer_planner/events.hpp"
#include "emel/buffer_planner/sm.hpp"
#include "emel/emel.h"

namespace emel::buffer_allocator::action {

inline constexpr int32_t k_max_buffers = 16;

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
  std::array<int32_t, k_max_buffers> committed_sizes = {};
  std::array<int32_t, 1024> last_node_buffer_ids = {};
  std::array<int32_t, 1024> last_leaf_buffer_ids = {};
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

inline int32_t get_buffer_id(const int32_t * ids, const int32_t index) noexcept {
  return ids == nullptr ? 0 : ids[index];
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
    emel::buffer_planner::sm * planner, const event::graph_view & graph,
    const int32_t * node_buffer_ids, const int32_t * leaf_buffer_ids, const int32_t buffer_count,
    const bool size_only, int32_t * sizes_out, const int32_t sizes_out_count,
    const emel::buffer_planner::strategy * planner_strategy,
    int32_t & out_error) noexcept {
  if (planner == nullptr) {
    out_error = EMEL_ERR_INVALID_ARGUMENT;
    return false;
  }

  int32_t planner_error = EMEL_OK;
  const bool ok = planner->process_event(emel::buffer_planner::event::plan{
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

    const bool shape_matches_last_reserve =
      ev.graph.n_nodes == c.last_n_nodes && ev.graph.n_leafs == c.last_n_leafs;
    if (c.buffer_count > 1 && !shape_matches_last_reserve) {
      c.pending_error = EMEL_ERR_BACKEND;
      c.step += 1;
      return;
    }

    std::array<int32_t, k_max_buffers> required = {};
    const int32_t * node_ids = nullptr;
    const int32_t * leaf_ids = nullptr;
    if (shape_matches_last_reserve) {
      node_ids = c.last_node_buffer_ids.data();
      leaf_ids = c.last_leaf_buffer_ids.data();
    }
    int32_t err = EMEL_OK;
    if (!detail::run_planner(
          ev.buffer_planner_sm, ev.graph, node_ids, leaf_ids, c.buffer_count, true, required.data(),
          c.buffer_count, ev.strategy, err)) {
      c.pending_error = err;
      c.step += 1;
      return;
    }

    if (!detail::requires_growth(c.committed_sizes, required, c.buffer_count)) {
      c.pending_error = EMEL_OK;
      c.step += 1;
      return;
    }

    if (c.buffer_count == 1) {
      detail::commit_sizes(c.committed_sizes, required, c.buffer_count);
      c.pending_error = EMEL_OK;
      c.step += 1;
      return;
    }

    c.pending_error = EMEL_ERR_BACKEND;
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

}  // namespace emel::buffer_allocator::action
