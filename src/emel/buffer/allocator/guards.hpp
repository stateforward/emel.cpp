#pragma once

#include "emel/buffer/allocator/actions.hpp"
#include "emel/buffer/allocator/events.hpp"

namespace emel::buffer::allocator::guard {

inline constexpr auto phase_ok = [](const action::context & c) {
  return c.phase_error == EMEL_OK;
};

inline constexpr auto phase_failed = [](const action::context & c) {
  return c.phase_error != EMEL_OK;
};

inline constexpr auto always = [](const action::context &) {
  return true;
};

struct valid_initialize {
  bool operator()(const event::initialize & ev) const noexcept {
    if (ev.buffer_count <= 0 || ev.buffer_count > action::k_max_buffers) {
      return false;
    }
    if (ev.buffer_alignments != nullptr) {
      for (int32_t i = 0; i < ev.buffer_count; ++i) {
        if (!action::detail::valid_alignment(ev.buffer_alignments[i])) {
          return false;
        }
      }
    }
    if (ev.buffer_max_sizes != nullptr) {
      for (int32_t i = 0; i < ev.buffer_count; ++i) {
        if (ev.buffer_max_sizes[i] < 0) {
          return false;
        }
        if (ev.buffer_max_sizes[i] == 0 ||
            ev.buffer_max_sizes[i] == action::k_default_max_size) {
          continue;
        }
        const int32_t alignment =
          ev.buffer_alignments != nullptr ? ev.buffer_alignments[i] : action::k_default_alignment;
        if (!action::detail::valid_alignment(alignment)) {
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
    return true;
  }
};

struct can_reserve_n_size {
  bool operator()(const event::reserve_n_size & ev, const action::context & c) const noexcept {
    return c.buffer_count > 0 &&
           action::detail::valid_buffer_ids_for_graph(
             ev.graph, ev.node_buffer_ids, ev.leaf_buffer_ids, c.buffer_count) &&
           ev.sizes_out != nullptr && ev.sizes_out_count >= c.buffer_count;
  }
};

struct can_reserve_n_size_cached {
  bool operator()(const event::reserve_n_size & ev, const action::context & c) const noexcept {
    if (!can_reserve_n_size{}(ev, c)) {
      return false;
    }
    if (!c.has_required_sizes) {
      return false;
    }
    if (!action::detail::buffer_map_matches(ev.graph, ev.node_buffer_ids, ev.leaf_buffer_ids, c)) {
      return false;
    }
    if (action::detail::graph_needs_realloc(ev.graph, c)) {
      return false;
    }
    return true;
  }
};

struct can_reserve_n {
  bool operator()(const event::reserve_n & ev, const action::context & c) const noexcept {
    return c.buffer_count > 0 &&
           action::detail::valid_buffer_ids_for_graph(
             ev.graph, ev.node_buffer_ids, ev.leaf_buffer_ids, c.buffer_count);
  }
};

struct can_reserve_n_cached {
  bool operator()(const event::reserve_n & ev, const action::context & c) const noexcept {
    if (!can_reserve_n{}(ev, c)) {
      return false;
    }
    if (!c.has_required_sizes) {
      return false;
    }
    if (!action::detail::buffer_map_matches(ev.graph, ev.node_buffer_ids, ev.leaf_buffer_ids, c)) {
      return false;
    }
    if (action::detail::graph_needs_realloc(ev.graph, c)) {
      return false;
    }
    if (action::detail::requires_growth(c.committed_sizes, c.last_required_sizes, c.buffer_count)) {
      return false;
    }
    if (!action::detail::chunk_bindings_cover_required(
          c, c.last_required_sizes, c.buffer_count)) {
      return false;
    }
    return true;
  }
};

struct can_reserve {
  bool operator()(const event::reserve & ev, const action::context & c) const noexcept {
    return c.buffer_count > 0 &&
           action::detail::valid_buffer_ids_for_graph(ev.graph, nullptr, nullptr, c.buffer_count);
  }
};

struct can_reserve_cached {
  bool operator()(const event::reserve & ev, const action::context & c) const noexcept {
    if (!can_reserve{}(ev, c)) {
      return false;
    }
    if (!c.has_required_sizes) {
      return false;
    }
    if (!action::detail::buffer_map_matches(ev.graph, nullptr, nullptr, c)) {
      return false;
    }
    if (action::detail::graph_needs_realloc(ev.graph, c)) {
      return false;
    }
    if (action::detail::requires_growth(c.committed_sizes, c.last_required_sizes, c.buffer_count)) {
      return false;
    }
    if (!action::detail::chunk_bindings_cover_required(
          c, c.last_required_sizes, c.buffer_count)) {
      return false;
    }
    return true;
  }
};

struct can_alloc_graph {
  bool operator()(const event::alloc_graph & ev, const action::context & c) const noexcept {
    return c.buffer_count > 0 && action::detail::valid_graph_tensors(ev.graph);
  }
};

struct can_alloc_graph_cached {
  bool operator()(const event::alloc_graph & ev, const action::context & c) const noexcept {
    if (!can_alloc_graph{}(ev, c)) {
      return false;
    }
    if (!c.has_required_sizes) {
      return false;
    }
    if (action::detail::graph_needs_realloc(ev.graph, c)) {
      return false;
    }
    if (action::detail::requires_growth(c.committed_sizes, c.last_required_sizes, c.buffer_count)) {
      return false;
    }
    if (!action::detail::chunk_bindings_cover_required(
          c, c.last_required_sizes, c.buffer_count)) {
      return false;
    }
    return true;
  }
};

}  // namespace emel::buffer::allocator::guard
