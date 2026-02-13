#pragma once

#include "emel/buffer_allocator/actions.hpp"
#include "emel/buffer_allocator/events.hpp"

namespace emel::buffer_allocator::guard {

struct valid_initialize {
  bool operator()(const event::initialize & ev) const noexcept {
    return ev.buffer_count > 0 && ev.buffer_count <= action::k_max_buffers;
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

struct can_reserve_n {
  bool operator()(const event::reserve_n & ev, const action::context & c) const noexcept {
    return c.buffer_count > 0 &&
           action::detail::valid_buffer_ids_for_graph(
             ev.graph, ev.node_buffer_ids, ev.leaf_buffer_ids, c.buffer_count);
  }
};

struct can_reserve {
  bool operator()(const event::reserve & ev, const action::context & c) const noexcept {
    return c.buffer_count > 0 &&
           action::detail::valid_buffer_ids_for_graph(ev.graph, nullptr, nullptr, c.buffer_count);
  }
};

struct can_alloc_graph {
  bool operator()(const event::alloc_graph & ev, const action::context & c) const noexcept {
    return c.buffer_count > 0 && action::detail::valid_graph_tensors(ev.graph);
  }
};

}  // namespace emel::buffer_allocator::guard
