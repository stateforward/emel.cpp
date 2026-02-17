#pragma once

#include "emel/buffer/realloc_analyzer/actions.hpp"

namespace emel::buffer::realloc_analyzer::guard {

struct no_error {
  template <class Event>
  bool operator()(const Event & ev, const action::context &) const noexcept {
    if constexpr (requires { ev.err; }) {
      return ev.err == EMEL_OK;
    }
    return true;
  }
};

struct has_error {
  template <class Event>
  bool operator()(const Event & ev, const action::context &) const noexcept {
    if constexpr (requires { ev.err; }) {
      return ev.err != EMEL_OK;
    }
    return false;
  }
};

struct valid_analyze_request {
  bool operator()(const event::validate & ev, const action::context &) const noexcept {
    if (ev.request == nullptr) {
      return false;
    }
    if (!action::detail::valid_graph_tensors(ev.graph)) {
      return false;
    }
    if (ev.node_alloc_count < 0 || ev.leaf_alloc_count < 0) {
      return false;
    }
    if (ev.graph.n_nodes > 0 && ev.node_allocs == nullptr) {
      return false;
    }
    if (ev.graph.n_leafs > 0 && ev.leaf_allocs == nullptr) {
      return false;
    }
    return true;
  }
};

struct invalid_analyze_request {
  bool operator()(const event::validate & ev, const action::context & c) const noexcept {
    return !valid_analyze_request{}(ev, c);
  }
};

}  // namespace emel::buffer::realloc_analyzer::guard
