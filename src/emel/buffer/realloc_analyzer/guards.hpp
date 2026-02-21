#pragma once

#include "emel/buffer/realloc_analyzer/actions.hpp"

namespace emel::buffer::realloc_analyzer::guard {

inline constexpr auto phase_ok = [](const action::context & c) {
  return c.phase_error == EMEL_OK;
};

inline constexpr auto phase_failed = [](const action::context & c) {
  return c.phase_error != EMEL_OK;
};

inline constexpr auto always = [](const action::context &) {
  return true;
};

struct no_error {
  template <class event>
  bool operator()(const event & ev, const action::context &) const noexcept {
    if constexpr (requires { ev.err; }) {
      return ev.err == EMEL_OK;
    }
    return true;
  }
};

struct has_error {
  template <class event>
  bool operator()(const event & ev, const action::context &) const noexcept {
    if constexpr (requires { ev.err; }) {
      return ev.err != EMEL_OK;
    }
    return false;
  }
};

struct can_analyze {
  bool operator()(const event::analyze & ev) const noexcept {
    return action::detail::valid_analyze_payload(
        ev.graph, ev.node_allocs, ev.node_alloc_count, ev.leaf_allocs, ev.leaf_alloc_count);
  }
};

struct valid_analyze_request {
  bool operator()(const event::validate & ev, const action::context &) const noexcept {
    if (ev.request == nullptr) {
      return false;
    }
    return action::detail::valid_analyze_payload(
        ev.graph, ev.node_allocs, ev.node_alloc_count, ev.leaf_allocs, ev.leaf_alloc_count);
  }
};

struct invalid_analyze_request {
  bool operator()(const event::validate & ev, const action::context & c) const noexcept {
    return !valid_analyze_request{}(ev, c);
  }
};

}  // namespace emel::buffer::realloc_analyzer::guard
