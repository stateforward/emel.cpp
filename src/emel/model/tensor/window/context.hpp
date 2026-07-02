#pragma once

#include <span>

#include "emel/io/mmap/sm.hpp"
#include "emel/io/staged_read/sm.hpp"
#include "emel/model/tensor/window/detail.hpp"

namespace emel::model::tensor::window::action {

// Persistent actor state: injected collaborators plus the residency window.
// io_staged provides one staged_read actor per window slot (slot-affinity is
// what preserves single-writer under concurrent slot loads: at most one
// in-flight load per slot, and slot s's pool task is the only entrant of
// io_staged[s]). io_pool is owner-constructed before bind so worker threads
// exist prior to any dispatch. Slot storage inside window is allocated once
// at bind (documented setup allocation) and freed here on destruction.
struct context {
  emel::io::mmap::sm *io_mmap = nullptr;
  std::span<emel::io::staged_read::sm> io_staged = {};
  detail::stream_io_pool *io_pool = nullptr;
  detail::window_state window = {};

  context() noexcept = default;
  context(const context &) = default;
  context &operator=(const context &) = default;

  ~context() noexcept { detail::reset_window(window); }
};

} // namespace emel::model::tensor::window::action
