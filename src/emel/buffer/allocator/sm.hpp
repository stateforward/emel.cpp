#pragma once

#include <cstdint>

#include "emel/buffer/allocator/actions.hpp"
#include "emel/buffer/allocator/events.hpp"
#include "emel/buffer/chunk_allocator/sm.hpp"
#include "emel/buffer/allocator/guards.hpp"
#include "emel/buffer/planner/sm.hpp"
#include "emel/buffer/realloc_analyzer/sm.hpp"
#include "emel/sm.hpp"

namespace emel::buffer::allocator {

using Process = boost::sml::back::process<
  events::initialize_done,
  events::initialize_error,
  events::reserve_n_size_done,
  events::reserve_n_size_error,
  events::reserve_done,
  events::reserve_error,
  events::alloc_graph_done,
  events::alloc_graph_error,
  events::release_done,
  events::release_error>;

/*
Buffer allocator architecture notes (single source of truth)

Scope
- Component boundary: buffer_allocator + buffer_planner
- Goal: deterministic allocator orchestration with gallocr-style planning semantics

Design overview
- `buffer_allocator::sm` owns allocator runtime context and lifecycle orchestration states.
- `buffer_planner::sm` is injected as a child planner machine and receives planning requests by event.
- API-facing request flow is run-to-completion:
  initialize -> reserve/reserve_n/reserve_n_size -> alloc_graph -> release.
- Planner strategy is per-invocation input (event payload). If omitted, allocator wires
  `buffer_planner::default_strategies::{reserve_n_size,reserve_n,reserve,alloc_graph}`.
- `default_strategies::*` currently alias to `gallocr_parity` by design to preserve one
  canonical behavior while keeping extension points explicit.

`needs_realloc` parity contract
- Reserve operations persist allocation assignment snapshots per node destination and per-node
  sources (`tensor_id`, `buffer_id`, `size_max`).
- `alloc_graph` checks snapshot validity before deciding behavior:
  - snapshot missing, node/leaf count changed, tensor identity changed, source identity changed,
    or `size_max < required_size` for allocatable tensors => `needs_realloc = true`.
  - external-data and view tensors bypass size/buffer checks (matching gallocr semantics).
- If `needs_realloc = true`:
  - single-buffer mode: auto-reserve (planner recompute + committed-size growth).
  - multi-buffer mode: fail with backend error and require explicit reserve first.

Multi-buffer mismatch error contract
- Any `needs_realloc` condition in multi-buffer mode returns allocator-path backend error.
- This includes shape drift, source wiring drift, and growth beyond reserved assignment capacity.

Unexpected-event policy intent
- Unexpected event = known event type with no valid transition from current state.
- This is a sequencing contract violation (not an unknown type).
- Concrete examples:
  - reserve/reserve_n/reserve_n_size/alloc_graph before initialize.
  - initialize while already ready/allocated.
  - reserve_done/alloc_graph_done injected when not in reserving/allocating_graph.
  - plan or phase *_done/_error delivered while planner is in the wrong phase.

Why unexpected events can still happen
- out-of-order internal dispatch from integration bugs,
- re-entrant/concurrent callers violating sequencing,
- future state/event additions without explicit handling.

Production completion checklist (open)
  - [ ] Explicit handling path for all unexpected event/state combinations.
  - [x] Reserve-to-alloc assignment validity persistence and verification parity
        (`needs_realloc`-style checks).
  - [x] Full view/in-place lifetime edge parity.
  - [x] Chunk/address placement internals (alignment, split/merge, reuse preference).
  - [x] Explicit multi-buffer mapping mismatch validation and error coding.
  - [x] Overflow/limit hardening across size/count paths.
  - [ ] Strategy contract tests for null/invalid/override tables.
  - [ ] Ported allocator parity scenarios from tmp/llama.cpp/tests/test-alloc.cpp.
  - [x] Public C API allocator-path tests for exact EMEL_* status mapping.
  - [ ] Performance guardrails for hot-path behavior and scaling.
*/

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    using process_t = Process;

    struct uninitialized {};
    struct initializing {};
    struct ready {};
    struct reserving_n_size {};
    struct reserving {};
    struct allocating_graph {};
    struct allocated {};
    struct releasing {};
    struct failed {};

    return sml::make_transition_table(
      *sml::state<uninitialized> + sml::event<event::initialize> /
          action::begin_initialize = sml::state<initializing>,
      sml::state<initializing> + sml::on_entry<event::initialize> /
          [](const event::initialize & ev, action::context &, process_t & process) noexcept {
            const int32_t err = ev.error_out != nullptr ? *ev.error_out : EMEL_OK;
            if (err == EMEL_OK) {
              process(events::initialize_done{.error_out = ev.error_out});
            } else {
              process(events::initialize_error{.err = err, .error_out = ev.error_out});
            }
          },
      sml::state<initializing> + sml::event<events::initialize_done> / action::on_initialize_done =
          sml::state<ready>,
      sml::state<initializing> + sml::event<events::initialize_error> /
          action::on_initialize_error = sml::state<failed>,

      sml::state<ready> + sml::event<event::reserve_n_size> /
          action::begin_reserve_n_size = sml::state<reserving_n_size>,
      sml::state<allocated> + sml::event<event::reserve_n_size> /
          action::begin_reserve_n_size = sml::state<reserving_n_size>,
      sml::state<reserving_n_size> + sml::on_entry<event::reserve_n_size> /
          [](const event::reserve_n_size & ev, action::context &, process_t & process) noexcept {
            const int32_t err = ev.error_out != nullptr ? *ev.error_out : EMEL_OK;
            if (err == EMEL_OK) {
              process(events::reserve_n_size_done{.error_out = ev.error_out});
            } else {
              process(events::reserve_n_size_error{.err = err, .error_out = ev.error_out});
            }
          },
      sml::state<reserving_n_size> + sml::event<events::reserve_n_size_done> /
          action::on_reserve_n_size_done = sml::state<ready>,
      sml::state<reserving_n_size> + sml::event<events::reserve_n_size_error> /
          action::on_reserve_n_size_error = sml::state<failed>,

      sml::state<ready> + sml::event<event::reserve_n> /
          action::begin_reserve_n = sml::state<reserving>,
      sml::state<allocated> + sml::event<event::reserve_n> /
          action::begin_reserve_n = sml::state<reserving>,
      sml::state<ready> + sml::event<event::reserve> /
          action::begin_reserve = sml::state<reserving>,
      sml::state<allocated> + sml::event<event::reserve> /
          action::begin_reserve = sml::state<reserving>,
      sml::state<reserving> + sml::on_entry<event::reserve_n> /
          [](const event::reserve_n & ev, action::context &, process_t & process) noexcept {
            const int32_t err = ev.error_out != nullptr ? *ev.error_out : EMEL_OK;
            if (err == EMEL_OK) {
              process(events::reserve_done{.error_out = ev.error_out});
            } else {
              process(events::reserve_error{.err = err, .error_out = ev.error_out});
            }
          },
      sml::state<reserving> + sml::on_entry<event::reserve> /
          [](const event::reserve & ev, action::context &, process_t & process) noexcept {
            const int32_t err = ev.error_out != nullptr ? *ev.error_out : EMEL_OK;
            if (err == EMEL_OK) {
              process(events::reserve_done{.error_out = ev.error_out});
            } else {
              process(events::reserve_error{.err = err, .error_out = ev.error_out});
            }
          },
      sml::state<reserving> + sml::event<events::reserve_done> / action::on_reserve_done =
          sml::state<ready>,
      sml::state<reserving> + sml::event<events::reserve_error> / action::on_reserve_error =
          sml::state<failed>,

      sml::state<ready> + sml::event<event::alloc_graph> /
          action::begin_alloc_graph = sml::state<allocating_graph>,
      sml::state<allocated> + sml::event<event::alloc_graph> /
          action::begin_alloc_graph = sml::state<allocating_graph>,
      sml::state<allocating_graph> + sml::on_entry<event::alloc_graph> /
          [](const event::alloc_graph & ev, action::context &, process_t & process) noexcept {
            const int32_t err = ev.error_out != nullptr ? *ev.error_out : EMEL_OK;
            if (err == EMEL_OK) {
              process(events::alloc_graph_done{.error_out = ev.error_out});
            } else {
              process(events::alloc_graph_error{.err = err, .error_out = ev.error_out});
            }
          },
      sml::state<allocating_graph> + sml::event<events::alloc_graph_done> /
          action::on_alloc_graph_done = sml::state<allocated>,
      sml::state<allocating_graph> + sml::event<events::alloc_graph_error> /
          action::on_alloc_graph_error = sml::state<failed>,

      sml::state<uninitialized> + sml::event<event::release> / action::begin_release =
          sml::state<releasing>,
      sml::state<initializing> + sml::event<event::release> / action::begin_release =
          sml::state<releasing>,
      sml::state<ready> + sml::event<event::release> / action::begin_release =
          sml::state<releasing>,
      sml::state<reserving_n_size> + sml::event<event::release> / action::begin_release =
          sml::state<releasing>,
      sml::state<reserving> + sml::event<event::release> / action::begin_release =
          sml::state<releasing>,
      sml::state<allocating_graph> + sml::event<event::release> / action::begin_release =
          sml::state<releasing>,
      sml::state<allocated> + sml::event<event::release> / action::begin_release =
          sml::state<releasing>,
      sml::state<failed> + sml::event<event::release> / action::begin_release =
          sml::state<releasing>,
      sml::state<releasing> + sml::on_entry<event::release> /
          [](const event::release & ev, action::context &, process_t & process) noexcept {
            const int32_t err = ev.error_out != nullptr ? *ev.error_out : EMEL_OK;
            if (err == EMEL_OK) {
              process(events::release_done{.error_out = ev.error_out});
            } else {
              process(events::release_error{.err = err, .error_out = ev.error_out});
            }
          },
      sml::state<releasing> + sml::event<events::release_done> / action::on_release_done =
          sml::state<uninitialized>,
      sml::state<releasing> + sml::event<events::release_error> / action::on_release_error =
          sml::state<failed>,

      sml::state<uninitialized> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<initializing> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<ready> + sml::event<sml::_> / action::on_unexpected = sml::state<failed>,
      sml::state<reserving_n_size> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<reserving> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<allocating_graph> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<allocated> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<releasing> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<failed> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>
    );
  }
};

struct sm : private emel::detail::process_support<sm, Process>, public emel::sm<model, Process> {
  using base_type = emel::sm<model, Process>;

  sm()
      : emel::detail::process_support<sm, Process>(this),
        base_type(
          context_,
          buffer_planner_sm_,
          buffer_chunk_allocator_sm_,
          buffer_realloc_analyzer_sm_,
          this->process_) {}

  using base_type::process_event;

  int32_t get_buffer_size(const int32_t buffer_id) const noexcept {
    if (buffer_id < 0 || buffer_id >= context_.buffer_count) {
      return 0;
    }
    return context_.committed_sizes[buffer_id];
  }

  int32_t get_buffer_chunk_id(const int32_t buffer_id) const noexcept {
    if (buffer_id < 0 || buffer_id >= context_.buffer_count) {
      return -1;
    }
    if (context_.committed_chunk_counts[buffer_id] <= 0) {
      return -1;
    }
    return context_.committed_chunk_ids[action::detail::chunk_binding_index(buffer_id, 0)];
  }

  uint64_t get_buffer_chunk_offset(const int32_t buffer_id) const noexcept {
    if (buffer_id < 0 || buffer_id >= context_.buffer_count) {
      return 0;
    }
    if (context_.committed_chunk_counts[buffer_id] <= 0) {
      return 0;
    }
    return context_.committed_chunk_offsets[action::detail::chunk_binding_index(buffer_id, 0)];
  }

  uint64_t get_buffer_alloc_size(const int32_t buffer_id) const noexcept {
    if (buffer_id < 0 || buffer_id >= context_.buffer_count) {
      return 0;
    }
    if (context_.committed_chunk_counts[buffer_id] <= 0) {
      return 0;
    }
    uint64_t total = 0;
    const int32_t count = context_.committed_chunk_counts[buffer_id];
    for (int32_t i = 0; i < count; ++i) {
      total += context_.committed_chunk_sizes[action::detail::chunk_binding_index(buffer_id, i)];
    }
    return total;
  }

  int32_t chunk_count() const noexcept { return buffer_chunk_allocator_sm_.chunk_count(); }

  emel::buffer::planner::sm & planner_sm() noexcept { return buffer_planner_sm_; }

 private:
  action::context context_{};
  emel::buffer::planner::sm buffer_planner_sm_{};
  emel::buffer::chunk_allocator::sm buffer_chunk_allocator_sm_{};
  emel::buffer::realloc_analyzer::sm buffer_realloc_analyzer_sm_{};
};

}  // namespace emel::buffer::allocator
