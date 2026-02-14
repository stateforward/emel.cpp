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
- [ ] Full view/in-place lifetime edge parity.
- [x] Chunk/address placement internals (alignment, split/merge, reuse preference).
- [x] Explicit multi-buffer mapping mismatch validation and error coding.
- [ ] Overflow/limit hardening across size/count paths.
- [ ] Strategy contract tests for null/invalid/override tables.
- [ ] Ported allocator parity scenarios from tmp/llama.cpp/tests/test-alloc.cpp.
- [ ] Public C API allocator-path tests for exact EMEL_* status mapping.
- [ ] Performance guardrails for hot-path behavior and scaling.
*/

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

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
      *sml::state<uninitialized> + sml::event<event::initialize>[guard::valid_initialize{}] /
          action::begin_initialize = sml::state<initializing>,
      sml::state<initializing> + sml::event<events::initialize_done> / action::on_initialize_done =
          sml::state<ready>,
      sml::state<initializing> + sml::event<events::initialize_error> /
          action::on_initialize_error = sml::state<failed>,

      sml::state<ready> + sml::event<event::reserve_n_size>[guard::can_reserve_n_size{}] /
          action::begin_reserve_n_size = sml::state<reserving_n_size>,
      sml::state<allocated> + sml::event<event::reserve_n_size>[guard::can_reserve_n_size{}] /
          action::begin_reserve_n_size = sml::state<reserving_n_size>,
      sml::state<reserving_n_size> + sml::event<events::reserve_n_size_done> /
          action::on_reserve_n_size_done = sml::state<ready>,
      sml::state<reserving_n_size> + sml::event<events::reserve_n_size_error> /
          action::on_reserve_n_size_error = sml::state<failed>,

      sml::state<ready> + sml::event<event::reserve_n>[guard::can_reserve_n{}] /
          action::begin_reserve_n = sml::state<reserving>,
      sml::state<allocated> + sml::event<event::reserve_n>[guard::can_reserve_n{}] /
          action::begin_reserve_n = sml::state<reserving>,
      sml::state<ready> + sml::event<event::reserve>[guard::can_reserve{}] /
          action::begin_reserve = sml::state<reserving>,
      sml::state<allocated> + sml::event<event::reserve>[guard::can_reserve{}] /
          action::begin_reserve = sml::state<reserving>,
      sml::state<reserving> + sml::event<events::reserve_done> / action::on_reserve_done =
          sml::state<ready>,
      sml::state<reserving> + sml::event<events::reserve_error> / action::on_reserve_error =
          sml::state<failed>,

      sml::state<ready> + sml::event<event::alloc_graph>[guard::can_alloc_graph{}] /
          action::begin_alloc_graph = sml::state<allocating_graph>,
      sml::state<allocated> + sml::event<event::alloc_graph>[guard::can_alloc_graph{}] /
          action::begin_alloc_graph = sml::state<allocating_graph>,
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
      sml::state<releasing> + sml::event<events::release_done> / action::on_release_done =
          sml::state<uninitialized>,
      sml::state<releasing> + sml::event<events::release_error> / action::on_release_error =
          sml::state<failed>
    );
  }
};

struct sm : emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;

  bool process_event(const event::initialize & ev) {
    auto wired = ev;
    wired.chunk_allocator_sm = &buffer_chunk_allocator_sm_;
    if (!base_type::process_event(wired)) {
      return false;
    }
    return finalize_request<events::initialize_done, events::initialize_error>();
  }

  bool process_event(const event::reserve_n_size & ev) {
    auto wired = ev;
    wired.buffer_planner_sm = &buffer_planner_sm_;
    wired.chunk_allocator_sm = &buffer_chunk_allocator_sm_;
    if (wired.strategy == nullptr) {
      wired.strategy = &emel::buffer::planner::default_strategies::reserve_n_size;
    }
    if (!base_type::process_event(wired)) {
      return false;
    }
    return finalize_request<events::reserve_n_size_done, events::reserve_n_size_error>();
  }

  bool process_event(const event::reserve_n & ev) {
    auto wired = ev;
    wired.buffer_planner_sm = &buffer_planner_sm_;
    wired.chunk_allocator_sm = &buffer_chunk_allocator_sm_;
    if (wired.strategy == nullptr) {
      wired.strategy = &emel::buffer::planner::default_strategies::reserve_n;
    }
    if (!base_type::process_event(wired)) {
      return false;
    }
    return finalize_request<events::reserve_done, events::reserve_error>();
  }

  bool process_event(const event::reserve & ev) {
    auto wired = ev;
    wired.buffer_planner_sm = &buffer_planner_sm_;
    wired.chunk_allocator_sm = &buffer_chunk_allocator_sm_;
    if (wired.strategy == nullptr) {
      wired.strategy = &emel::buffer::planner::default_strategies::reserve;
    }
    if (!base_type::process_event(wired)) {
      return false;
    }
    return finalize_request<events::reserve_done, events::reserve_error>();
  }

  bool process_event(const event::alloc_graph & ev) {
    auto wired = ev;
    wired.buffer_planner_sm = &buffer_planner_sm_;
    wired.chunk_allocator_sm = &buffer_chunk_allocator_sm_;
    wired.buffer_realloc_analyzer_sm = &buffer_realloc_analyzer_sm_;
    if (wired.strategy == nullptr) {
      wired.strategy = &emel::buffer::planner::default_strategies::alloc_graph;
    }
    if (!base_type::process_event(wired)) {
      return false;
    }
    return finalize_request<events::alloc_graph_done, events::alloc_graph_error>();
  }

  bool process_event(const event::release & ev) {
    auto wired = ev;
    wired.chunk_allocator_sm = &buffer_chunk_allocator_sm_;
    if (!base_type::process_event(wired)) {
      return false;
    }
    return finalize_request<events::release_done, events::release_error>();
  }

  int32_t get_buffer_size(const int32_t buffer_id) const noexcept {
    if (buffer_id < 0 || buffer_id >= context_.buffer_count) {
      return 0;
    }
    return context_.committed_sizes[buffer_id];
  }

  int32_t last_error() const noexcept { return context_.last_error; }

  int32_t get_buffer_chunk_id(const int32_t buffer_id) const noexcept {
    if (buffer_id < 0 || buffer_id >= context_.buffer_count) {
      return -1;
    }
    return context_.committed_chunk_ids[buffer_id];
  }

  uint64_t get_buffer_chunk_offset(const int32_t buffer_id) const noexcept {
    if (buffer_id < 0 || buffer_id >= context_.buffer_count) {
      return 0;
    }
    return context_.committed_chunk_offsets[buffer_id];
  }

  uint64_t get_buffer_alloc_size(const int32_t buffer_id) const noexcept {
    if (buffer_id < 0 || buffer_id >= context_.buffer_count) {
      return 0;
    }
    return context_.committed_chunk_sizes[buffer_id];
  }

  int32_t chunk_count() const noexcept { return buffer_chunk_allocator_sm_.chunk_count(); }

  emel::buffer::planner::sm & planner_sm() noexcept { return buffer_planner_sm_; }

 private:
  template <class DoneEvent, class ErrorEvent>
  bool finalize_request() {
    if (context_.pending_error == EMEL_OK) {
      return base_type::process_event(DoneEvent{});
    }
    (void)base_type::process_event(ErrorEvent{
      .err = context_.pending_error,
    });
    return false;
  }

  action::context context_{};
  emel::buffer::planner::sm buffer_planner_sm_{};
  emel::buffer::chunk_allocator::sm buffer_chunk_allocator_sm_{};
  emel::buffer::realloc_analyzer::sm buffer_realloc_analyzer_sm_{};
};

}  // namespace emel::buffer::allocator
