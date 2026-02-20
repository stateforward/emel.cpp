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

// benchmark: ready
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
  sources (`buffer_id`, `size_max`, `alignment`).
- `alloc_graph` checks snapshot validity before deciding behavior:
  - snapshot missing, node/leaf count changed, source tensor id not found in the graph,
    or `size_max < required_size` for allocatable tensors => `needs_realloc = true`.
  - external-data and view tensors bypass size/buffer checks (matching gallocr semantics).
- If `needs_realloc = true`:
  - single-buffer mode: auto-reserve (planner recompute + committed-size growth).
  - multi-buffer mode: fail with backend error and require explicit reserve first.

Multi-buffer mismatch error contract
- Any `needs_realloc` condition in multi-buffer mode returns allocator-path backend error.
- This includes shape drift, missing sources, and growth beyond reserved assignment capacity.

Unexpected-event policy intent
- Unexpected event = known intent event type with no valid transition from current state.
- This is a sequencing contract violation (not an unknown type).
- Concrete examples:
  - reserve/reserve_n/reserve_n_size/alloc_graph before initialize.
  - initialize while already ready/allocated.
  - re-entrant callers dispatching intent events during in-flight phases.

Why unexpected events can still happen
- out-of-order integration calls,
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
  - [ ] Ported allocator parity scenarios from reference `test-alloc.cpp`.
  - [x] Public C API allocator-path tests for exact EMEL_* status mapping.
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
      *sml::state<uninitialized> +
              sml::event<event::initialize>[guard::valid_initialize{}] / action::begin_initialize =
          sml::state<initializing>,
      sml::state<uninitialized> + sml::event<event::initialize> / action::reject_invalid =
          sml::state<failed>,
      sml::state<initializing> [guard::phase_failed] / action::on_initialize_error =
          sml::state<failed>,
      sml::state<initializing> [guard::phase_ok] / action::on_initialize_done =
          sml::state<ready>,

      sml::state<ready> + sml::event<event::reserve_n_size>[guard::can_reserve_n_size_cached{}] /
          action::begin_reserve_n_size_cached = sml::state<reserving_n_size>,
      sml::state<ready> + sml::event<event::reserve_n_size>[guard::can_reserve_n_size{}] /
          action::begin_reserve_n_size = sml::state<reserving_n_size>,
      sml::state<ready> + sml::event<event::reserve_n_size> / action::reject_invalid =
          sml::state<failed>,
      sml::state<allocated> + sml::event<event::reserve_n_size>[guard::can_reserve_n_size_cached{}] /
          action::begin_reserve_n_size_cached = sml::state<reserving_n_size>,
      sml::state<allocated> + sml::event<event::reserve_n_size>[guard::can_reserve_n_size{}] /
          action::begin_reserve_n_size = sml::state<reserving_n_size>,
      sml::state<allocated> + sml::event<event::reserve_n_size> / action::reject_invalid =
          sml::state<failed>,
      sml::state<reserving_n_size> [guard::phase_failed] / action::on_reserve_n_size_error =
          sml::state<failed>,
      sml::state<reserving_n_size> [guard::phase_ok] / action::on_reserve_n_size_done =
          sml::state<ready>,

      sml::state<ready> + sml::event<event::reserve_n>[guard::can_reserve_n_cached{}] /
          action::begin_reserve_n_cached = sml::state<reserving>,
      sml::state<ready> + sml::event<event::reserve_n>[guard::can_reserve_n{}] /
          action::begin_reserve_n = sml::state<reserving>,
      sml::state<ready> + sml::event<event::reserve_n> / action::reject_invalid =
          sml::state<failed>,
      sml::state<allocated> + sml::event<event::reserve_n>[guard::can_reserve_n_cached{}] /
          action::begin_reserve_n_cached = sml::state<reserving>,
      sml::state<allocated> + sml::event<event::reserve_n>[guard::can_reserve_n{}] /
          action::begin_reserve_n = sml::state<reserving>,
      sml::state<allocated> + sml::event<event::reserve_n> / action::reject_invalid =
          sml::state<failed>,
      sml::state<ready> + sml::event<event::reserve>[guard::can_reserve_cached{}] /
          action::begin_reserve_cached = sml::state<reserving>,
      sml::state<ready> + sml::event<event::reserve>[guard::can_reserve{}] /
          action::begin_reserve = sml::state<reserving>,
      sml::state<ready> + sml::event<event::reserve> / action::reject_invalid =
          sml::state<failed>,
      sml::state<allocated> + sml::event<event::reserve>[guard::can_reserve_cached{}] /
          action::begin_reserve_cached = sml::state<reserving>,
      sml::state<allocated> + sml::event<event::reserve>[guard::can_reserve{}] /
          action::begin_reserve = sml::state<reserving>,
      sml::state<allocated> + sml::event<event::reserve> / action::reject_invalid =
          sml::state<failed>,
      sml::state<reserving> [guard::phase_failed] / action::on_reserve_error =
          sml::state<failed>,
      sml::state<reserving> [guard::phase_ok] / action::on_reserve_done =
          sml::state<ready>,

      sml::state<ready> + sml::event<event::alloc_graph>[guard::can_alloc_graph_cached{}] /
          action::begin_alloc_graph_cached = sml::state<allocating_graph>,
      sml::state<ready> + sml::event<event::alloc_graph>[guard::can_alloc_graph{}] /
          action::begin_alloc_graph = sml::state<allocating_graph>,
      sml::state<ready> + sml::event<event::alloc_graph> / action::reject_invalid =
          sml::state<failed>,
      sml::state<allocated> + sml::event<event::alloc_graph>[guard::can_alloc_graph_cached{}] /
          action::begin_alloc_graph_cached = sml::state<allocating_graph>,
      sml::state<allocated> + sml::event<event::alloc_graph>[guard::can_alloc_graph{}] /
          action::begin_alloc_graph = sml::state<allocating_graph>,
      sml::state<allocated> + sml::event<event::alloc_graph> / action::reject_invalid =
          sml::state<failed>,
      sml::state<allocating_graph> [guard::phase_failed] / action::on_alloc_graph_error =
          sml::state<failed>,
      sml::state<allocating_graph> [guard::phase_ok] / action::on_alloc_graph_done =
          sml::state<allocated>,

      sml::state<uninitialized> + sml::event<event::release> / action::begin_release_noop =
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
      sml::state<releasing> [guard::phase_failed] / action::on_release_error =
          sml::state<failed>,
      sml::state<releasing> [guard::phase_ok] / action::on_release_done =
          sml::state<uninitialized>,

      sml::state<uninitialized> + sml::event<event::reserve_n_size> / action::on_unexpected =
          sml::state<failed>,
      sml::state<uninitialized> + sml::event<event::reserve_n> / action::on_unexpected =
          sml::state<failed>,
      sml::state<uninitialized> + sml::event<event::reserve> / action::on_unexpected =
          sml::state<failed>,
      sml::state<uninitialized> + sml::event<event::alloc_graph> / action::on_unexpected =
          sml::state<failed>,

      sml::state<initializing> + sml::event<event::initialize> / action::on_unexpected =
          sml::state<failed>,
      sml::state<initializing> + sml::event<event::reserve_n_size> / action::on_unexpected =
          sml::state<failed>,
      sml::state<initializing> + sml::event<event::reserve_n> / action::on_unexpected =
          sml::state<failed>,
      sml::state<initializing> + sml::event<event::reserve> / action::on_unexpected =
          sml::state<failed>,
      sml::state<initializing> + sml::event<event::alloc_graph> / action::on_unexpected =
          sml::state<failed>,

      sml::state<ready> + sml::event<event::initialize> / action::on_unexpected =
          sml::state<failed>,

      sml::state<reserving_n_size> + sml::event<event::initialize> / action::on_unexpected =
          sml::state<failed>,
      sml::state<reserving_n_size> + sml::event<event::reserve_n_size> / action::on_unexpected =
          sml::state<failed>,
      sml::state<reserving_n_size> + sml::event<event::reserve_n> / action::on_unexpected =
          sml::state<failed>,
      sml::state<reserving_n_size> + sml::event<event::reserve> / action::on_unexpected =
          sml::state<failed>,
      sml::state<reserving_n_size> + sml::event<event::alloc_graph> / action::on_unexpected =
          sml::state<failed>,

      sml::state<reserving> + sml::event<event::initialize> / action::on_unexpected =
          sml::state<failed>,
      sml::state<reserving> + sml::event<event::reserve_n_size> / action::on_unexpected =
          sml::state<failed>,
      sml::state<reserving> + sml::event<event::reserve_n> / action::on_unexpected =
          sml::state<failed>,
      sml::state<reserving> + sml::event<event::reserve> / action::on_unexpected =
          sml::state<failed>,
      sml::state<reserving> + sml::event<event::alloc_graph> / action::on_unexpected =
          sml::state<failed>,

      sml::state<allocating_graph> + sml::event<event::initialize> / action::on_unexpected =
          sml::state<failed>,
      sml::state<allocating_graph> + sml::event<event::reserve_n_size> / action::on_unexpected =
          sml::state<failed>,
      sml::state<allocating_graph> + sml::event<event::reserve_n> / action::on_unexpected =
          sml::state<failed>,
      sml::state<allocating_graph> + sml::event<event::reserve> / action::on_unexpected =
          sml::state<failed>,
      sml::state<allocating_graph> + sml::event<event::alloc_graph> / action::on_unexpected =
          sml::state<failed>,

      sml::state<allocated> + sml::event<event::initialize> / action::on_unexpected =
          sml::state<failed>,

      sml::state<releasing> + sml::event<event::initialize> / action::on_unexpected =
          sml::state<failed>,
      sml::state<releasing> + sml::event<event::reserve_n_size> / action::on_unexpected =
          sml::state<failed>,
      sml::state<releasing> + sml::event<event::reserve_n> / action::on_unexpected =
          sml::state<failed>,
      sml::state<releasing> + sml::event<event::reserve> / action::on_unexpected =
          sml::state<failed>,
      sml::state<releasing> + sml::event<event::alloc_graph> / action::on_unexpected =
          sml::state<failed>,
      sml::state<releasing> + sml::event<event::release> / action::on_unexpected =
          sml::state<failed>,

      sml::state<failed> + sml::event<event::initialize> / action::on_unexpected =
          sml::state<failed>,
      sml::state<failed> + sml::event<event::reserve_n_size> / action::on_unexpected =
          sml::state<failed>,
      sml::state<failed> + sml::event<event::reserve_n> / action::on_unexpected =
          sml::state<failed>,
      sml::state<failed> + sml::event<event::reserve> / action::on_unexpected =
          sml::state<failed>,
      sml::state<failed> + sml::event<event::alloc_graph> / action::on_unexpected =
          sml::state<failed>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm()
      : base_type(
          context_,
          buffer_planner_sm_,
          buffer_chunk_allocator_sm_,
          buffer_realloc_analyzer_sm_) {}

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
