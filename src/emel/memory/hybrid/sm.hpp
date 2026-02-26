#pragma once

/*
design doc: docs/designs/memory/hybrid.design.md
 ---
 title: memory/hybrid architecture design
 status: draft
 ---
 
 # memory/hybrid architecture design
 
 this document defines the hybrid memory actor. it provides a unified lifecycle surface for models
 that utilize both KV Cache and Recurrent memory (e.g., Jamba, certain RWKV variants).
 
 ## role
 - act as a transparent orchestrator over both a `memory/kv::sm` and a `memory/recurrent::sm`.
 - expose a single `memory` API to the `generator`.
 - synchronize lifecycle events across both underlying memory architectures.
 
 ## architecture: the unified facade
 rather than building a complex, three-tiered "coordinator" state machine, the hybrid memory actor
 is a simple facade. the true complexity of PagedAttention and Recurrent State copying remains
 isolated inside their respective actors.
 
 when the `generator` issues a lifecycle event, the hybrid actor simply multi-casts it:
 
 1. **allocate:** dispatches to both `kv` and `recurrent`. if *either* fails (hits `out_of_memory`),
    the hybrid actor gracefully rolls back the successful one and returns `out_of_memory` to the
    generator.
 2. **branch:** dispatches to both. `kv` handles the blazing-fast DOD reference bump for zero-copy
    block sharing, while `recurrent` handles the physical state buffer copy into a new slot.
 3. **free:** dispatches to both, freeing the recurrent slot and dropping the KV block references via
    their respective DOD arrays.
 
 ## composition
 - owned by the `generator`.
 - owns one instance of `memory/kv::sm`.
 - owns one instance of `memory/recurrent::sm`.
 
 ## responsibilities
 - multi-cast sequence lifecycle events (`allocate`, `branch`, `free`) to both sub-actors.
 - deterministic error handling: if one subsystem fails an allocation, ensure the other is safely
   rolled back to maintain sequence parity between the two memory domains.
 - provide a unified `memory::any` view for the `graph/processor` to bind during execution.
*/

// benchmark: scaffold

#include <algorithm>
#include <cstdint>
#include <memory>

#include "emel/memory/hybrid/actions.hpp"
#include "emel/memory/hybrid/events.hpp"
#include "emel/memory/hybrid/guards.hpp"
#include "emel/memory/view.hpp"
#include "emel/sm.hpp"

namespace emel::memory::hybrid {

struct uninitialized {};
struct initializing {};
struct ready {};
struct allocating_sequence {};
struct allocating_slots {};
struct branching_sequence {};
struct freeing_sequence {};
struct rolling_back_slots {};
struct out_of_memory {};
struct errored {};
struct unexpected {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
        *sml::state<uninitialized> + sml::event<event::reserve> / action::begin_reserve =
            sml::state<initializing>,
        sml::state<ready> + sml::event<event::reserve> / action::begin_reserve =
            sml::state<initializing>,
        sml::state<unexpected> + sml::event<event::reserve> / action::begin_reserve =
            sml::state<initializing>,

        sml::state<initializing>[guard::phase_ok] = sml::state<ready>,
        sml::state<initializing>[guard::phase_out_of_memory] = sml::state<out_of_memory>,
        sml::state<initializing>[guard::phase_failed] = sml::state<errored>,

        sml::state<ready> + sml::event<event::allocate_sequence> / action::begin_allocate_sequence =
            sml::state<allocating_sequence>,
        sml::state<allocating_sequence>[guard::phase_ok] = sml::state<ready>,
        sml::state<allocating_sequence>[guard::phase_out_of_memory] = sml::state<out_of_memory>,
        sml::state<allocating_sequence>[guard::phase_failed] = sml::state<errored>,

        sml::state<ready> + sml::event<event::allocate_slots> / action::begin_allocate_slots =
            sml::state<allocating_slots>,
        sml::state<allocating_slots>[guard::phase_ok] = sml::state<ready>,
        sml::state<allocating_slots>[guard::phase_out_of_memory] = sml::state<out_of_memory>,
        sml::state<allocating_slots>[guard::phase_failed] = sml::state<errored>,

        sml::state<ready> + sml::event<event::branch_sequence> / action::begin_branch_sequence =
            sml::state<branching_sequence>,
        sml::state<branching_sequence>[guard::phase_ok] = sml::state<ready>,
        sml::state<branching_sequence>[guard::phase_out_of_memory] = sml::state<out_of_memory>,
        sml::state<branching_sequence>[guard::phase_failed] = sml::state<errored>,

        sml::state<ready> + sml::event<event::free_sequence> / action::begin_free_sequence =
            sml::state<freeing_sequence>,
        sml::state<freeing_sequence>[guard::phase_ok] = sml::state<ready>,
        sml::state<freeing_sequence>[guard::phase_failed] = sml::state<errored>,

        sml::state<ready> + sml::event<event::rollback_slots> / action::begin_rollback_slots =
            sml::state<rolling_back_slots>,
        sml::state<rolling_back_slots>[guard::phase_ok] = sml::state<ready>,
        sml::state<rolling_back_slots>[guard::phase_failed] = sml::state<errored>,

        sml::state<out_of_memory> / action::clear_out_of_memory = sml::state<ready>,
        sml::state<errored> / action::ensure_last_error = sml::state<ready>,

        sml::state<uninitialized> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<uninitialized>,
        sml::state<initializing> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<initializing>,
        sml::state<ready> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<ready>,
        sml::state<allocating_sequence> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<allocating_sequence>,
        sml::state<allocating_slots> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<allocating_slots>,
        sml::state<branching_sequence> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<branching_sequence>,
        sml::state<freeing_sequence> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<freeing_sequence>,
        sml::state<rolling_back_slots> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<rolling_back_slots>,
        sml::state<out_of_memory> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<out_of_memory>,
        sml::state<errored> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<errored>,
        sml::state<unexpected> + sml::event<event::capture_view> / action::capture_view{} =
            sml::state<unexpected>,

        sml::state<uninitialized> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<initializing> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<ready> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<allocating_sequence> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<allocating_slots> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<branching_sequence> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<freeing_sequence> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<rolling_back_slots> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<out_of_memory> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<errored> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>,
        sml::state<unexpected> + sml::unexpected_event<sml::_> / action::on_unexpected =
            sml::state<unexpected>);
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  // One-time heap storage keeps snapshot handoff frozen without per-dispatch allocation.
  sm() : base_type(context_), snapshot_(std::make_unique<view::snapshot>()) {}

  bool process_event(const event::reserve & ev) { return process_lifecycle_event(ev); }
  bool process_event(const event::allocate_sequence & ev) { return process_lifecycle_event(ev); }
  bool process_event(const event::allocate_slots & ev) { return process_lifecycle_event(ev); }
  bool process_event(const event::branch_sequence & ev) { return process_lifecycle_event(ev); }
  bool process_event(const event::free_sequence & ev) { return process_lifecycle_event(ev); }
  bool process_event(const event::rollback_slots & ev) { return process_lifecycle_event(ev); }
  bool process_event(const event::capture_view & ev) { return base_type::process_event(ev); }

  using base_type::process_event;

  int32_t last_error() const noexcept { return context_.last_error; }

  bool is_sequence_active(const int32_t seq_id) const noexcept {
    return context_.kv.is_sequence_active(seq_id) && context_.recurrent.is_sequence_active(seq_id);
  }

  int32_t sequence_length(const int32_t seq_id) const noexcept {
    return std::min(context_.kv.sequence_length(seq_id), context_.recurrent.sequence_length(seq_id));
  }

  int32_t lookup_kv_block(const int32_t seq_id, const int32_t pos) const noexcept {
    return context_.kv.lookup_kv_block(seq_id, pos);
  }

  int32_t lookup_recurrent_slot(const int32_t seq_id) const noexcept {
    return context_.recurrent.lookup_recurrent_slot(seq_id);
  }

  view::any view() noexcept {
    if (snapshot_ == nullptr) {
      return view::any{};
    }
    int32_t err = EMEL_OK;
    (void)this->base_type::process_event(event::capture_view{
      .snapshot_out = snapshot_.get(),
      .error_out = &err,
    });
    if (err != EMEL_OK) {
      return view::any{};
    }
    return view::any{.frozen = snapshot_.get()};
  }

 private:
  template <class ev>
  bool process_lifecycle_event(const ev & event) {
    const bool accepted = base_type::process_event(event);
    if constexpr (requires { event.error_out; }) {
      if (event.error_out != nullptr) {
        *event.error_out = context_.last_error;
      }
    }
    return accepted && context_.last_error == EMEL_OK;
  }

  std::unique_ptr<view::snapshot> snapshot_;
  action::context context_{};
};

}  // namespace emel::memory::hybrid
