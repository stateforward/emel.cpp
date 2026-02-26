#pragma once

/*
design doc: docs/designs/memory/recurrent.design.md
 ---
 title: memory/recurrent architecture design
 status: draft
 ---
 
 # memory/recurrent architecture design
 
 this document defines the recurrent memory actor. it provides storage and lifecycle management for
 models that maintain a fixed-size state vector per sequence (e.g., RWKV, Mamba, Jamba).
 
 ## role
 - manage recurrent state buffers (which are fixed-size per sequence, unlike kv caches that grow per
   token).
 - orchestrate sequence lifecycle events (allocate, free, branch) received from the `generator`.
 - enforce capacity boundaries safely via SML transitions.
 
 ## architecture shift: DOD with `sml::sm_pool`
 unlike the kv cache which partitions memory into paged blocks, recurrent memory is much simpler.
 each sequence requires a single, contiguous, fixed-size state buffer.
 
 to maximize cache locality and execution speed while preserving Actor Model safety, the recurrent states
 are managed using `boost::sml::sm_pool<recurrent_slot_sm>`.
 - **cache locality:** `sm_pool` ensures all slot actors are contiguous in memory.
 - **declarative safety:** each slot is a true SML actor tracking its lifecycle (`empty` -> `filled`).
 - **batch dispatch:** the `memory/recurrent` manager coordinates slots using the high-performance
   `process_event_batch` API, ensuring allocation and freeing remain strictly safe with only ~11% overhead
   compared to raw procedural arrays.
 
 1. **fixed allocation:** allocating a sequence simply finds an inactive slot in the array and marks
    it active.
 2. **explicit branching:** because recurrent state is mutated at every step (unlike historical kv
    keys/values which are read-only), sequence branching requires a physical memory copy. when sequence
    B branches from sequence A, the recurrent manager finds a new inactive slot for B, and physically
    copies the data from A's slot into B's newly allocated slot.
 
 ## composition
 - owned by the `generator` (or `memory/hybrid` if mixed).
 - owns the physical backing tensors for recurrent states.
 
 ## state model
 
 ```text
 uninitialized ──► initializing ──► ready
                                      │
 ready ──► allocating_sequence ──► (ready | out_of_memory)
   ▲                                  │
   └──────────────────────────────────┘
 
 ready ──► branching_sequence ──► (ready | out_of_memory)
 ready ──► freeing_sequence   ──► ready
 ```
 
 - `uninitialized` — awaiting initial setup.
 - `initializing` — partitioning the backing tensor into fixed-size sequence slots.
 - `ready` — waiting for lifecycle commands.
 - `allocating_sequence` — finding an empty state slot. if capacity is reached, gracefully reject
   via SML guard (`out_of_memory`).
 - `branching_sequence` — allocating a new slot and copying the parent's state data into it.
 - `freeing_sequence` — clearing a sequence ID mapping and marking the slot inactive.
 - unexpected events route to `unexpected`.
 
 ## responsibilities
 - **initialization:** partition the backing tensor into fixed-size sequence slots based on model dims.
 - **allocation:** map a requesting sequence ID to the first available `slot_active == false` index.
 - **branching:** duplicate the parent sequence's physical state data into a new child sequence's slot.
 - **freeing:** clear the sequence mapping, making the physical slot available for future allocations.
 
 ## error codes
 
 this actor can produce the following error codes:
 
 - `EMEL_ERR_OOM` — no free slots available when allocating a sequence.
 - `EMEL_ERR_INVALID_ARGUMENT` — invalid sequence or parent id (e.g., branching from an unassigned parent).
 - `EMEL_ERR_INTERNAL` — internal invariant violation.
*/

// benchmark: scaffold

#include <memory>
#include <cstdint>

#include "emel/memory/recurrent/actions.hpp"
#include "emel/memory/recurrent/events.hpp"
#include "emel/memory/recurrent/guards.hpp"
#include "emel/memory/view.hpp"
#include "emel/sm.hpp"

namespace emel::memory::recurrent {

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
    return action::is_active(context_, seq_id);
  }

  int32_t sequence_length(const int32_t seq_id) const noexcept {
    return action::sequence_length_value(context_, seq_id);
  }

  int32_t lookup_kv_block(const int32_t, const int32_t) const noexcept { return -1; }

  int32_t lookup_recurrent_slot(const int32_t seq_id) const noexcept {
    return action::lookup_slot(context_, seq_id);
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

}  // namespace emel::memory::recurrent
