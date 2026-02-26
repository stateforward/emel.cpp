#pragma once

/*
design doc: docs/designs/memory/kv.design.md
 ---
 title: memory/kv architecture design
 status: draft
 ---
 
 # memory/kv architecture design
 
 this document defines the kv cache actor. it acts as the central paged block manager, orchestrating
 allocations, prefix caching, and zero-copy sequence branching using a strict Data-Oriented Design
 (DOD) page table.
 
 ## role
 - own the physical kv tensor buffers and logically partition them into fixed-size blocks (e.g., 16
   or 32 tokens).
 - manage the "page table" (sequence-to-block mappings) and block reference counts.
 - orchestrate complex sequence lifecycle events (allocate, free, branch) received from the `generator`.
 - handle out-of-memory states safely via SML transitions rather than crashing or returning silent
   errors.
 
 ## architecture shift: DOD with `sml::sm_pool`
 initially, every block of the kv cache was modeled as an independent `boost::sml` actor to track references.
 however, dispatching SML events (like `event::link` or `event::unlink`) to thousands of independent blocks
 introduced unacceptable overhead in the hot path.
 
 to achieve maximum performance without abandoning the safety of the Actor Model, the `memory/kv` actor manages
 its blocks using `boost::sml::sm_pool<block_sm>`.
 - **cache locality:** `sm_pool` lays out the state machine data contiguously in memory.
 - **batch dispatch:** when a sequence is freed or branched, the `memory/kv` actor does not loop over individual
   actors. instead, it calls the highly optimized `process_event_batch(seq_to_blocks, event::unlink{})` API.
 - **the result:** this batch API provides an 81% overhead reduction compared to independent actors, bringing the
   cost of SML dispatch so close to raw C arrays (~11% overhead) that we can retain the strict declarative reference
   counting (`empty` -> `filled` -> `empty`) for every single block without compromising performance.
 
 ## composition
 - owned by the `generator` (or `memory/hybrid` if mixed).
 - owns the physical backing tensors for keys and values.
 
 ## state model
 
 ```text
 uninitialized ──► initializing ──► ready
                                      │
 ready ──► allocating_sequence ──► (ready | out_of_memory)
   ▲                                  │
   └──────────────────────────────────┘
 
 ready ──► branching_sequence ──► ready
 ready ──► freeing_sequence   ──► ready
 ```
 
 - `uninitialized` — awaiting initial setup.
 - `initializing` — partitioning the backing tensors and populating the `free_pool`.
 - `ready` — waiting for lifecycle commands.
 - `allocating_sequence` — reserving a sequence ID slot. if successful, returns to `ready`. if the
   maximum number of active sequences is reached, drops to `out_of_memory`.
 - `branching_sequence` — zero-copy duplication of a parent's block mapping to a new child.
 - `freeing_sequence` — unlinking all blocks associated with a sequence ID.
 - unexpected events route to `unexpected`.
 
 ## responsibilities
 - **allocation (`event::allocate_slots`):** when a sequence needs to write new tokens, the manager
   pops `N` indices from the `free_pool`, sets their `block_refs = 1`, and appends them to the
   sequence's mapping array. if the `free_pool` is empty, gracefully reject via SML guard.
 - **zero-copy branching (`event::branch_sequence`):** given a parent sequence, duplicate its
   `seq_to_blocks` mapping to a new child sequence. iterate over those blocks and increment
   `block_refs[idx]++`. no physical memory is copied.
 - **freeing (`event::free_sequence`):** look up a sequence's blocks. iterate over them and decrement
   `block_refs[idx]--`. if a block's ref count hits 0, push its index back to the `free_pool`. clear
   the sequence mapping.
 
 ## determinism
 
 block allocation uses deterministic ordering: the allocator always selects the lowest-index free
 block first. given identical sequence state and identical event sequences, the block layout is
 identical across runs. this ensures reproducible memory layouts for debugging and testing.
 
 ## error codes
 
 this actor can produce the following error codes:
 
 - `EMEL_ERR_OOM` — the free pool is empty and no blocks are available for allocation.
 - `EMEL_ERR_CAPACITY` — capacity overflow, for example when a branch target's mapping array is too small.
 - `EMEL_ERR_INVALID_ARGUMENT` — the input payload contained an invalid sequence id or out-of-range argument.
 - `EMEL_ERR_INTERNAL` — an internal invariant was violated, such as a reference count underflow.
*/


#include <memory>
#include <cstdint>

#include "emel/memory/kv/actions.hpp"
#include "emel/memory/kv/events.hpp"
#include "emel/memory/kv/guards.hpp"
#include "emel/memory/view.hpp"
#include "emel/sm.hpp"

namespace emel::memory::kv {

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
    return action::valid_sequence_id(context_, seq_id) &&
           context_.sequence_active[static_cast<size_t>(seq_id)];
  }

  int32_t sequence_length(const int32_t seq_id) const noexcept {
    return action::sequence_length_value(context_, seq_id);
  }

  int32_t lookup_kv_block(const int32_t seq_id, const int32_t pos) const noexcept {
    return action::lookup_block_at_pos(context_, seq_id, pos);
  }

  int32_t lookup_recurrent_slot(const int32_t) const noexcept { return -1; }

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

}  // namespace emel::memory::kv
