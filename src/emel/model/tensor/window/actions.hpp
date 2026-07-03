#pragma once

#include <cstdint>
#include <new>

#include "emel/io/mmap/events.hpp"
#include "emel/model/tensor/window/context.hpp"
#include "emel/model/tensor/window/detail.hpp"
#include "emel/model/tensor/window/errors.hpp"
#include "emel/model/tensor/window/events.hpp"

namespace emel::model::tensor::window::action {

namespace thunk {

inline void on_source_map_done(
    void *object, const emel::io::mmap::events::map_tensor_done &ev) noexcept {
  auto *status = static_cast<detail::bind_attempt_status *>(object);
  status->source_map_ok = true;
  status->source_base = ev.buffer;
  status->source_bytes = ev.buffer_bytes;
  status->source_handle = ev.handle;
}

} // namespace thunk

// Shared by prime, acquire-miss, and ring-advance effects (all in this file):
// arms the slot's completion source, hints the OS ahead of the copy, stages
// the load ticket, and hands it to the I/O pool. A rejected submit runs the
// ticket inline on the caller (bounded backpressure, mirroring the sliced
// matmul fallback) and fires the source so drain semantics stay uniform.
inline void bind_and_submit_layer_load(context &ctx,
                                       detail::stream_scheduler &scheduler,
                                       const int32_t layer) noexcept {
  const uint32_t slot_index = detail::compute_slot_for_layer(ctx.window, layer);
  detail::window_slot &slot = ctx.window.slots[slot_index];
  const detail::layer_descriptor &layout =
      ctx.window.plan[static_cast<uint32_t>(layer)];

  slot.layer = layer;
  slot.lifecycle = detail::slot_lifecycle::loading;

  const emel::io::mmap::event::advise_mapping willneed{
      detail::k_stream_source_tensor_id, ctx.window.source_handle,
      layout.file_begin, layout.file_span,
      emel::io::mmap::event::advice::k_willneed};
  (void)ctx.io_mmap->process_event(willneed);

  detail::load_ticket &ticket = ctx.window.tickets[slot_index];
  ticket.io_staged = &ctx.io_staged[slot_index];
  ticket.layout = &layout;
  ticket.source_base = ctx.window.source_base;
  ticket.source_bytes = ctx.window.source_bytes;
  ticket.slot_base = slot.storage;
  ticket.stage_chunk_bytes = ctx.window.stage_chunk_bytes;
  ticket.ok = false;

  scheduler.source(slot_index).arm();
  const bool submitted = ctx.io_pool->try_submit_with_completion(
      [ticket_ptr = &ticket]() noexcept { detail::load_ticket::run_task(ticket_ptr); },
      &scheduler.source(slot_index), &emel::policy::completion_source::fire);
  if (!submitted) {
    ticket.run();
    emel::policy::completion_source::fire(&scheduler.source(slot_index));
  }
}

//------------------------------------------------------------------------------//
// bind chain

struct effect_begin_bind {
  void operator()(const detail::bind_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = false;
    ev.status.source_map_ok = false;
    ev.status.source_base = nullptr;
    ev.status.source_bytes = 0u;
    ev.status.source_handle = emel::io::mmap::k_invalid_mapping_handle;
  }
};

struct effect_mark_bind_invalid {
  void operator()(const detail::bind_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::invalid_request);
    ev.status.ok = false;
  }
};

struct effect_mark_bind_already_bound {
  void operator()(const detail::bind_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::already_bound);
    ev.status.ok = false;
  }
};

struct effect_map_source {
  void operator()(const detail::bind_window_runtime &ev,
                  context &ctx) const noexcept {
    const emel::io::mmap::event::map_tensor_request request{
        .tensor_id = detail::k_stream_source_tensor_id,
        .file_index = 0u,
        .file_offset = 0u,
        .byte_size = ev.request.request.file_size_bytes,
        .file_path = ev.request.request.file_path,
    };
    emel::io::mmap::event::map_tensor map_request{request};
    map_request.on_done = {&ev.status, thunk::on_source_map_done};
    (void)ctx.io_mmap->process_event(map_request);
  }
};

struct effect_mark_source_map_failed {
  void operator()(const detail::bind_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::source_map_failed);
    ev.status.ok = false;
  }
};

struct effect_scan_layer_plan {
  void operator()(const detail::bind_window_runtime &ev,
                  context &ctx) const noexcept {
    detail::scan_layer_descriptors(ev.request.request.extents,
                                   ev.request.request.layer_weight_counts,
                                   ctx.window);
    ctx.window.source_base = ev.status.source_base;
    ctx.window.source_bytes = ev.status.source_bytes;
    ctx.window.source_handle = ev.status.source_handle;
    ctx.window.budget_bytes = ev.request.request.budget_bytes;
    ctx.window.prefetch_depth = ev.request.request.prefetch_depth;
    ctx.window.stage_chunk_bytes = ev.request.request.stage_chunk_bytes;
  }
};

struct effect_mark_budget_too_small {
  void operator()(const detail::bind_window_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::budget_too_small);
    ev.status.ok = false;
  }
};

// Bind rejections after the source was mapped must give the mapping back to
// the shared io_mmap pool, or repeated failing binds exhaust its fixed slots.
struct effect_release_rejected_source {
  void operator()(const detail::bind_window_runtime &,
                  context &ctx) const noexcept {
    const emel::io::mmap::event::release_mapping release{
        detail::k_stream_source_tensor_id, ctx.window.source_handle};
    (void)ctx.io_mmap->process_event(release);
    detail::reset_window(ctx.window);
  }
};

struct effect_activate_passthrough {
  void operator()(const detail::bind_window_runtime &ev,
                  context &ctx) const noexcept {
    ctx.window.streaming_active = false;
    ctx.window.bound = true;
    ev.status.ok = true;
  }
};

// One-time setup allocation: window slots are the streaming feature's working
// memory, sized at bind before any hot-path dispatch, freed at unbind/context
// destruction. This is the sanctioned construction-time heap use. The
// allocation is an attempt whose outcome the alloc-decision guards route on:
// nothrow new, all-or-nothing recorded in status.slots_alloc_ok.
struct effect_allocate_slots {
  void operator()(const detail::bind_window_runtime &ev,
                  context &ctx) const noexcept {
    ctx.window.slot_count = ev.request.request.window_slots;
    bool all_ok = true;
    for (uint32_t index = 0; index < ctx.window.slot_count; ++index) {
      ctx.window.slots[index].storage = static_cast<uint8_t *>(::operator new(
          ctx.window.slot_capacity_bytes,
          std::align_val_t{detail::k_slot_alignment_bytes}, std::nothrow));
      ctx.window.slots[index].layer = -1;
      ctx.window.slots[index].lifecycle = detail::slot_lifecycle::vacant;
      all_ok = all_ok && ctx.window.slots[index].storage != nullptr;
    }
    ev.status.slots_alloc_ok = all_ok;
  }
};

struct effect_finish_streaming {
  void operator()(const detail::bind_window_runtime &ev,
                  context &ctx) const noexcept {
    ctx.window.streaming_active = true;
    ctx.window.bound = true;
    ev.status.ok = true;
  }
};

struct effect_prime_window {
  void operator()(const detail::bind_window_runtime &ev,
                  context &ctx) const noexcept {
    const uint32_t depth = ctx.window.prefetch_depth < ctx.window.layer_count
                               ? ctx.window.prefetch_depth
                               : ctx.window.layer_count;
    for (uint32_t layer = 0; layer < depth; ++layer) {
      bind_and_submit_layer_load(ctx, ev.scheduler, static_cast<int32_t>(layer));
    }
  }
};

struct effect_publish_bind_done {
  void operator()(const detail::bind_window_runtime &ev,
                  context &ctx) const noexcept {
    ev.request.on_done(events::bind_window_done{
        .request = ev.request,
        .streaming_active = ctx.window.streaming_active,
        .source_base = ctx.window.source_base,
        .source_bytes = ctx.window.source_bytes,
        .window_slots = ctx.window.slot_count,
    });
  }
};

struct effect_record_bind_done {
  void operator()(const detail::bind_window_runtime &,
                  context &) const noexcept {}
};

struct effect_publish_bind_error {
  void operator()(const detail::bind_window_runtime &ev,
                  context &) const noexcept {
    ev.request.on_error(events::bind_window_error{
        .request = ev.request,
        .err = ev.status.err,
    });
  }
};

struct effect_record_bind_error {
  void operator()(const detail::bind_window_runtime &,
                  context &) const noexcept {}
};

//------------------------------------------------------------------------------//
// acquire resolve chain (first dispatch)

struct effect_begin_acquire_resolve {
  void operator()(const detail::acquire_resolve_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = false;
    ev.status.slot_base = nullptr;
    ev.status.layout = nullptr;
  }
};

struct effect_mark_resolve_out_of_range {
  void operator()(const detail::acquire_resolve_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::layer_out_of_range);
  }
};

// Joins the busy slot: the drain suspends this dispatch until the in-flight
// load for the slot's current occupant fires and commits.
struct effect_require_busy_slot {
  void operator()(const detail::acquire_resolve_runtime &ev,
                  context &ctx) const noexcept {
    ev.scheduler.require(
        detail::compute_slot_for_layer(ctx.window, ev.request.layer_index));
  }
};

struct effect_mark_resolve_not_streaming {
  void operator()(const detail::acquire_resolve_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::not_streaming);
  }
};

struct effect_mark_resolve_not_bound {
  void operator()(const detail::acquire_resolve_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::not_bound);
  }
};

struct effect_begin_resolve_not_streaming {
  void operator()(const detail::acquire_resolve_runtime &ev,
                  context &ctx) const noexcept {
    effect_begin_acquire_resolve{}(ev, ctx);
    effect_mark_resolve_not_streaming{}(ev, ctx);
  }
};

struct effect_begin_resolve_not_bound {
  void operator()(const detail::acquire_resolve_runtime &ev,
                  context &ctx) const noexcept {
    effect_begin_acquire_resolve{}(ev, ctx);
    effect_mark_resolve_not_bound{}(ev, ctx);
  }
};

//------------------------------------------------------------------------------//
// acquire settle chain (second dispatch)

struct effect_mark_acquire_out_of_range {
  void operator()(const detail::acquire_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::layer_out_of_range);
  }
};

struct effect_submit_layer_load {
  void operator()(const detail::acquire_runtime &ev,
                  context &ctx) const noexcept {
    bind_and_submit_layer_load(ctx, ev.scheduler, ev.request.layer_index);
  }
};

// Marks the acquired layer's slot as blocking this dispatch: the co_sm drain
// suspends the dispatch coroutine until the already-submitted load fires.
struct effect_require_layer_completion {
  void operator()(const detail::acquire_runtime &ev,
                  context &ctx) const noexcept {
    ev.scheduler.require(
        detail::compute_slot_for_layer(ctx.window, ev.request.layer_index));
  }
};

//------------------------------------------------------------------------------//
// completion commit (delivered by the co_sm drain or the next dispatch sweep)

struct effect_commit_slot_load {
  void operator()(const emel::event::completion &ev,
                  context &ctx) const noexcept {
    detail::window_slot &slot = ctx.window.slots[ev.source_index];
    const detail::load_ticket &ticket = ctx.window.tickets[ev.source_index];
    // resident(2) on success, failed(3) otherwise - branch-free select.
    slot.lifecycle = static_cast<detail::slot_lifecycle>(
        3u - static_cast<uint8_t>(ticket.ok));
    // The copy pulled the source pages through the page cache; drop them
    // behind the window to bound cache pressure for larger-than-RAM files.
    const emel::io::mmap::event::advise_mapping dontneed{
        detail::k_stream_source_tensor_id, ctx.window.source_handle,
        ticket.layout->file_begin, ticket.layout->file_span,
        emel::io::mmap::event::advice::k_dontneed};
    (void)ctx.io_mmap->process_event(dontneed);
  }
};

//------------------------------------------------------------------------------//
// acquire publish chain (second dispatch)

struct effect_stage_acquire_result {
  void operator()(const detail::acquire_publish_runtime &ev,
                  context &ctx) const noexcept {
    const uint32_t slot_index =
        detail::compute_slot_for_layer(ctx.window, ev.request.layer_index);
    ev.status.slot_base = ctx.window.slots[slot_index].storage;
    ev.status.layout =
        &ctx.window.plan[static_cast<uint32_t>(ev.request.layer_index)];
    ev.status.ok = true;
  }
};

struct effect_mark_slot_copy_failed {
  void operator()(const detail::acquire_publish_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::slot_copy_failed);
    ev.status.ok = false;
  }
};

// Ring advance: prefetch the layer prefetch_depth ahead (wrapping into the
// next pass for a warm window across tokens) so its slot load overlaps the
// caller's compute on the just-published layer. The prefetch-needed decision
// is a guard (guard_prefetch_ahead_needed); this effect only executes it.
struct effect_advance_window {
  void operator()(const detail::acquire_publish_runtime &ev,
                  context &ctx) const noexcept {
    const int32_t ahead = ev.request.layer_index +
                          static_cast<int32_t>(ctx.window.prefetch_depth);
    const int32_t wrapped =
        ahead % static_cast<int32_t>(ctx.window.layer_count);
    bind_and_submit_layer_load(ctx, ev.scheduler, wrapped);
  }
};

struct effect_publish_acquire_done {
  void operator()(const detail::acquire_publish_runtime &ev,
                  context &) const noexcept {
    ev.request.on_done(events::acquire_layer_window_done{
        .request = ev.request,
        .slot_base = ev.status.slot_base,
        .layout = *ev.status.layout,
    });
  }
};

struct effect_record_acquire_done {
  void operator()(const detail::acquire_publish_runtime &,
                  context &) const noexcept {}
};

struct effect_publish_acquire_error {
  void operator()(const detail::acquire_publish_runtime &ev,
                  context &) const noexcept {
    ev.request.on_error(events::acquire_layer_window_error{
        .request = ev.request,
        .err = ev.status.err,
    });
  }
};

struct effect_record_acquire_error {
  void operator()(const detail::acquire_publish_runtime &,
                  context &) const noexcept {}
};

//------------------------------------------------------------------------------//
// unbind chain

struct effect_begin_unbind {
  void operator()(const detail::unbind_runtime &ev,
                  context &ctx) const noexcept {
    ev.status.err = emel::error::cast(error::none);
    ev.status.ok = false;
    // Data-plane scan: register a wait for exactly the in-flight slot set so
    // the drain joins outstanding loads before this dispatch returns.
    for (uint32_t index = 0; index < ctx.window.slot_count; ++index) {
      if (ctx.window.slots[index].lifecycle == detail::slot_lifecycle::loading) {
        ev.scheduler.require(index);
      }
    }
  }
};

// Attempt only: the release outcome routes the finish decision. The window
// is reset by effect_reset_window_after_release on the success path; a
// failed release keeps the handle and slot state intact so the caller can
// retry unbind (the mmap actor keeps the slot reserved on failure).
struct effect_attempt_release_source {
  void operator()(const detail::unbind_finish_runtime &ev,
                  context &ctx) const noexcept {
    emel::io::mmap::event::release_mapping release{
        detail::k_stream_source_tensor_id, ctx.window.source_handle};
    ev.status.ok = ctx.io_mmap->process_event(release);
  }
};

struct effect_reset_window_after_release {
  void operator()(const detail::unbind_finish_runtime &,
                  context &ctx) const noexcept {
    detail::reset_window(ctx.window);
  }
};

struct effect_publish_unbind_done {
  void operator()(const detail::unbind_finish_runtime &ev,
                  context &) const noexcept {
    ev.request.on_done(events::unbind_window_done{.request = ev.request});
  }
};

struct effect_record_unbind_done {
  void operator()(const detail::unbind_finish_runtime &,
                  context &) const noexcept {}
};

struct effect_mark_unbind_not_bound {
  void operator()(const detail::unbind_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::not_bound);
    ev.status.ok = false;
  }
};

struct effect_mark_unbind_release_failed {
  void operator()(const detail::unbind_finish_runtime &ev,
                  context &) const noexcept {
    ev.status.err = emel::error::cast(error::internal_error);
    ev.status.ok = false;
  }
};

struct effect_publish_unbind_error {
  void operator()(const detail::unbind_finish_runtime &ev,
                  context &) const noexcept {
    ev.request.on_error(events::unbind_window_error{
        .request = ev.request,
        .err = ev.status.err,
    });
  }
};

struct effect_record_unbind_error {
  void operator()(const detail::unbind_finish_runtime &,
                  context &) const noexcept {}
};

// A completion that arrives with no bound window (post-unbind stray): record
// only, never touch slot state.
struct effect_record_stray_completion {
  void operator()(const emel::event::completion &, context &) const noexcept {}
};

//------------------------------------------------------------------------------//
// composed effects: one callable per transition row.

struct effect_activate_passthrough_and_publish {
  void operator()(const detail::bind_window_runtime &ev,
                  context &ctx) const noexcept {
    effect_activate_passthrough{}(ev, ctx);
    effect_publish_bind_done{}(ev, ctx);
  }
};

struct effect_finish_streaming_and_publish {
  void operator()(const detail::bind_window_runtime &ev,
                  context &ctx) const noexcept {
    effect_finish_streaming{}(ev, ctx);
    effect_prime_window{}(ev, ctx);
    effect_publish_bind_done{}(ev, ctx);
  }
};

struct effect_finish_streaming_and_record {
  void operator()(const detail::bind_window_runtime &ev,
                  context &ctx) const noexcept {
    effect_finish_streaming{}(ev, ctx);
    effect_prime_window{}(ev, ctx);
    effect_record_bind_done{}(ev, ctx);
  }
};

struct effect_release_and_mark_alloc_failed {
  void operator()(const detail::bind_window_runtime &ev,
                  context &ctx) const noexcept {
    // reset_window frees whatever slots the attempt did allocate.
    effect_release_rejected_source{}(ev, ctx);
    ev.status.err = emel::error::cast(error::slot_alloc_failed);
    ev.status.ok = false;
  }
};

struct effect_release_and_mark_budget_too_small {
  void operator()(const detail::bind_window_runtime &ev,
                  context &ctx) const noexcept {
    effect_release_rejected_source{}(ev, ctx);
    effect_mark_budget_too_small{}(ev, ctx);
  }
};

struct effect_release_and_mark_streaming_config_invalid {
  void operator()(const detail::bind_window_runtime &ev,
                  context &ctx) const noexcept {
    effect_release_rejected_source{}(ev, ctx);
    effect_mark_bind_invalid{}(ev, ctx);
  }
};

struct effect_activate_passthrough_and_record {
  void operator()(const detail::bind_window_runtime &ev,
                  context &ctx) const noexcept {
    effect_activate_passthrough{}(ev, ctx);
    effect_record_bind_done{}(ev, ctx);
  }
};

struct effect_mark_already_bound_and_publish {
  void operator()(const detail::bind_window_runtime &ev,
                  context &ctx) const noexcept {
    effect_mark_bind_already_bound{}(ev, ctx);
    effect_publish_bind_error{}(ev, ctx);
  }
};

struct effect_submit_and_require_layer {
  void operator()(const detail::acquire_runtime &ev,
                  context &ctx) const noexcept {
    effect_submit_layer_load{}(ev, ctx);
    effect_require_layer_completion{}(ev, ctx);
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &ev, context &) const noexcept {
    if constexpr (requires { ev.status.err; }) {
      ev.status.err = emel::error::cast(error::internal_error);
      ev.status.ok = false;
    }
  }
};

inline constexpr effect_begin_bind effect_begin_bind{};
inline constexpr effect_mark_bind_invalid effect_mark_bind_invalid{};
inline constexpr effect_mark_bind_already_bound effect_mark_bind_already_bound{};
inline constexpr effect_map_source effect_map_source{};
inline constexpr effect_mark_source_map_failed effect_mark_source_map_failed{};
inline constexpr effect_scan_layer_plan effect_scan_layer_plan{};
inline constexpr effect_mark_budget_too_small effect_mark_budget_too_small{};
inline constexpr effect_activate_passthrough effect_activate_passthrough{};
inline constexpr effect_allocate_slots effect_allocate_slots{};
inline constexpr effect_finish_streaming effect_finish_streaming{};
inline constexpr effect_release_rejected_source
    effect_release_rejected_source{};
inline constexpr effect_prime_window effect_prime_window{};
inline constexpr effect_publish_bind_done effect_publish_bind_done{};
inline constexpr effect_record_bind_done effect_record_bind_done{};
inline constexpr effect_publish_bind_error effect_publish_bind_error{};
inline constexpr effect_record_bind_error effect_record_bind_error{};
inline constexpr effect_begin_acquire_resolve effect_begin_acquire_resolve{};
inline constexpr effect_mark_resolve_out_of_range effect_mark_resolve_out_of_range{};
inline constexpr effect_require_busy_slot effect_require_busy_slot{};
inline constexpr effect_begin_resolve_not_streaming effect_begin_resolve_not_streaming{};
inline constexpr effect_begin_resolve_not_bound effect_begin_resolve_not_bound{};
inline constexpr effect_mark_acquire_out_of_range effect_mark_acquire_out_of_range{};
inline constexpr effect_submit_layer_load effect_submit_layer_load{};
inline constexpr effect_require_layer_completion effect_require_layer_completion{};
inline constexpr effect_commit_slot_load effect_commit_slot_load{};
inline constexpr effect_stage_acquire_result effect_stage_acquire_result{};
inline constexpr effect_mark_slot_copy_failed effect_mark_slot_copy_failed{};
inline constexpr effect_advance_window effect_advance_window{};
inline constexpr effect_publish_acquire_done effect_publish_acquire_done{};
inline constexpr effect_record_acquire_done effect_record_acquire_done{};
inline constexpr effect_publish_acquire_error effect_publish_acquire_error{};
inline constexpr effect_record_acquire_error effect_record_acquire_error{};
inline constexpr effect_begin_unbind effect_begin_unbind{};
inline constexpr effect_attempt_release_source effect_attempt_release_source{};
inline constexpr effect_reset_window_after_release
    effect_reset_window_after_release{};
inline constexpr effect_publish_unbind_done effect_publish_unbind_done{};
inline constexpr effect_record_unbind_done effect_record_unbind_done{};
inline constexpr effect_mark_unbind_not_bound effect_mark_unbind_not_bound{};
inline constexpr effect_mark_unbind_release_failed effect_mark_unbind_release_failed{};
inline constexpr effect_publish_unbind_error effect_publish_unbind_error{};
inline constexpr effect_record_unbind_error effect_record_unbind_error{};
inline constexpr effect_record_stray_completion effect_record_stray_completion{};
inline constexpr effect_activate_passthrough_and_publish
    effect_activate_passthrough_and_publish{};
inline constexpr effect_activate_passthrough_and_record
    effect_activate_passthrough_and_record{};
inline constexpr effect_finish_streaming_and_publish
    effect_finish_streaming_and_publish{};
inline constexpr effect_finish_streaming_and_record
    effect_finish_streaming_and_record{};
inline constexpr effect_release_and_mark_alloc_failed
    effect_release_and_mark_alloc_failed{};
inline constexpr effect_release_and_mark_budget_too_small
    effect_release_and_mark_budget_too_small{};
inline constexpr effect_release_and_mark_streaming_config_invalid
    effect_release_and_mark_streaming_config_invalid{};
inline constexpr effect_mark_already_bound_and_publish
    effect_mark_already_bound_and_publish{};
inline constexpr effect_submit_and_require_layer effect_submit_and_require_layer{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

} // namespace emel::model::tensor::window::action
