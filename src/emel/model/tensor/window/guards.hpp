#pragma once

#include "emel/model/tensor/window/context.hpp"
#include "emel/model/tensor/window/detail.hpp"
#include "emel/model/tensor/window/errors.hpp"
#include "emel/model/tensor/window/events.hpp"

namespace emel::model::tensor::window::guard {

//------------------------------------------------------------------------------//
// bind chain

struct guard_bind_request_valid {
  bool operator()(const detail::bind_window_runtime &ev,
                  const action::context &ctx) const noexcept {
    const event::bind_window_request &request = ev.request.request;
    if (ctx.window.bound || ctx.io_mmap == nullptr || ctx.io_pool == nullptr) {
      return false;
    }
    if (request.file_path.empty() || request.file_size_bytes == 0u) {
      return false;
    }
    const size_t layer_count = request.layer_weight_counts.size();
    if (layer_count == 0u || layer_count > detail::k_max_stream_layers) {
      return false;
    }
    size_t extent_total = 0u;
    for (const uint16_t count : request.layer_weight_counts) {
      if (count == 0u || count > detail::k_max_weights_per_layer) {
        return false;
      }
      extent_total += count;
    }
    if (extent_total != request.extents.size()) {
      return false;
    }
    if (request.stage_chunk_bytes < detail::k_min_stream_chunk_bytes ||
        request.stage_chunk_bytes > detail::k_max_stream_chunk_bytes) {
      return false;
    }
    if (request.window_slots < 2u ||
        request.window_slots > detail::k_max_window_slots ||
        request.prefetch_depth == 0u ||
        request.prefetch_depth >= request.window_slots) {
      return false;
    }
    return ctx.io_staged.size() >= request.window_slots;
  }
};

struct guard_bind_request_invalid {
  bool operator()(const detail::bind_window_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_bind_request_valid{}(ev, ctx);
  }
};

struct guard_source_map_succeeded {
  bool operator()(const detail::bind_window_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.source_map_ok;
  }
};

struct guard_source_map_failed {
  bool operator()(const detail::bind_window_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_source_map_succeeded{}(ev, ctx);
  }
};

struct guard_bind_fits_budget {
  bool operator()(const detail::bind_window_runtime &ev,
                  const action::context &ctx) const noexcept {
    const uint64_t budget = ev.request.request.budget_bytes;
    return budget == 0u || detail::compute_total_stream_bytes(ctx.window) <= budget;
  }
};

struct guard_bind_requires_streaming {
  bool operator()(const detail::bind_window_runtime &ev,
                  const action::context &ctx) const noexcept {
    if (guard_bind_fits_budget{}(ev, ctx)) {
      return false;
    }
    const uint64_t slot_bytes = static_cast<uint64_t>(ev.request.request.window_slots) *
                                ctx.window.slot_capacity_bytes;
    return slot_bytes <= ev.request.request.budget_bytes;
  }
};

struct guard_bind_budget_too_small {
  bool operator()(const detail::bind_window_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_bind_fits_budget{}(ev, ctx) &&
           !guard_bind_requires_streaming{}(ev, ctx);
  }
};

struct guard_bind_done_callback_present {
  bool operator()(const detail::bind_window_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct guard_bind_done_callback_absent {
  bool operator()(const detail::bind_window_runtime &ev) const noexcept {
    return !guard_bind_done_callback_present{}(ev);
  }
};

struct guard_bind_error_callback_present {
  bool operator()(const detail::bind_window_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct guard_bind_error_callback_absent {
  bool operator()(const detail::bind_window_runtime &ev) const noexcept {
    return !guard_bind_error_callback_present{}(ev);
  }
};

//------------------------------------------------------------------------------//
// acquire begin chain

struct guard_acquire_layer_in_range {
  bool operator()(const detail::acquire_runtime &ev,
                  const action::context &ctx) const noexcept {
    return ev.request.layer_index >= 0 &&
           static_cast<uint32_t>(ev.request.layer_index) < ctx.window.layer_count;
  }
};

struct guard_acquire_layer_out_of_range {
  bool operator()(const detail::acquire_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_acquire_layer_in_range{}(ev, ctx);
  }
};

struct guard_acquire_layer_resident {
  bool operator()(const detail::acquire_runtime &ev,
                  const action::context &ctx) const noexcept {
    if (!guard_acquire_layer_in_range{}(ev, ctx)) {
      return false;
    }
    const detail::window_slot &slot =
        ctx.window.slots[detail::compute_slot_for_layer(ctx.window, ev.request.layer_index)];
    return slot.layer == ev.request.layer_index &&
           slot.lifecycle == detail::slot_lifecycle::resident;
  }
};

struct guard_acquire_layer_loading {
  bool operator()(const detail::acquire_runtime &ev,
                  const action::context &ctx) const noexcept {
    if (!guard_acquire_layer_in_range{}(ev, ctx)) {
      return false;
    }
    const detail::window_slot &slot =
        ctx.window.slots[detail::compute_slot_for_layer(ctx.window, ev.request.layer_index)];
    return slot.layer == ev.request.layer_index &&
           slot.lifecycle == detail::slot_lifecycle::loading;
  }
};

struct guard_acquire_layer_unscheduled {
  bool operator()(const detail::acquire_runtime &ev,
                  const action::context &ctx) const noexcept {
    if (!guard_acquire_layer_in_range{}(ev, ctx)) {
      return false;
    }
    const detail::window_slot &slot =
        ctx.window.slots[detail::compute_slot_for_layer(ctx.window, ev.request.layer_index)];
    return slot.layer != ev.request.layer_index;
  }
};

//------------------------------------------------------------------------------//
// acquire publish chain (second dispatch, routed on committed slot state)

struct guard_acquire_result_ready {
  bool operator()(const detail::acquire_publish_runtime &ev,
                  const action::context &ctx) const noexcept {
    if (ev.status.err != emel::error::cast(error::none)) {
      return false;
    }
    const detail::window_slot &slot =
        ctx.window.slots[detail::compute_slot_for_layer(ctx.window, ev.request.layer_index)];
    return slot.layer == ev.request.layer_index &&
           slot.lifecycle == detail::slot_lifecycle::resident;
  }
};

struct guard_acquire_error_pending {
  bool operator()(const detail::acquire_publish_runtime &ev,
                  const action::context &) const noexcept {
    return ev.status.err != emel::error::cast(error::none);
  }
};

struct guard_acquire_copy_failed {
  bool operator()(const detail::acquire_publish_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_acquire_error_pending{}(ev, ctx) &&
           !guard_acquire_result_ready{}(ev, ctx);
  }
};

// Ring advance: the layer prefetch_depth ahead (wrapping into the next pass)
// still needs a load submitted for its slot.
struct guard_prefetch_ahead_needed {
  bool operator()(const detail::acquire_publish_runtime &ev,
                  const action::context &ctx) const noexcept {
    const int32_t ahead = ev.request.layer_index +
                          static_cast<int32_t>(ctx.window.prefetch_depth);
    const int32_t wrapped =
        ahead % static_cast<int32_t>(ctx.window.layer_count);
    const detail::window_slot &slot =
        ctx.window.slots[detail::compute_slot_for_layer(ctx.window, wrapped)];
    return slot.layer != wrapped &&
           slot.lifecycle != detail::slot_lifecycle::loading;
  }
};

struct guard_prefetch_ahead_not_needed {
  bool operator()(const detail::acquire_publish_runtime &ev,
                  const action::context &ctx) const noexcept {
    return !guard_prefetch_ahead_needed{}(ev, ctx);
  }
};

struct guard_acquire_done_callback_present {
  bool operator()(const detail::acquire_publish_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct guard_acquire_done_callback_absent {
  bool operator()(const detail::acquire_publish_runtime &ev) const noexcept {
    return !guard_acquire_done_callback_present{}(ev);
  }
};

struct guard_acquire_error_callback_present {
  bool operator()(const detail::acquire_publish_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_error);
  }
};

struct guard_acquire_error_callback_absent {
  bool operator()(const detail::acquire_publish_runtime &ev) const noexcept {
    return !guard_acquire_error_callback_present{}(ev);
  }
};

//------------------------------------------------------------------------------//
// unbind chain

struct guard_unbind_done_callback_present {
  bool operator()(const detail::unbind_finish_runtime &ev) const noexcept {
    return static_cast<bool>(ev.request.on_done);
  }
};

struct guard_unbind_done_callback_absent {
  bool operator()(const detail::unbind_finish_runtime &ev) const noexcept {
    return !guard_unbind_done_callback_present{}(ev);
  }
};

} // namespace emel::model::tensor::window::guard
