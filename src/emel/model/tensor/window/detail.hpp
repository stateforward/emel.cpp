#pragma once

#include <cstddef>
#include <cstdint>
#include <new>
#include <span>
#include <string_view>

#include "emel/io/mmap/errors.hpp"
#include "emel/io/staged_read/events.hpp"
#include "emel/io/staged_read/sm.hpp"
#include "emel/model/tensor/window/errors.hpp"
#include "emel/sm.hpp"

namespace emel::model::tensor::window::detail {

inline constexpr uint32_t k_max_window_slots = 8u;
inline constexpr uint32_t k_max_stream_layers = 160u;
inline constexpr uint32_t k_max_weights_per_layer = 12u;
inline constexpr uint32_t k_stream_io_lanes = 2u;
inline constexpr uint64_t k_min_stream_chunk_bytes = 1u * 1024u * 1024u;
inline constexpr uint64_t k_default_stream_chunk_bytes = 8u * 1024u * 1024u;
inline constexpr uint64_t k_max_stream_chunk_bytes = 16u * 1024u * 1024u;
inline constexpr uint64_t k_slot_alignment_bytes = 64u;
inline constexpr int32_t k_stream_source_tensor_id = -2;

using stream_io_pool = emel::policy::thread_pool_scheduler<k_stream_io_lanes, 16u, 128u>;
using stream_scheduler = emel::policy::external_completion_scheduler<k_max_window_slots>;

// One contiguous weight span inside the model file, placed at slot_offset
// within its layer's window slot.
struct weight_extent {
  int32_t tensor_id = -1;
  uint64_t file_offset = 0u;
  uint64_t byte_size = 0u;
  uint64_t slot_offset = 0u;
};

struct layer_descriptor {
  std::array<weight_extent, k_max_weights_per_layer> weights = {};
  uint32_t weight_count = 0u;
  uint64_t file_begin = 0u;   // lowest file offset across weights
  uint64_t file_span = 0u;    // bytes from file_begin to the highest extent end
  uint64_t slot_bytes = 0u;   // aligned bytes this layer occupies in a slot
};

enum class slot_lifecycle : uint8_t {
  vacant = 0,
  loading = 1,
  resident = 2,
  failed = 3,
};

// Everything an I/O pool worker needs to load one layer into one slot. The
// worker's last write is ok; the completion fire's release store publishes it
// to the dispatching thread (which reads only after observing fired).
struct load_ticket {
  emel::io::staged_read::sm *io_staged = nullptr;
  const layer_descriptor *layout = nullptr;
  const void *source_base = nullptr;
  uint64_t source_bytes = 0u;
  uint8_t *slot_base = nullptr;
  uint64_t stage_chunk_bytes = 0u;
  bool ok = false;

  // staged_read requires both callbacks; the ticket reads the dispatch's
  // synchronous bool result, so these only satisfy the contract.
  static void on_copy_done(
      void *, const emel::io::staged_read::events::staged_window_done &) noexcept {}
  static void on_copy_error(
      void *, const emel::io::staged_read::events::staged_window_error &) noexcept {}

  // Runs on an I/O pool worker (or inline on submit rejection): one staged
  // chunked copy per weight extent, monotonic bounded data-plane iteration.
  void run() noexcept {
    bool all_ok = true;
    for (uint32_t index = 0; index < layout->weight_count; ++index) {
      const weight_extent &extent = layout->weights[index];
      // staged_read's single-window contract copies from the span start and
      // requires an exact-size span: pre-position it at the extent and clamp
      // the chunk to the logical length for small extents.
      const uint64_t chunk = stage_chunk_bytes < extent.byte_size
                                 ? stage_chunk_bytes
                                 : extent.byte_size;
      const emel::io::staged_read::event::staged_window_request request{
          .file_offset = extent.file_offset,
          .logical_byte_length = extent.byte_size,
          .stage_chunk_bytes = chunk,
          .source_span =
              static_cast<const uint8_t *>(source_base) + extent.file_offset,
          .source_span_bytes = extent.byte_size,
          .target_buffer = slot_base + extent.slot_offset,
          .target_window_bytes = extent.byte_size,
      };
      emel::io::staged_read::event::staged_window copy{request};
      copy.on_done = {nullptr, &load_ticket::on_copy_done};
      copy.on_error = {nullptr, &load_ticket::on_copy_error};
      all_ok = io_staged->process_event(copy) && all_ok;
    }
    ok = all_ok;
  }

  static void run_task(load_ticket *ticket) noexcept { ticket->run(); }
};

struct window_slot {
  uint8_t *storage = nullptr; // view into the caller-provided bind arena
  int32_t layer = -1;
  slot_lifecycle lifecycle = slot_lifecycle::vacant;
};

// Actor-owned residency state. Slot storage is a partition of the
// caller-provided bind arena (the machine never allocates during dispatch);
// unbind and the machine destructor join in-flight loads before the caller
// may release that arena.
struct window_state {
  std::array<layer_descriptor, k_max_stream_layers> plan = {};
  uint32_t layer_count = 0u;

  std::array<window_slot, k_max_window_slots> slots = {};
  std::array<load_ticket, k_max_window_slots> tickets = {};
  uint32_t slot_count = 0u;
  uint64_t slot_capacity_bytes = 0u;

  const void *source_base = nullptr;
  uint64_t source_bytes = 0u;
  uint32_t source_handle = emel::io::mmap::k_invalid_mapping_handle;

  uint64_t budget_bytes = 0u;
  uint32_t prefetch_depth = 0u;
  uint64_t stage_chunk_bytes = 0u;
  int32_t next_prefetch_layer = -1;
  bool streaming_active = false;
  bool bound = false;
};

inline uint32_t compute_slot_for_layer(const window_state &window,
                                       const int32_t layer) noexcept {
  return static_cast<uint32_t>(layer) % window.slot_count;
}

inline uint64_t compute_aligned_bytes(const uint64_t bytes) noexcept {
  return (bytes + (k_slot_alignment_bytes - 1u)) &
         ~(k_slot_alignment_bytes - 1u);
}

// Copies caller-provided extents into the plan and computes slot layout.
// Pure data-plane scan: no routing, no outcome selection.
inline void scan_layer_descriptors(
    const std::span<const weight_extent> extents,
    const std::span<const uint16_t> layer_weight_counts,
    window_state &window) noexcept {
  window.layer_count = static_cast<uint32_t>(layer_weight_counts.size());
  uint64_t max_slot_bytes = 0u;
  size_t cursor = 0u;
  for (uint32_t layer = 0; layer < window.layer_count; ++layer) {
    layer_descriptor &layout = window.plan[layer];
    layout.weight_count = layer_weight_counts[layer];
    uint64_t slot_offset = 0u;
    uint64_t file_begin = ~0ull;
    uint64_t file_end = 0u;
    for (uint32_t index = 0; index < layout.weight_count; ++index) {
      weight_extent extent = extents[cursor];
      cursor += 1u;
      extent.slot_offset = slot_offset;
      slot_offset += compute_aligned_bytes(extent.byte_size);
      file_begin = extent.file_offset < file_begin ? extent.file_offset : file_begin;
      const uint64_t extent_end = extent.file_offset + extent.byte_size;
      file_end = extent_end > file_end ? extent_end : file_end;
      layout.weights[index] = extent;
    }
    layout.slot_bytes = slot_offset;
    layout.file_begin = layout.weight_count > 0u ? file_begin : 0u;
    layout.file_span = layout.weight_count > 0u ? file_end - layout.file_begin : 0u;
    max_slot_bytes = layout.slot_bytes > max_slot_bytes ? layout.slot_bytes : max_slot_bytes;
  }
  window.slot_capacity_bytes = max_slot_bytes;
}

inline uint64_t compute_total_stream_bytes(const window_state &window) noexcept {
  uint64_t total = 0u;
  for (uint32_t layer = 0; layer < window.layer_count; ++layer) {
    total += window.plan[layer].slot_bytes;
  }
  return total;
}

} // namespace emel::model::tensor::window::detail

namespace emel::model::tensor::window::event {

struct bind_window;
struct acquire_layer_window;
struct unbind_window;

} // namespace emel::model::tensor::window::event

namespace emel::model::tensor::window::detail {

struct bind_attempt_status {
  emel::error::type err = emel::error::cast(error::none);
  bool ok = false;
  bool source_map_ok = false;
  // Result of the post-map rejection release attempt; defaults true so
  // pre-map rejections (nothing mapped) take the plain unbound exit.
  bool release_ok = true;
  const void *source_base = nullptr;
  uint64_t source_bytes = 0u;
  uint32_t source_handle = emel::io::mmap::k_invalid_mapping_handle;
};

struct acquire_attempt_status {
  emel::error::type err = emel::error::cast(error::none);
  bool ok = false;
  const uint8_t *slot_base = nullptr;
  const layer_descriptor *layout = nullptr;
};

struct unbind_attempt_status {
  emel::error::type err = emel::error::cast(error::none);
  bool ok = false;
};

// Internal same-RTC carriers: public request + mutable status + the co_sm
// scheduler the effects arm/require completion sources on (threaded through
// the event so context stays free of dispatch-local pointers).
struct bind_window_runtime {
  const event::bind_window &request;
  bind_attempt_status &status;
  stream_scheduler &scheduler;
};

// First acquire dispatch: joins a slot that is still mid-load for a different
// layer (non-sequential access) so the settle dispatch can safely reuse it.
struct acquire_resolve_runtime {
  const event::acquire_layer_window &request;
  acquire_attempt_status &status;
  stream_scheduler &scheduler;
};

// Second acquire dispatch: decides resident/loading/unscheduled for the
// target layer, submitting and requiring its slot load as needed.
struct acquire_runtime {
  const event::acquire_layer_window &request;
  acquire_attempt_status &status;
  stream_scheduler &scheduler;
};

// Second acquire dispatch: the begin dispatch's drain has already committed
// the required slot load; this one routes on the committed residency,
// publishes done/error with the original request payload, and advances the
// prefetch ring (arming the ahead slot's source on the same scheduler).
struct acquire_publish_runtime {
  const event::acquire_layer_window &request;
  acquire_attempt_status &status;
  stream_scheduler &scheduler;
};

struct unbind_runtime {
  const event::unbind_window &request;
  unbind_attempt_status &status;
  stream_scheduler &scheduler;
};

struct unbind_finish_runtime {
  const event::unbind_window &request;
  unbind_attempt_status &status;
};

inline void reset_window(window_state &window) noexcept {
  for (window_slot &slot : window.slots) {
    slot.storage = nullptr;
    slot.layer = -1;
    slot.lifecycle = slot_lifecycle::vacant;
  }
  window.slot_count = 0u;
  window.slot_capacity_bytes = 0u;
  window.source_base = nullptr;
  window.source_bytes = 0u;
  window.source_handle = emel::io::mmap::k_invalid_mapping_handle;
  window.layer_count = 0u;
  window.next_prefetch_layer = -1;
  window.streaming_active = false;
  window.bound = false;
}

} // namespace emel::model::tensor::window::detail
