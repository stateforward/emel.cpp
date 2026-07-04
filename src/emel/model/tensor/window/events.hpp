#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/model/tensor/window/detail.hpp"
#include "emel/model/tensor/window/errors.hpp"

namespace emel::model::tensor::window::events {

struct bind_window_done;
struct bind_window_error;
struct acquire_layer_window_done;
struct acquire_layer_window_error;
struct unbind_window_done;
struct unbind_window_error;

} // namespace emel::model::tensor::window::events

namespace emel::model::tensor::window::event {

// Establishes the streaming window over a model file: maps the whole file as
// the copy source, records the caller-built per-layer weight extents, and —
// when the streamed weight bytes exceed budget_bytes — partitions the
// caller-provided slot_storage into window slots, starts prefetch, and
// reports streaming_active. A zero budget or a fitting model resolves to the
// passthrough mode with no slots and no overhead. Slot storage is owned by
// the caller for the lifetime of the bind (the machine never allocates
// during dispatch): it must be k_slot_alignment_bytes-aligned and hold
// window_slots * (largest aligned layer span) bytes; unbind (or the machine
// destructor's drain) joins in-flight loads before the caller may free it.
struct bind_window_request {
  std::string_view file_path = {};
  uint64_t file_size_bytes = 0u;
  // Flattened per-layer weight extents: layer_weight_counts[i] entries per
  // layer i, in ascending layer order. slot_offset fields are computed by the
  // machine; callers fill tensor_id/file_offset/byte_size only.
  std::span<const detail::weight_extent> extents = {};
  std::span<const uint16_t> layer_weight_counts = {};
  uint64_t budget_bytes = 0u;
  std::span<uint8_t> slot_storage = {};
  uint32_t window_slots = 0u;
  uint32_t prefetch_depth = 0u;
  uint64_t stage_chunk_bytes = detail::k_default_stream_chunk_bytes;
};

struct bind_window {
  const bind_window_request &request;
  emel::callback<void(const events::bind_window_done &)> on_done = {};
  emel::callback<void(const events::bind_window_error &)> on_error = {};

  explicit bind_window(const bind_window_request &request_in) noexcept
      : request(request_in) {}
};

// Hot-path acquire: publishes the slot view for layer_index, suspending the
// dispatch on the already-submitted slot load when the layer is still in
// flight, and advancing the ring (evict oldest + prefetch ahead) on success.
struct acquire_layer_window {
  int32_t layer_index = -1;
  emel::callback<void(const events::acquire_layer_window_done &)> on_done = {};
  emel::callback<void(const events::acquire_layer_window_error &)> on_error = {};

  explicit acquire_layer_window(int32_t layer_index_in) noexcept
      : layer_index(layer_index_in) {}
};

struct unbind_window {
  emel::callback<void(const events::unbind_window_done &)> on_done = {};
  emel::callback<void(const events::unbind_window_error &)> on_error = {};
};

} // namespace emel::model::tensor::window::event

namespace emel::model::tensor::window::events {

struct bind_window_done {
  const event::bind_window &request;
  bool streaming_active = false;
  // Whole-file mapping the caller may use for pinned tensors / loader binding.
  const void *source_base = nullptr;
  uint64_t source_bytes = 0u;
  uint32_t window_slots = 0u;
};

struct bind_window_error {
  const event::bind_window &request;
  emel::error::type err = emel::error::cast(error::none);
};

struct acquire_layer_window_done {
  const event::acquire_layer_window &request;
  const uint8_t *slot_base = nullptr;
  const detail::layer_descriptor &layout;
};

struct acquire_layer_window_error {
  const event::acquire_layer_window &request;
  emel::error::type err = emel::error::cast(error::none);
};

struct unbind_window_done {
  const event::unbind_window &request;
};

struct unbind_window_error {
  const event::unbind_window &request;
  emel::error::type err = emel::error::cast(error::none);
};

} // namespace emel::model::tensor::window::events
