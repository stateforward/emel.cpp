#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <doctest/doctest.h>

#include "emel/io/mmap/sm.hpp"
#include "emel/io/staged_read/sm.hpp"
#include "emel/model/tensor/window/events.hpp"
#include "emel/model/tensor/window/sm.hpp"

namespace emel_window_test {

namespace window = emel::model::tensor::window;

inline constexpr uint32_t k_layers = 6u;
inline constexpr uint32_t k_weights_per_layer = 2u;
inline constexpr uint64_t k_weight_bytes = 8192u;
inline constexpr uint64_t k_header_bytes = 4096u;

inline uint8_t sentinel_byte(const uint32_t layer, const uint32_t weight) {
  return static_cast<uint8_t>(0x11u + layer * 0x10u + weight);
}

// A synthetic "model file": k_layers layers of k_weights_per_layer weights,
// each filled with a per-(layer, weight) sentinel byte, after a header pad.
struct stream_file {
  std::filesystem::path path{};
  std::string path_str{};
  uint64_t file_size = 0u;
  std::vector<window::detail::weight_extent> extents{};
  std::vector<uint16_t> layer_weight_counts{};

  explicit stream_file(std::string_view tag) {
    path = std::filesystem::temp_directory_path() /
           (std::string{"emel_window_"} + std::string{tag} + ".bin");
    std::ofstream out{path, std::ios::binary | std::ios::trunc};
    REQUIRE(out.good());
    const std::vector<char> header(static_cast<size_t>(k_header_bytes), '\0');
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    uint64_t offset = k_header_bytes;
    for (uint32_t layer = 0; layer < k_layers; ++layer) {
      layer_weight_counts.push_back(static_cast<uint16_t>(k_weights_per_layer));
      for (uint32_t weight = 0; weight < k_weights_per_layer; ++weight) {
        const std::vector<char> payload(
            static_cast<size_t>(k_weight_bytes),
            static_cast<char>(sentinel_byte(layer, weight)));
        out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
        extents.push_back(window::detail::weight_extent{
            .tensor_id = static_cast<int32_t>(layer * k_weights_per_layer + weight),
            .file_offset = offset,
            .byte_size = k_weight_bytes,
            .slot_offset = 0u,
        });
        offset += k_weight_bytes;
      }
    }
    out.close();
    file_size = offset;
    path_str = path.string();
  }

  ~stream_file() { std::filesystem::remove(path); }
};

struct bind_capture {
  bool done = false;
  bool error = false;
  bool streaming_active = false;
  const void *source_base = nullptr;
  uint32_t window_slots = 0u;
  emel::error::type err = 0u;
};

inline void on_bind_done(void *object,
                         const window::events::bind_window_done &ev) noexcept {
  auto *capture = static_cast<bind_capture *>(object);
  capture->done = true;
  capture->streaming_active = ev.streaming_active;
  capture->source_base = ev.source_base;
  capture->window_slots = ev.window_slots;
}

inline void on_bind_error(void *object,
                          const window::events::bind_window_error &ev) noexcept {
  auto *capture = static_cast<bind_capture *>(object);
  capture->error = true;
  capture->err = ev.err;
}

struct acquire_capture {
  bool done = false;
  bool error = false;
  const uint8_t *slot_base = nullptr;
  const window::detail::layer_descriptor *layout = nullptr;
  emel::error::type err = 0u;
};

inline void on_acquire_done(
    void *object, const window::events::acquire_layer_window_done &ev) noexcept {
  auto *capture = static_cast<acquire_capture *>(object);
  capture->done = true;
  capture->slot_base = ev.slot_base;
  capture->layout = &ev.layout;
}

inline void on_acquire_error(
    void *object, const window::events::acquire_layer_window_error &ev) noexcept {
  auto *capture = static_cast<acquire_capture *>(object);
  capture->error = true;
  capture->err = ev.err;
}

struct unbind_capture {
  bool done = false;
  bool error = false;
};

inline void on_unbind_done(void *object,
                           const window::events::unbind_window_done &) noexcept {
  static_cast<unbind_capture *>(object)->done = true;
}

inline void on_unbind_error(void *object,
                            const window::events::unbind_window_error &) noexcept {
  static_cast<unbind_capture *>(object)->error = true;
}

struct window_fixture {
  emel::io::mmap::sm io_mmap{};
  std::array<emel::io::staged_read::sm, window::detail::k_max_window_slots>
      io_staged{};
  window::detail::stream_io_pool io_pool{};
  window::sm machine;

  window::action::context make_context() noexcept {
    window::action::context ctx{};
    ctx.io_mmap = &io_mmap;
    ctx.io_staged = io_staged;
    ctx.io_pool = &io_pool;
    return ctx;
  }

  window_fixture() : machine{make_context()} {}

  bool bind(const stream_file &file, bind_capture &capture,
            const uint64_t budget_bytes, const uint32_t slots = 4u,
            const uint32_t prefetch_depth = 2u) {
    const window::event::bind_window_request request{
        .file_path = file.path_str,
        .file_size_bytes = file.file_size,
        .extents = file.extents,
        .layer_weight_counts = file.layer_weight_counts,
        .budget_bytes = budget_bytes,
        .window_slots = slots,
        .prefetch_depth = prefetch_depth,
        .stage_chunk_bytes = window::detail::k_default_stream_chunk_bytes,
    };
    window::event::bind_window bind_request{request};
    bind_request.on_done = {&capture, on_bind_done};
    bind_request.on_error = {&capture, on_bind_error};
    return machine.process_event(bind_request);
  }

  bool acquire(const int32_t layer, acquire_capture &capture) {
    window::event::acquire_layer_window request{layer};
    request.on_done = {&capture, on_acquire_done};
    request.on_error = {&capture, on_acquire_error};
    return machine.process_event(request);
  }

  bool unbind(unbind_capture &capture) {
    window::event::unbind_window request{};
    request.on_done = {&capture, on_unbind_done};
    request.on_error = {&capture, on_unbind_error};
    return machine.process_event(request);
  }
};

// Budget that forces streaming for the fixture file while fitting 4 slots.
inline uint64_t streaming_budget() {
  const uint64_t layer_bytes = 2u * k_weight_bytes;  // aligned sizes are exact here
  return 5u * layer_bytes;  // < 6 layers total, >= 4 slots
}

inline bool slot_content_matches(const acquire_capture &capture,
                                 const uint32_t layer) {
  if (capture.slot_base == nullptr || capture.layout == nullptr) {
    return false;
  }
  for (uint32_t weight = 0; weight < k_weights_per_layer; ++weight) {
    const auto &extent = capture.layout->weights[weight];
    const uint8_t *bytes = capture.slot_base + extent.slot_offset;
    const uint8_t expected = sentinel_byte(layer, weight);
    for (uint64_t index = 0; index < extent.byte_size; index += 1024u) {
      if (bytes[index] != expected) {
        return false;
      }
    }
    if (bytes[extent.byte_size - 1u] != expected) {
      return false;
    }
  }
  return true;
}

}  // namespace emel_window_test
