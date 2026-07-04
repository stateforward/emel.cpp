#pragma once

#include <array>
#include <cstdint>
#include <string_view>

#include "emel/io/mmap/errors.hpp"

namespace emel::io::mmap::action {

struct unmap_result {
  bool unmap_base_released = false;
  bool os_resource_released = false;
  bool ok = false;
};

// Platform mapping boundary, injected at construction. Production defaults
// bind the OS mapping calls (owned exclusively by actions.cpp per the
// boundary audit); ports supply platform alternates and tests supply failing
// operations to drive the modeled failure routes through process_event. The
// struct is the portability wrapper the platform rules require - effects
// never call OS APIs directly. It selects no behavior: every route stays an
// explicit guard/transition; only the platform implementation is injected.
struct platform_ops {
  bool (*open)(std::string_view path, intptr_t &os_resource_out) noexcept =
      nullptr;
  bool (*file_size)(intptr_t os_resource,
                    uint64_t &file_size_bytes_out) noexcept = nullptr;
  bool (*map)(intptr_t os_resource, uint64_t file_offset, uint64_t byte_size,
              void *&base_out, uint64_t &mapped_bytes_out) noexcept = nullptr;
  unmap_result (*unmap)(intptr_t os_resource, void *base,
                        uint64_t mapped_bytes) noexcept = nullptr;
  void (*close)(intptr_t os_resource) noexcept = nullptr;
  bool (*advise_sequential)(void *base, uint64_t offset,
                            uint64_t length) noexcept = nullptr;
  bool (*advise_willneed)(void *base, uint64_t offset,
                          uint64_t length) noexcept = nullptr;
  bool (*advise_dontneed)(void *base, uint64_t offset,
                          uint64_t length) noexcept = nullptr;
};

// The production operations (defaulted into every context by the default
// constructor); exposed so an injector can override a single op and keep the
// rest.
platform_ops default_platform_ops() noexcept;

struct slot {
  bool in_use = false;
  int32_t tensor_id = -1;
  void *base = nullptr;
  uint64_t mapped_bytes = 0u;
  intptr_t os_resource = -1;
  uint64_t file_offset = 0u;
  uint64_t requested_bytes = 0u;
};

struct context {
  std::array<slot, k_max_mappings> slots{};
  std::array<uint32_t, k_max_mappings> free_stack{};
  uint32_t free_count = 0u;
  uint64_t required_offset_alignment = k_required_offset_alignment;
  platform_ops platform{};

  context() noexcept;
  explicit context(const platform_ops &platform_in) noexcept;
  ~context() noexcept;
};

} // namespace emel::io::mmap::action
