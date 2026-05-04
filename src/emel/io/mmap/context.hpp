#pragma once

#include <array>
#include <cstdint>

#include "emel/io/mmap/errors.hpp"

namespace emel::io::mmap::action {

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

  context() noexcept {
    for (uint32_t i = 0; i < k_max_mappings; ++i) {
      free_stack[i] = (k_max_mappings - 1u) - i;
    }
    free_count = k_max_mappings;
  }
};

} // namespace emel::io::mmap::action
