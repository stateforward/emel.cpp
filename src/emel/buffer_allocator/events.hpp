#pragma once

#include <cstdint>

namespace emel::buffer_allocator::event {

struct allocate_buffer {
  const void * request = nullptr;
};

struct upload_bytes {
  const void * handle = nullptr;
  const void * bytes = nullptr;
  uint64_t nbytes = 0;
};

}  // namespace emel::buffer_allocator::event

namespace emel::buffer_allocator::events {

struct allocation_done {};
struct allocation_error {};
struct upload_done {};
struct upload_error {};

}  // namespace emel::buffer_allocator::events
