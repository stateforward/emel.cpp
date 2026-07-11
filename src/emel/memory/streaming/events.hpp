#pragma once

#include <cstdint>

namespace emel::memory::streaming {

// Half-open logical window [logical_begin, logical_end). physical_begin is
// the ring slot holding logical_begin; next_physical_position is the slot the
// next successful advance will overwrite. These values are derived at the
// event boundary and are never mirrored into persistent actor context.
struct window_view {
  int64_t logical_begin = 0;
  int64_t logical_end = 0;
  int32_t physical_begin = 0;
  int32_t next_physical_position = 0;
  int32_t valid_positions = 0;
  int32_t capacity = 0;
};

struct advance_result {
  int64_t logical_position = -1;
  int32_t physical_position = -1;
  window_view window = {};
};

namespace event {

struct initialize {
  int32_t &error_out;
};

struct advance {
  advance_result &result;
  int32_t &error_out;
};

struct reset {
  int32_t &error_out;
};

struct capture_view {
  window_view &view_out;
  int32_t &error_out;
};

} // namespace event

} // namespace emel::memory::streaming
