#pragma once

#include <cstdint>
#include <memory>

#include "emel/kernel/metal/detail.hpp"

namespace emel::kernel::metal::detail {

// Opaque Metal runtime (device, command queue, compiled library, pipeline
// states, staging buffer pool). Constructed once per actor; nullptr when the
// host has no Metal device or this translation unit cannot link Metal.
class metal_runtime;

} // namespace emel::kernel::metal::detail

namespace emel::kernel::metal::action {

struct context {
  // True when a Metal device was found and the kernel library compiled.
  // Guards route every dispatch to the explicit reject rows when false, so
  // the actor never falls back to another backend silently.
  bool metal_available = false;
  std::unique_ptr<detail::metal_runtime> runtime = {};

  context() {
    // One-time construction cost: device probe, MSL compile, pipeline
    // creation, and the fixed staging pool happen here, never during
    // dispatch.
    runtime = std::make_unique<detail::metal_runtime>();
    metal_available = runtime->available();
  }

  context(const context &) = delete;
  context &operator=(const context &) = delete;
  context(context &&) = delete;
  context &operator=(context &&) = delete;

  // TODO(emel): remove once dispatch observability no longer relies on this
  // counter.
  uint64_t dispatch_generation = 0;
  uint64_t metal_dispatch_count = 0;
};

} // namespace emel::kernel::metal::action
