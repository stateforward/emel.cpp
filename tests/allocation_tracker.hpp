#pragma once

#include <atomic>
#include <cstddef>

namespace emel::test::allocation {

inline std::atomic<bool> g_track_allocations = false;
inline std::atomic<size_t> g_allocation_count = 0u;

struct allocation_scope {
  allocation_scope() noexcept {
    g_allocation_count.store(0u, std::memory_order_relaxed);
    g_track_allocations.store(true, std::memory_order_relaxed);
  }

  ~allocation_scope() {
    g_track_allocations.store(false, std::memory_order_relaxed);
  }

  size_t allocations() const noexcept {
    return g_allocation_count.load(std::memory_order_relaxed);
  }
};

}  // namespace emel::test::allocation
