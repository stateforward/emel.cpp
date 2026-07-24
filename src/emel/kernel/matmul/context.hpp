#pragma once

#include <array>
#include <cstddef>
#include <exception>
#include <memory>
#include <new>

#include "emel/kernel/matmul/detail.hpp"
#include "emel/kernel/sm.hpp"

namespace emel::kernel::matmul::action {

inline constexpr size_t MAX_PARALLEL_LANES = 8u;

struct lane_storage {
  explicit lane_storage(const emel::kernel::kernel_kind kind) noexcept
      : kernels{emel::kernel::sm{kind}, emel::kernel::sm{kind},
                emel::kernel::sm{kind}, emel::kernel::sm{kind},
                emel::kernel::sm{kind}, emel::kernel::sm{kind},
                emel::kernel::sm{kind}, emel::kernel::sm{kind}} {}

  std::array<emel::kernel::sm, MAX_PARALLEL_LANES> kernels;
};

struct context {
  context() noexcept
      : context(execution_policy{
            .parallel_matmul_lanes = nullptr,
            .kernel_kind = emel::kernel::detect_host_kind(),
            .active_lanes = 1u,
            .mode = lane_mode::serial,
        }) {}

  explicit context(const execution_policy &policy) noexcept
      : parallel_matmul_lanes(policy.parallel_matmul_lanes),
        kernel_kind(policy.kernel_kind), active_lanes(policy.active_lanes),
        kernel(policy.kernel_kind),
        lanes(new (std::nothrow) lane_storage{policy.kernel_kind}) {
    // Lane actors are large (~67 KiB each). One construction-time allocation
    // keeps the owning actor portable on platforms with small thread stacks;
    // dispatch reuses this stable storage and never allocates.
    if (lanes == nullptr) {
      std::terminate();
    }
  }

  lane_pool *parallel_matmul_lanes = nullptr;
  emel::kernel::kernel_kind kernel_kind = emel::kernel::kernel_kind::x86_64;
  size_t active_lanes = 1u;
  emel::kernel::sm kernel = {};
  std::unique_ptr<lane_storage> lanes = {};
};

} // namespace emel::kernel::matmul::action
