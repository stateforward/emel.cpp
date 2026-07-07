#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <span>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/sm.hpp"

namespace emel::text::generator::matmul {

template <size_t worker_lanes, size_t inline_task_bytes, size_t idle_spin_budget>
using lane_pool =
    emel::policy::fork_join_lane_pool<worker_lanes, inline_task_bytes, idle_spin_budget>;

struct lane_task {
  void (*run)(void *) noexcept = nullptr;
  void * ctx = nullptr;

  void operator()() const noexcept {
    if (run != nullptr) {
      run(ctx);
    }
  }
};

class lane_join_group {
 public:
  lane_join_group() = default;

  lane_join_group(const lane_join_group &) = delete;
  lane_join_group & operator=(const lane_join_group &) = delete;

  ~lane_join_group() { reset(); }

  template <class group_type>
  group_type & get() noexcept {
    static_assert(sizeof(group_type) <= storage_bytes,
                  "matmul lane join group storage is too small");
    static_assert(alignof(group_type) <= alignof(std::max_align_t),
                  "matmul lane join group alignment is too large");
    if (self_ == nullptr) {
      self_ = new (storage.data()) group_type();
      wait_ = [](void * ptr) noexcept {
        return static_cast<group_type *>(ptr)->wait();
      };
      destroy_ = [](void * ptr) noexcept {
        static_cast<group_type *>(ptr)->~group_type();
      };
    }
    return *static_cast<group_type *>(self_);
  }

  bool wait() noexcept {
    return wait_ == nullptr || wait_(self_);
  }

  void reset() noexcept {
    if (destroy_ != nullptr) {
      destroy_(self_);
    }
    self_ = nullptr;
    wait_ = nullptr;
    destroy_ = nullptr;
  }

 private:
  static constexpr size_t storage_bytes = 64u;

  alignas(std::max_align_t) std::array<unsigned char, storage_bytes> storage = {};
  void * self_ = nullptr;
  bool (*wait_)(void *) noexcept = nullptr;
  void (*destroy_)(void *) noexcept = nullptr;
};

struct lane_pool_ref {
  void * pool = nullptr;
  size_t worker_lanes = 0u;
  size_t lane_capacity = 0u;
  bool (*submit)(void *, lane_join_group &, lane_task) noexcept = nullptr;
  bool (*is_current_thread_worker)(void *) noexcept = nullptr;

  bool valid() const noexcept {
    return pool != nullptr && submit != nullptr && worker_lanes > 0u &&
           lane_capacity == worker_lanes + 1u;
  }

  bool try_submit(lane_join_group & group, lane_task task) const noexcept {
    return submit != nullptr && submit(pool, group, task);
  }
};

template <class pool_type>
lane_pool_ref make_lane_pool_ref(pool_type & pool) noexcept {
  using group_type = typename pool_type::join_group;
  return lane_pool_ref{
      .pool = &pool,
      .worker_lanes = pool_type::static_worker_count,
      .lane_capacity = pool_type::static_worker_count + 1u,
      .submit =
          [](void * pool_ptr, lane_join_group & group, lane_task task) noexcept {
            auto & typed_group = group.get<group_type>();
            return static_cast<pool_type *>(pool_ptr)->try_submit(typed_group, task);
          },
      .is_current_thread_worker =
          [](void * pool_ptr) noexcept {
            return static_cast<pool_type *>(pool_ptr)->is_current_thread_worker();
          },
  };
}

struct execution_policy {
  lane_pool_ref parallel_matmul_lanes;
  emel::kernel::kernel_kind kernel_kind;
  size_t active_lanes;
};

template <class pool_type>
inline execution_policy make_execution_policy(pool_type & parallel_matmul_lanes,
                                              const emel::kernel::kernel_kind kernel_kind,
                                              const size_t active_lanes) noexcept {
  const lane_pool_ref pool_ref = make_lane_pool_ref(parallel_matmul_lanes);
  return execution_policy{
      .parallel_matmul_lanes = pool_ref,
      .kernel_kind = kernel_kind,
      .active_lanes = active_lanes,
  };
}

template <class pool_type>
inline execution_policy make_auto_execution_policy(pool_type & parallel_matmul_lanes) noexcept {
  const lane_pool_ref pool_ref = make_lane_pool_ref(parallel_matmul_lanes);
  return execution_policy{
      .parallel_matmul_lanes = pool_ref,
      .kernel_kind = emel::kernel::detect_host_kind(),
      .active_lanes = pool_ref.lane_capacity,
  };
}

enum class lane_mode : uint8_t {
  serial = 0,
  parallel = 1,
};

namespace detail {

struct matmul_row_slice {
  int32_t row_begin = 0;
  int32_t row_count = 0;
};

inline size_t compute_matmul_row_slices(
    const uint64_t rows,
    const uint64_t group_rows,
    const size_t active_lanes,
    std::span<matmul_row_slice> slices) noexcept {
  const uint64_t groups = (rows + group_rows - 1u) / group_rows;
  const uint64_t lane_capacity =
      std::min<uint64_t>(slices.size(), std::max<size_t>(active_lanes, 1u));
  const uint64_t lane_count = std::min<uint64_t>(lane_capacity, std::max<uint64_t>(groups, 1u));
  const uint64_t groups_per_lane = groups / lane_count;
  const uint64_t extra_groups = groups % lane_count;
  uint64_t begin_group = 0u;
  for (uint64_t lane = 0u; lane < lane_count; ++lane) {
    const uint64_t lane_groups = groups_per_lane + static_cast<uint64_t>(lane < extra_groups);
    const uint64_t begin_row = begin_group * group_rows;
    const uint64_t end_row = std::min(rows, (begin_group + lane_groups) * group_rows);
    slices[lane].row_begin = static_cast<int32_t>(begin_row);
    slices[lane].row_count = static_cast<int32_t>(end_row - begin_row);
    begin_group += lane_groups;
  }
  return static_cast<size_t>(lane_count);
}

inline emel::kernel::event::op_mul_mat compute_sliced_mul_mat_event(
    const emel::kernel::event::op_mul_mat & ev,
    const uint64_t group_rows,
    const matmul_row_slice slice) noexcept {
  emel::kernel::event::op_mul_mat sliced = ev;
  const uint64_t begin = static_cast<uint64_t>(slice.row_begin);
  const uint64_t count = static_cast<uint64_t>(slice.row_count);
  const uint64_t slice_groups = (count + group_rows - 1u) / group_rows;
  sliced.src0.data =
      static_cast<const uint8_t *>(ev.src0.data) + (begin / group_rows) * ev.src0.nb[1];
  sliced.src0.ne[1] = count;
  sliced.src0.nb[2] = ev.src0.nb[1] * slice_groups;
  sliced.src0.nb[3] = sliced.src0.nb[2];
  sliced.dst.data = static_cast<uint8_t *>(ev.dst.data) + begin * ev.dst.nb[1];
  sliced.dst.ne[1] = count;
  sliced.dst.nb[2] = ev.dst.nb[1] * count;
  sliced.dst.nb[3] = sliced.dst.nb[2];
  return sliced;
}

}  // namespace detail

}  // namespace emel::text::generator::matmul
