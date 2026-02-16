#include <array>
#include <cstddef>
#include <new>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/sm.hpp"

namespace {

struct dummy_event {
  int32_t * error_out = nullptr;
};

struct owner_probe {
  int32_t calls = 0;

  bool process_event(const dummy_event &) noexcept {
    calls += 1;
    return true;
  }
};

}  // namespace

TEST_CASE("sm_normalize_event_result_handles_error_out") {
  int32_t err = EMEL_OK;
  dummy_event ok{.error_out = &err};
  CHECK_FALSE(emel::detail::normalize_event_result(ok, false));
  CHECK(emel::detail::normalize_event_result(ok, true));

  err = EMEL_ERR_BACKEND;
  CHECK_FALSE(emel::detail::normalize_event_result(ok, true));

  struct no_error_event {};
  CHECK(emel::detail::normalize_event_result(no_error_event{}, true));
}

TEST_CASE("sm_process_support_dispatches_events") {
  using process_t = boost::sml::back::process<dummy_event>;

  owner_probe owner{};
  emel::detail::process_support<owner_probe, process_t> support{&owner};
  support.queue_.push(dummy_event{});
  CHECK(owner.calls == 1);

  emel::detail::process_support<owner_probe, process_t> no_owner{nullptr};
  no_owner.queue_.push(dummy_event{});
  CHECK(owner.calls == 1);
}

TEST_CASE("sm_fifo_scheduler_immediate_and_nested_tasks") {
  emel::policy::fifo_scheduler<4, 64> scheduler;

  int32_t count = 0;
  CHECK(scheduler.try_run_immediate([&] { count += 1; }));
  CHECK(count == 1);

  std::array<int32_t, 3> seq = {{0, 0, 0}};
  int32_t idx = 0;
  scheduler.schedule([&] {
    seq[idx++] = 1;
    CHECK_FALSE(scheduler.try_run_immediate([&] { seq[idx++] = 99; }));
    scheduler.schedule([&] { seq[idx++] = 3; });
    seq[idx++] = 2;
  });

  CHECK(idx == 3);
  CHECK(seq[0] == 1);
  CHECK(seq[1] == 2);
  CHECK(seq[2] == 3);
}

TEST_CASE("sm_pooled_coroutine_allocator_uses_pool_and_heap") {
  using allocator_t = emel::policy::pooled_coroutine_allocator<64, 2>;

  allocator_t allocator{};
  void * p1 = allocator.allocate(32, alignof(std::max_align_t));
  void * p2 = allocator.allocate(32, alignof(std::max_align_t));
  void * p3 = allocator.allocate(32, alignof(std::max_align_t));

  CHECK(p1 != nullptr);
  CHECK(p2 != nullptr);
  CHECK(p3 != nullptr);
  CHECK(p1 != p2);

  allocator.deallocate(nullptr, 0, alignof(std::max_align_t));
  allocator.deallocate(p1, 32, alignof(std::max_align_t));
  allocator.deallocate(p2, 32, alignof(std::max_align_t));
  allocator.deallocate(p3, 32, alignof(std::max_align_t));

  void * heap_ptr = ::operator new(16, std::align_val_t(alignof(std::max_align_t)));
  allocator.deallocate(heap_ptr, 16, alignof(std::max_align_t));

  const std::size_t aligned = alignof(std::max_align_t) * 2;
  void * aligned_ptr = allocator.allocate(16, aligned);
  allocator.deallocate(aligned_ptr, 16, aligned);
}
