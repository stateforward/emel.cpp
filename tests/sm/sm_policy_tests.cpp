#include <doctest/doctest.h>
#include <stateforward/sml/utility/dispatch_table.hpp>

#include <array>
#include <atomic>
#include <semaphore>
#include <thread>

#include "../allocation_tracker.hpp"
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

struct runtime_event {
  int id = 0;
  int32_t & marker;
};

struct event_one {
  static constexpr auto id = 1;

  int32_t & marker;

  explicit event_one(const runtime_event & ev) noexcept
    : marker(ev.marker) {}
};

struct event_two {
  static constexpr auto id = 2;

  int32_t & marker;

  explicit event_two(const runtime_event & ev) noexcept
    : marker(ev.marker) {}
};

struct state_idle {};
struct state_seen_one {};

struct effect_mark_one {
  void operator()(const event_one & ev) const noexcept {
    ev.marker = 1;
  }
};

struct effect_mark_two {
  void operator()(const event_two & ev) const noexcept {
    ev.marker = 2;
  }
};

struct dispatch_surface_model {
  auto operator()() const noexcept {
    namespace sml = stateforward::sml;
    // clang-format off
    return sml::make_transition_table(
        sml::state<state_seen_one> <= *sml::state<state_idle>
          + sml::event<event_one> / effect_mark_one{}
      , sml::X <= sml::state<state_seen_one>
          + sml::event<event_two> / effect_mark_two{}
    );
    // clang-format on
  }
};

struct logger_event {
  int32_t & marker;
};

struct logger_counters {
  int32_t processed = 0;
  int32_t guards = 0;
  int32_t actions = 0;
  int32_t state_changes = 0;

  template <class SM, class TEvent>
  void log_process_event(const TEvent &) noexcept {
    ++processed;
  }

  template <class SM, class TGuard, class TEvent>
  void log_guard(const TGuard &, const TEvent &, bool) noexcept {
    ++guards;
  }

  template <class SM, class TAction, class TEvent>
  void log_action(const TAction &, const TEvent &) noexcept {
    ++actions;
  }

  template <class SM, class TSrcState, class TDstState>
  void log_state_change(const TSrcState &, const TDstState &) noexcept {
    ++state_changes;
  }
};

struct guard_accept {
  bool operator()() const noexcept {
    return true;
  }
};

struct effect_mark_logged {
  void operator()(const logger_event & ev) const noexcept {
    ev.marker = 7;
  }
};

struct state_logger_idle {};
struct state_logger_done {};

struct logger_surface_model {
  auto operator()() const noexcept {
    namespace sml = stateforward::sml;
    // clang-format off
    return sml::make_transition_table(
        sml::state<state_logger_done> <= *sml::state<state_logger_idle>
          + sml::event<logger_event> [ guard_accept{} ] / effect_mark_logged{}
      , sml::state<state_logger_idle> <= sml::state<state_logger_idle>
          + sml::unexpected_event<sml::_>
    );
    // clang-format on
  }
};

using co_inline_policy =
    emel::policy::coroutine_scheduler<emel::policy::inline_scheduler>;
using co_static_policy =
    emel::policy::coroutine_scheduler<emel::policy::fifo_scheduler<8u, 64u>>;
using co_thread_pool_pool =
    emel::policy::thread_pool_scheduler<2u, 8u, 128u>;
using co_thread_pool_scheduler =
    emel::policy::thread_pool_scheduler_ref<co_thread_pool_pool>;
using co_thread_pool_policy =
    emel::policy::coroutine_scheduler<co_thread_pool_scheduler>;
using allocation_scope = emel::test::allocation::allocation_scope;

struct event_co_mark {
  int32_t & marker;
};

struct event_co_error {
  int32_t * error_out = nullptr;
};

struct event_co_wait {
  std::atomic<bool> & release;
  std::atomic<int32_t> & entered;
};

struct state_co_idle {};
struct state_co_done {};
struct state_co_error {};

struct effect_co_mark {
  void operator()(const event_co_mark & ev) const noexcept {
    ev.marker = 11;
  }
};

struct effect_co_error {
  void operator()(const event_co_error & ev) const noexcept {
    *ev.error_out = 3;
  }
};

struct effect_co_wait {
  void operator()(const event_co_wait & ev) const noexcept {
    ev.entered.fetch_add(1, std::memory_order_release);
    while (!ev.release.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }
};

struct co_surface_model {
  auto operator()() const noexcept {
    namespace sml = stateforward::sml;
    // clang-format off
    return sml::make_transition_table(
        sml::state<state_co_done> <= *sml::state<state_co_idle>
          + sml::event<event_co_mark> / effect_co_mark{}
      , sml::state<state_co_done> <= sml::state<state_co_idle>
          + sml::event<event_co_wait> / effect_co_wait{}
      , sml::state<state_co_error> <= sml::state<state_co_idle>
          + sml::event<event_co_error> / effect_co_error{}
    );
    // clang-format on
  }
};

struct context_co_probe {
  int32_t value = 0;
};

struct event_co_context_mark {
  int32_t & marker;
};

struct state_co_context_idle {};
struct state_co_context_done {};

struct effect_co_context_mark {
  void operator()(const event_co_context_mark & ev,
                  context_co_probe & ctx) const noexcept {
    ctx.value += 1;
    ev.marker = ctx.value;
  }
};

struct co_context_model {
  auto operator()() const noexcept {
    namespace sml = stateforward::sml;
    // clang-format off
    return sml::make_transition_table(
        sml::state<state_co_context_done> <= *sml::state<state_co_context_idle>
          + sml::event<event_co_context_mark> / effect_co_context_mark{}
    );
    // clang-format on
  }
};

template <class predicate>
void require_eventually(const char * label, predicate && pred) {
  for (int32_t attempt = 0; attempt < 100000; ++attempt) {
    if (pred()) {
      return;
    }
    std::this_thread::yield();
  }
  FAIL(label);
}

template <class scheduler>
concept has_schedule_method = requires(scheduler & scheduler_in) {
  scheduler_in.schedule([]() noexcept {});
};

}  // namespace

TEST_CASE("sm_normalize_event_result_handles_error_out") {
  int32_t err = 0;
  dummy_event ok{.error_out = &err};
  CHECK_FALSE(emel::detail::normalize_event_result(ok, false));
  CHECK(emel::detail::normalize_event_result(ok, true));

  err = (1 << 1);
  CHECK_FALSE(emel::detail::normalize_event_result(ok, true));

  struct no_error_event {};
  CHECK(emel::detail::normalize_event_result(no_error_event{}, true));
}

TEST_CASE("sm_process_support_dispatches_events") {
  using process_t = stateforward::sml::back::process<dummy_event>;

  owner_probe owner{};
  emel::detail::process_support<owner_probe, process_t> support{&owner};
  support.queue_.push(dummy_event{});
  CHECK(owner.calls == 1);

  emel::detail::process_support<owner_probe, process_t> no_owner{nullptr};
  no_owner.queue_.push(dummy_event{});
  CHECK(owner.calls == 1);
}

TEST_CASE("stateforward_sml_dispatch_table_routes_runtime_events") {
  namespace sml = stateforward::sml;
  sml::sm<dispatch_surface_model> machine{};
  auto dispatch_event = sml::utility::make_dispatch_table<runtime_event, 1, 2>(machine);

  int32_t marker = 0;
  CHECK(dispatch_event(runtime_event{.id = 1, .marker = marker}, 1));
  CHECK(marker == 1);
  CHECK(machine.is(sml::state<state_seen_one>));

  CHECK_FALSE(dispatch_event(runtime_event{.id = 99, .marker = marker}, 99));
  CHECK(marker == 1);
  CHECK(machine.is(sml::state<state_seen_one>));

  CHECK(dispatch_event(runtime_event{.id = 2, .marker = marker}, 2));
  CHECK(marker == 2);
  CHECK(machine.is(sml::X));
}

TEST_CASE("stateforward_sml_logger_policy_observes_dispatch") {
  namespace sml = stateforward::sml;
  logger_counters logger{};
  sml::sm<logger_surface_model, sml::logger<logger_counters>> machine{logger};

  int32_t marker = 0;
  CHECK(machine.process_event(logger_event{.marker = marker}));

  CHECK(marker == 7);
  CHECK(logger.processed >= 1);
  CHECK(logger.guards >= 1);
  CHECK(logger.actions >= 1);
  CHECK(logger.state_changes >= 1);
  CHECK(machine.is(sml::state<state_logger_done>));
}

TEST_CASE("co_sm_process_event_uses_stateforward_utility_surface") {
  using default_machine_type = emel::co_sm<co_surface_model>;
  using machine_type = emel::co_sm<co_surface_model, void, co_inline_policy>;
  static_assert(std::same_as<default_machine_type::scheduler_type,
                             emel::policy::inline_scheduler>);
  static_assert(std::same_as<machine_type::scheduler_type,
                             emel::policy::inline_scheduler>);
  static_assert(
      emel::policy::strict_ordering_scheduler_contract<
          machine_type::scheduler_type>);

  namespace sml = stateforward::sml;
  machine_type machine{};

  int32_t marker = 0;
  CHECK(machine.process_event(event_co_mark{.marker = marker}));
  CHECK(marker == 11);
  CHECK(machine.is(sml::state<state_co_done>));

  bool scheduled = false;
  machine.scheduler().schedule([&scheduled]() noexcept { scheduled = true; });
  CHECK(scheduled);
}

TEST_CASE("co_sm_process_event_async_inline_completes_immediately") {
  namespace sml = stateforward::sml;
  emel::co_sm<co_surface_model, void, co_inline_policy> machine{};

  int32_t marker = 0;
  emel::bool_task task =
      machine.process_event_async(event_co_mark{.marker = marker});

  CHECK(task.result());
  CHECK(marker == 11);
  CHECK(machine.is(sml::state<state_co_done>));
}

TEST_CASE("co_sm_normalizes_error_out_for_sync_and_inline_async") {
  emel::co_sm<co_surface_model, void, co_inline_policy> sync_machine{};
  int32_t sync_err = 0;
  CHECK_FALSE(sync_machine.process_event(event_co_error{.error_out = &sync_err}));
  CHECK(sync_err == 3);

  emel::co_sm<co_surface_model, void, co_inline_policy> async_machine{};
  int32_t async_err = 0;
  emel::bool_task task =
      async_machine.process_event_async(event_co_error{.error_out = &async_err});
  CHECK_FALSE(task.result());
  CHECK(async_err == 3);
}

TEST_CASE("co_sm_normalizes_error_out_for_static_scheduler_immediate_async") {
  emel::co_sm<co_surface_model, void, co_static_policy> machine{};
  int32_t err = 0;
  emel::bool_task task = machine.process_event_async(event_co_error{.error_out = &err});

  CHECK(task.await_ready());
  CHECK_FALSE(task.result());
  CHECK(err == 3);
}

TEST_CASE("co_sm_static_scheduler_rejects_busy_async_without_escaping_rtc") {
  namespace sml = stateforward::sml;
  emel::co_sm<co_surface_model, void, co_static_policy> machine{};

  int32_t marker = 0;
  bool task_ready = false;
  bool task_result = true;
  const bool drained = machine.scheduler().try_run_immediate([&]() {
    emel::bool_task task =
        machine.process_event_async(event_co_mark{.marker = marker});
    task_ready = task.await_ready();
    task_result = task.result();
  });

  CHECK(drained);
  CHECK(task_ready);
  CHECK_FALSE(task_result);
  CHECK(marker == 0);
  CHECK_FALSE(machine.is(sml::state<state_co_done>));
}

TEST_CASE("thread_pool_scheduler_policy_is_static_multi_consumer") {
  using scheduler_type = co_thread_pool_policy::scheduler_type;
  static_assert(scheduler_type::multi_consumer);
  static_assert(!scheduler_type::single_consumer);
  static_assert(!scheduler_type::owns_workers);
  static_assert(scheduler_type::run_to_completion);
  static_assert(scheduler_type::static_worker_count == 2u);
  static_assert(scheduler_type::static_capacity == 8u);
  static_assert(std::is_copy_constructible_v<scheduler_type>);
  static_assert(!co_thread_pool_pool::run_to_completion);
  static_assert(!has_schedule_method<co_thread_pool_pool>);
  static_assert(
      !stateforward::sml::utility::policy::valid_coroutine_scheduler<
          co_thread_pool_pool>);

  co_thread_pool_pool pool{};
  emel::co_sm<co_surface_model, void, co_thread_pool_policy> machine{
      co_thread_pool_scheduler{pool}};
  CHECK(machine.scheduler().immediate_run_count() == 0u);
  CHECK(machine.scheduler().scheduled_run_count() == 0u);
  CHECK(machine.scheduler().worker_run_count() == 0u);
}

TEST_CASE("co_sm_thread_pool_scheduler_runs_inline_when_idle") {
  namespace sml = stateforward::sml;
  co_thread_pool_pool pool{};
  emel::co_sm<co_surface_model, void, co_thread_pool_policy> machine{
      co_thread_pool_scheduler{pool}};

  int32_t marker = 0;
  emel::bool_task task =
      machine.process_event_async(event_co_mark{.marker = marker});

  CHECK(task.await_ready());
  CHECK(task.result());
  CHECK(marker == 11);
  CHECK(machine.is(sml::state<state_co_done>));
  CHECK(machine.scheduler().immediate_run_count() == 1u);
  CHECK(machine.scheduler().scheduled_run_count() == 0u);
  CHECK(machine.scheduler().worker_run_count() == 0u);
}

TEST_CASE("co_sm_thread_pool_scheduler_uses_worker_when_inline_busy") {
  namespace sml = stateforward::sml;
  co_thread_pool_pool pool{};
  emel::co_sm<co_surface_model, void, co_thread_pool_policy> machine{
      co_thread_pool_scheduler{pool}};

  int32_t marker = 0;
  bool task_ready = false;
  bool task_result = false;
  const bool outer_completed = pool.try_run_immediate([&]() {
    emel::bool_task task =
        machine.process_event_async(event_co_mark{.marker = marker});
    task_ready = task.await_ready();
    task_result = task.result();
  });

  CHECK(outer_completed);
  CHECK(task_ready);
  CHECK(task_result);
  CHECK(marker == 11);
  CHECK(machine.is(sml::state<state_co_done>));
  CHECK(machine.scheduler().immediate_run_count() == 1u);
  CHECK(machine.scheduler().scheduled_run_count() == 1u);
  CHECK(machine.scheduler().worker_run_count() == 1u);
}

TEST_CASE("co_sm_thread_pool_scheduler_worker_result_waits_for_scheduler_quiescence") {
  co_thread_pool_pool pool{};
  emel::co_sm<co_surface_model, void, co_thread_pool_policy> first{
      co_thread_pool_scheduler{pool}};
  emel::co_sm<co_surface_model, void, co_thread_pool_policy> second{
      co_thread_pool_scheduler{pool}};
  std::binary_semaphore inline_lane_held{0};
  std::binary_semaphore release_inline_lane{0};
  std::atomic<bool> holder_result{false};
  std::thread inline_lane_holder{[&]() {
    const bool held = pool.try_run_immediate([&]() noexcept {
      holder_result.store(true, std::memory_order_release);
      inline_lane_held.release();
      release_inline_lane.acquire();
    });
    if (!held) {
      inline_lane_held.release();
    }
  }};
  inline_lane_held.acquire();
  CHECK(holder_result.load(std::memory_order_acquire));

  int32_t first_marker = 0;
  emel::bool_task first_task =
      first.process_event_async(event_co_mark{.marker = first_marker});
  CHECK(first_task.result());
  CHECK(first_marker == 11);
  CHECK(pool.worker_run_count() == 1u);
  CHECK(pool.scheduled_run_count() == 1u);

  release_inline_lane.release();
  inline_lane_holder.join();

  const uint64_t immediate_before_second = pool.immediate_run_count();
  const uint64_t scheduled_before_second = pool.scheduled_run_count();
  int32_t second_marker = 0;
  emel::bool_task second_task =
      second.process_event_async(event_co_mark{.marker = second_marker});
  CHECK(second_task.result());
  CHECK(second_marker == 11);
  CHECK(pool.immediate_run_count() == immediate_before_second + 1u);
  CHECK(pool.scheduled_run_count() == scheduled_before_second);
}

TEST_CASE("co_sm_thread_pool_scheduler_rejects_concurrent_actor_dispatch") {
  co_thread_pool_pool pool{};
  emel::co_sm<co_surface_model, void, co_thread_pool_policy> machine{
      co_thread_pool_scheduler{pool}};
  std::atomic<bool> release{false};
  std::atomic<int32_t> entered{0};
  bool first_result = false;

  std::thread first_dispatch([&]() {
    first_result = machine
                       .process_event_async(event_co_wait{
                           .release = release,
                           .entered = entered,
                       })
                       .result();
  });

  require_eventually("first dispatch did not enter action", [&]() {
    return entered.load(std::memory_order_acquire) == 1;
  });

  int32_t marker = 0;
  emel::bool_task second_task =
      machine.process_event_async(event_co_mark{.marker = marker});
  CHECK(second_task.await_ready());
  CHECK_FALSE(second_task.result());
  CHECK(marker == 0);

  release.store(true, std::memory_order_release);
  first_dispatch.join();
  CHECK(first_result);
}

TEST_CASE("co_sm_thread_pool_scheduler_rejects_concurrent_state_inspection") {
  namespace sml = stateforward::sml;
  co_thread_pool_pool pool{};
  emel::co_sm<co_surface_model, void, co_thread_pool_policy> machine{
      co_thread_pool_scheduler{pool}};
  std::atomic<bool> release{false};
  std::atomic<int32_t> entered{0};
  bool first_result = false;
  bool visited = false;

  std::thread first_dispatch([&]() {
    first_result = machine
                       .process_event_async(event_co_wait{
                           .release = release,
                           .entered = entered,
                       })
                       .result();
  });

  require_eventually("first dispatch did not enter action", [&]() {
    return entered.load(std::memory_order_acquire) == 1;
  });

  CHECK_FALSE(machine.is(sml::state<state_co_done>));
  machine.visit_current_states([&](const auto &) { visited = true; });
  CHECK_FALSE(visited);

  release.store(true, std::memory_order_release);
  first_dispatch.join();
  CHECK(first_result);
}

TEST_CASE("co_sm_thread_pool_scheduler_allows_concurrent_different_actors") {
  co_thread_pool_pool pool{};
  emel::co_sm<co_surface_model, void, co_thread_pool_policy> first{
      co_thread_pool_scheduler{pool}};
  emel::co_sm<co_surface_model, void, co_thread_pool_policy> second{
      co_thread_pool_scheduler{pool}};
  int32_t first_marker = 0;
  int32_t second_marker = 0;
  bool first_result = false;
  bool second_result = false;

  std::thread first_thread([&]() {
    first_result = first.process_event_async(
                            event_co_mark{.marker = first_marker})
                       .result();
  });
  std::thread second_thread([&]() {
    second_result = second.process_event_async(
                              event_co_mark{.marker = second_marker})
                        .result();
  });

  first_thread.join();
  second_thread.join();

  CHECK(first_result);
  CHECK(second_result);
  CHECK(first_marker == 11);
  CHECK(second_marker == 11);
}

TEST_CASE("thread_pool_scheduler_rejects_same_pool_nested_wait") {
  using scheduler_type = emel::policy::thread_pool_scheduler<1u, 8u, 128u>;
  scheduler_type scheduler{};
  std::atomic<bool> outer_entered{false};
  std::atomic<bool> nested_ran{false};
  std::atomic<bool> nested_completed{true};
  std::atomic<bool> done{false};

  scheduler.submit([&]() noexcept {
    outer_entered.store(true, std::memory_order_release);
    nested_completed.store(
        scheduler.run_or_schedule_and_wait([&]() noexcept {
          nested_ran.store(true, std::memory_order_release);
        }),
        std::memory_order_release);
    done.store(true, std::memory_order_release);
  });

  require_eventually("nested scheduler test did not finish", [&]() {
    return done.load(std::memory_order_acquire);
  });

  CHECK(outer_entered.load(std::memory_order_acquire));
  CHECK_FALSE(nested_completed.load(std::memory_order_acquire));
  CHECK_FALSE(nested_ran.load(std::memory_order_acquire));
}

TEST_CASE("thread_pool_scheduler_ref_fork_join_runs_submitted_tasks_before_wait_returns") {
  using scheduler_type = emel::policy::thread_pool_scheduler<2u, 8u, 128u>;
  using scheduler_ref = emel::policy::thread_pool_scheduler_ref<scheduler_type>;
  scheduler_type scheduler{};
  scheduler_ref ref{scheduler};
  scheduler_ref::join_group group{};
  std::atomic<int32_t> entered{0};
  std::atomic<int32_t> finished{0};
  std::atomic<bool> release{false};

  CHECK(ref.try_submit(group, [&]() noexcept {
    entered.fetch_add(1, std::memory_order_release);
    while (!release.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    finished.fetch_add(1, std::memory_order_release);
  }));
  CHECK(ref.try_submit(group, [&]() noexcept {
    entered.fetch_add(1, std::memory_order_release);
    while (!release.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    finished.fetch_add(1, std::memory_order_release);
  }));

  require_eventually("fork/join tasks did not enter concurrently", [&]() {
    return entered.load(std::memory_order_acquire) == 2;
  });

  release.store(true, std::memory_order_release);
  CHECK(group.wait());
  CHECK(finished.load(std::memory_order_acquire) == 2);
  CHECK(scheduler.worker_run_count() == 2u);
}

TEST_CASE("thread_pool_scheduler_ref_fork_join_ignores_pre_wait_completion_tokens") {
  using scheduler_type = emel::policy::thread_pool_scheduler<1u, 8u, 128u>;
  using scheduler_ref = emel::policy::thread_pool_scheduler_ref<scheduler_type>;
  scheduler_type scheduler{};
  scheduler_ref ref{scheduler};
  scheduler_ref::join_group group{};
  std::atomic<bool> fast_done{false};
  std::atomic<bool> blocking_entered{false};
  std::atomic<bool> release_blocking{false};
  std::atomic<bool> wait_returned{false};
  bool wait_result = false;

  CHECK(ref.try_submit(group, [&]() noexcept {
    fast_done.store(true, std::memory_order_release);
  }));
  require_eventually("fast fork/join task did not finish", [&]() {
    return fast_done.load(std::memory_order_acquire);
  });

  CHECK(ref.try_submit(group, [&]() noexcept {
    blocking_entered.store(true, std::memory_order_release);
    while (!release_blocking.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }));
  require_eventually("blocking fork/join task did not enter", [&]() {
    return blocking_entered.load(std::memory_order_acquire);
  });

  std::thread waiter{[&]() {
    wait_result = group.wait();
    wait_returned.store(true, std::memory_order_release);
  }};

  for (int32_t attempt = 0; attempt < 1000; ++attempt) {
    if (wait_returned.load(std::memory_order_acquire)) {
      break;
    }
    std::this_thread::yield();
  }
  CHECK_FALSE(wait_returned.load(std::memory_order_acquire));

  release_blocking.store(true, std::memory_order_release);
  waiter.join();
  CHECK(wait_result);
  CHECK(wait_returned.load(std::memory_order_acquire));
}

TEST_CASE("thread_pool_scheduler_ref_fork_join_rejects_same_pool_worker_submit") {
  using scheduler_type = emel::policy::thread_pool_scheduler<1u, 8u, 128u>;
  using scheduler_ref = emel::policy::thread_pool_scheduler_ref<scheduler_type>;
  scheduler_type scheduler{};
  scheduler_ref ref{scheduler};
  std::atomic<bool> nested_submitted{true};
  std::atomic<bool> nested_joined{true};
  std::atomic<bool> done{false};

  scheduler.submit([&]() noexcept {
    scheduler_ref::join_group group{};
    nested_submitted.store(
        ref.try_submit(group, []() noexcept {}), std::memory_order_release);
    nested_joined.store(group.wait(), std::memory_order_release);
    done.store(true, std::memory_order_release);
  });

  require_eventually("same-pool fork/join rejection did not finish", [&]() {
    return done.load(std::memory_order_acquire);
  });

  CHECK_FALSE(nested_submitted.load(std::memory_order_acquire));
  CHECK_FALSE(nested_joined.load(std::memory_order_acquire));
}

TEST_CASE("thread_pool_scheduler_try_submit_reports_full_queue") {
  using scheduler_type = emel::policy::thread_pool_scheduler<1u, 2u, 128u>;
  scheduler_type scheduler{};
  std::atomic<bool> worker_entered{false};
  std::atomic<bool> release_worker{false};
  std::atomic<bool> queued_ran{false};

  CHECK(scheduler.try_submit([&]() noexcept {
    worker_entered.store(true, std::memory_order_release);
    while (!release_worker.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }));

  require_eventually("worker did not enter first task", [&]() {
    return worker_entered.load(std::memory_order_acquire);
  });

  CHECK(scheduler.try_submit([&]() noexcept {
    queued_ran.store(true, std::memory_order_release);
  }));
  CHECK_FALSE(scheduler.try_submit([]() noexcept {}));

  release_worker.store(true, std::memory_order_release);
  require_eventually("queued task did not run", [&]() {
    return queued_ran.load(std::memory_order_acquire);
  });
}

TEST_CASE("thread_pool_scheduler_accepts_multiple_producers") {
  using scheduler_type = emel::policy::thread_pool_scheduler<2u, 32u, 128u>;
  scheduler_type scheduler{};
  std::atomic<int32_t> completed{0};
  std::array<std::thread, 4> producers{};

  for (auto & producer : producers) {
    producer = std::thread([&scheduler, &completed]() {
      for (int32_t idx = 0; idx < 4; ++idx) {
        scheduler.submit([&completed]() noexcept {
          completed.fetch_add(1, std::memory_order_release);
        });
      }
    });
  }

  for (auto & producer : producers) {
    producer.join();
  }

  require_eventually("multi-producer tasks did not finish", [&]() {
    return completed.load(std::memory_order_acquire) == 16;
  });

  CHECK(scheduler.scheduled_run_count() == 16u);
  CHECK(scheduler.worker_run_count() == 16u);
}

TEST_CASE("thread_pool_scheduler_ref_fork_join_survives_rapid_repeated_rounds") {
  // Regression: the join latch previously used a closed_/pending_ handshake plus
  // a per-group binary_semaphore. The handshake had a Dekker-style StoreLoad
  // race (wait() could miss the final completion while the last completer missed
  // the close), and the semaphore could be destroyed by the returning waiter
  // while the completer was still inside release()/notify. Either could strand a
  // wakeup and deadlock wait() after many back-to-back rounds, most readily when
  // the lane count equals the worker count. Drive enough rounds that a
  // reintroduced race surfaces (a regression makes wait() hang here).
  using scheduler_type = emel::policy::thread_pool_scheduler<8u, 64u, 128u>;
  using scheduler_ref = emel::policy::thread_pool_scheduler_ref<scheduler_type>;
  scheduler_type scheduler{};
  constexpr int32_t k_rounds = 20000;
  constexpr int32_t k_lanes = 8;
  std::atomic<int64_t> ran{0};
  bool all_submitted = true;
  bool all_joined = true;

  for (int32_t round = 0; round < k_rounds; ++round) {
    scheduler_ref ref{scheduler};
    scheduler_ref::join_group group{};
    for (int32_t lane = 0; lane < k_lanes; ++lane) {
      all_submitted &= ref.try_submit(group, [&ran]() noexcept {
        ran.fetch_add(1, std::memory_order_relaxed);
      });
    }
    all_joined &= group.wait();
  }

  CHECK(all_submitted);
  CHECK(all_joined);
  CHECK(ran.load(std::memory_order_acquire) ==
        static_cast<int64_t>(k_rounds) * k_lanes);
  CHECK(scheduler.worker_run_count() ==
        static_cast<uint64_t>(k_rounds) * k_lanes);
}

TEST_CASE("co_sm_contextful_wrapper_injects_context") {
  namespace sml = stateforward::sml;
  context_co_probe ctx{.value = 40};
  emel::co_sm<co_context_model, context_co_probe, co_inline_policy> machine{ctx};

  int32_t marker = 0;
  CHECK(machine.process_event(event_co_context_mark{.marker = marker}));
  CHECK(marker == 41);
  CHECK(machine.is(sml::state<state_co_context_done>));
}

TEST_CASE("fixed_coroutine_allocator_has_no_heap_fallback") {
  emel::policy::fixed_coroutine_allocator<64, 1> allocator{};
  void * first = allocator.allocate(16, alignof(std::max_align_t));
  CHECK(first != nullptr);
  CHECK(allocator.allocate(16, alignof(std::max_align_t)) == nullptr);

  allocator.deallocate(first, 16, alignof(std::max_align_t));
  void * reused = allocator.allocate(16, alignof(std::max_align_t));
  CHECK(reused == first);
  allocator.deallocate(reused, 16, alignof(std::max_align_t));
}

TEST_CASE("co_sm_thread_pool_scheduler_async_dispatch_does_not_allocate") {
  {
    co_thread_pool_pool pool{};
    emel::co_sm<co_surface_model, void, co_thread_pool_policy> machine{
        co_thread_pool_scheduler{pool}};
    int32_t marker = 0;
    allocation_scope allocations{};

    emel::bool_task task =
        machine.process_event_async(event_co_mark{.marker = marker});

    CHECK(task.result());
    CHECK(marker == 11);
    CHECK(allocations.allocations() == 0u);
  }

  {
    co_thread_pool_pool pool{};
    emel::co_sm<co_surface_model, void, co_thread_pool_policy> machine{
        co_thread_pool_scheduler{pool}};
    std::binary_semaphore inline_lane_held{0};
    std::binary_semaphore release_inline_lane{0};
    std::atomic<bool> holder_result{false};
    std::thread inline_lane_holder{[&]() {
      const bool held = pool.try_run_immediate([&]() noexcept {
        holder_result.store(true, std::memory_order_release);
        inline_lane_held.release();
        release_inline_lane.acquire();
      });
      if (!held) {
        inline_lane_held.release();
      }
    }};
    inline_lane_held.acquire();
    CHECK(holder_result.load(std::memory_order_acquire));

    int32_t marker = 0;
    {
      allocation_scope allocations{};
      emel::bool_task task =
          machine.process_event_async(event_co_mark{.marker = marker});

      CHECK(task.result());
      CHECK(marker == 11);
      CHECK(allocations.allocations() == 0u);
    }

    release_inline_lane.release();
    inline_lane_holder.join();
  }
}
