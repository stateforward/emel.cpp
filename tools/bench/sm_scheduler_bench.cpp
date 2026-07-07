#include "bench_cases.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <semaphore>
#include <thread>

#include "emel/sm.hpp"

namespace {

bool bench_internal_enabled() {
  const char * value = std::getenv("EMEL_BENCH_INTERNAL");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return value[0] != '0';
}

struct event_tick {
  volatile int32_t & sink;
};

struct state_idle {};

struct effect_tick {
  void operator()(const event_tick & ev) const noexcept {
    ev.sink += 1;
  }
};

struct scheduler_model {
  auto operator()() const noexcept {
    namespace sml = stateforward::sml;
    // clang-format off
    return sml::make_transition_table(
        sml::state<state_idle> <= *sml::state<state_idle>
          + sml::event<event_tick> / effect_tick{}
    );
    // clang-format on
  }
};

using inline_co_policy =
    emel::policy::coroutine_scheduler<emel::policy::inline_scheduler>;
using thread_pool_pool =
    emel::policy::thread_pool_scheduler<2u, 1024u, 128u>;
using thread_pool_co_policy =
    emel::policy::coroutine_scheduler<thread_pool_pool>;

using inline_machine = emel::co_sm<scheduler_model, void, inline_co_policy>;
using thread_pool_machine =
    emel::co_sm<scheduler_model, void, thread_pool_co_policy>;

bool await_result(emel::bool_task & task) {
  while (!task.await_ready()) {
    emel::policy::cpu_relax();
  }
  return task.result();
}

emel::bench::result annotate_result(emel::bench::result result,
                                    const char * lane,
                                    const char * backend_id) {
  result.lane = lane;
  result.backend_id = backend_id;
  result.workload_id = "single_transition_event";
  result.comparison_mode = "thread_pool_scheduler_vs_inline_co_sm";
  result.comparable = true;
  result.note = "internal scheduler overhead microbenchmark";
  return result;
}

void append_thread_pool_idle_case(std::vector<emel::bench::result> & results,
                                  const emel::bench::config & cfg) {
  volatile int32_t sink = 0;
  event_tick ev{sink};
  thread_pool_machine machine{};
  auto fn = [&]() {
    emel::bool_task task = machine.process_event_async(ev);
    (void)await_result(task);
  };
  results.push_back(annotate_result(
      emel::bench::measure_case("sm_scheduler/idle_async", cfg, fn),
      "thread_pool_idle",
      "emel_thread_pool_scheduler"));
}

void append_thread_pool_busy_case(std::vector<emel::bench::result> & results,
                                  const emel::bench::config & cfg) {
  volatile int32_t sink = 0;
  event_tick ev{sink};
  thread_pool_machine machine{};
  std::binary_semaphore inline_lane_held{0};
  std::binary_semaphore release_inline_lane{0};
  std::thread inline_lane_holder{[&]() noexcept {
    const bool held = machine.scheduler().try_run_immediate([&]() noexcept {
      inline_lane_held.release();
      release_inline_lane.acquire();
    });
    if (!held) {
      std::fprintf(stderr,
                   "error: sm_scheduler busy benchmark could not hold inline lane\n");
      std::abort();
    }
  }};
  inline_lane_held.acquire();

  auto fn = [&]() {
    emel::bool_task task = machine.process_event_async(ev);
    (void)await_result(task);
  };
  results.push_back(annotate_result(
      emel::bench::measure_case("sm_scheduler/busy_worker_async", cfg, fn),
      "thread_pool_worker",
      "emel_thread_pool_scheduler"));
  release_inline_lane.release();
  inline_lane_holder.join();
}

void append_inline_idle_case(std::vector<emel::bench::result> & results,
                             const emel::bench::config & cfg,
                             const char * name,
                             const char * lane) {
  volatile int32_t sink = 0;
  event_tick ev{sink};
  inline_machine machine{};
  auto fn = [&]() {
    emel::bool_task task = machine.process_event_async(ev);
    (void)await_result(task);
  };
  results.push_back(annotate_result(
      emel::bench::measure_case(name, cfg, fn), lane, "emel_inline_scheduler"));
}

}  // namespace

namespace emel::bench {

void append_emel_sm_scheduler_cases(std::vector<result> & results,
                                    const config & cfg) {
  if (!bench_internal_enabled()) {
    return;
  }

  append_thread_pool_idle_case(results, cfg);
  append_thread_pool_busy_case(results, cfg);
}

void append_reference_sm_scheduler_cases(std::vector<result> & results,
                                         const config & cfg) {
  if (!bench_internal_enabled()) {
    return;
  }

  append_inline_idle_case(results, cfg, "sm_scheduler/idle_async", "inline_idle");
  append_inline_idle_case(
      results, cfg, "sm_scheduler/busy_worker_async", "inline_idle_baseline");
}

}  // namespace emel::bench
