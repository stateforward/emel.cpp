#include <doctest/doctest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <thread>

#include "../allocation_tracker.hpp"
#include "emel/sm.hpp"

// Coverage for the emel wrapper surface of the upstream external-completion
// co_sm policy: dispatches that suspend awaiting completions fired by worker
// threads must drain before process_event returns, deliver deterministically
// on the dispatching thread, and stay allocation-free via the fixed coroutine
// allocator.

namespace {

namespace sml = stateforward::sml;

using stream_scheduler = emel::policy::external_completion_scheduler<8>;
using stream_pool = emel::policy::thread_pool_scheduler<1, 16u, 128u>;
using stream_co_policy = emel::policy::external_completion_co_policy<8>;
using allocation_scope = emel::test::allocation::allocation_scope;

static_assert(emel::policy::external_completion_scheduler_contract<stream_scheduler>,
              "external completion scheduler must satisfy the contract");
static_assert(!emel::policy::external_completion_scheduler_contract<emel::policy::inline_scheduler>,
              "inline scheduler must not satisfy the external completion contract");
static_assert(!emel::policy::external_completion_scheduler_contract<emel::policy::fifo_scheduler<>>,
              "fifo scheduler must not satisfy the external completion contract");

constexpr std::size_t k_probe_marker = 99u;

struct stream_probe {
  stream_scheduler * scheduler = nullptr;
  stream_pool * pool = nullptr;
  std::array<std::size_t, 16> log{};
  std::size_t log_count = 0;
  std::thread::id completion_thread{};
  bool worker_delay = false;
};

struct context_stream {
  stream_probe * probe = nullptr;
};

struct event_require_via_worker {
  std::size_t count;
};

struct event_require_fired_inline {
  std::size_t count;
};

struct event_hand_unrequired {
  std::size_t index;
};

struct event_probe {};

struct event_error_probe {
  int32_t * error_out = nullptr;
};

void submit_worker_fire(stream_probe & probe, const std::size_t index) {
  const bool delay = probe.worker_delay;
  (void)probe.pool->try_submit_with_completion(
      [delay]() noexcept {
        if (delay) {
          const auto until =
              std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
          while (std::chrono::steady_clock::now() < until) {
          }
        }
      },
      &probe.scheduler->source(index), &emel::policy::completion_source::fire);
}

struct state_stream_ready {};

struct effect_require_via_worker {
  void operator()(const event_require_via_worker & ev, context_stream & ctx) const noexcept {
    for (std::size_t index = 0; index < ev.count; ++index) {
      ctx.probe->scheduler->source(index).arm();
      submit_worker_fire(*ctx.probe, index);
    }
    for (std::size_t index = 0; index < ev.count; ++index) {
      ctx.probe->scheduler->require(index);
    }
  }
};

struct effect_require_fired_inline {
  void operator()(const event_require_fired_inline & ev, context_stream & ctx) const noexcept {
    for (std::size_t index = 0; index < ev.count; ++index) {
      ctx.probe->scheduler->source(index).arm();
    }
    // Fire descending on this thread: delivery must still ascend.
    for (std::size_t reverse = ev.count; reverse > 0; --reverse) {
      emel::policy::completion_source::fire(&ctx.probe->scheduler->source(reverse - 1));
    }
    for (std::size_t index = 0; index < ev.count; ++index) {
      ctx.probe->scheduler->require(index);
    }
  }
};

struct effect_hand_unrequired {
  void operator()(const event_hand_unrequired & ev, context_stream & ctx) const noexcept {
    ctx.probe->scheduler->source(ev.index).arm();
    submit_worker_fire(*ctx.probe, ev.index);
  }
};

struct effect_probe {
  void operator()(const event_probe &, context_stream & ctx) const noexcept {
    ctx.probe->log[ctx.probe->log_count++] = k_probe_marker;
  }
};

struct effect_error_probe {
  void operator()(const event_error_probe & ev, context_stream &) const noexcept {
    if (ev.error_out != nullptr) {
      *ev.error_out = 7;
    }
  }
};

struct effect_record_completion {
  void operator()(const emel::event::completion & ev, context_stream & ctx) const noexcept {
    ctx.probe->log[ctx.probe->log_count++] = ev.source_index;
    ctx.probe->completion_thread = std::this_thread::get_id();
  }
};

struct stream_model {
  auto operator()() const noexcept {
    // clang-format off
    return sml::make_transition_table(
        sml::state<state_stream_ready> <= *sml::state<state_stream_ready>
          + sml::event<event_require_via_worker> / effect_require_via_worker{}
      , sml::state<state_stream_ready> <= sml::state<state_stream_ready>
          + sml::event<event_require_fired_inline> / effect_require_fired_inline{}
      , sml::state<state_stream_ready> <= sml::state<state_stream_ready>
          + sml::event<event_hand_unrequired> / effect_hand_unrequired{}
      , sml::state<state_stream_ready> <= sml::state<state_stream_ready>
          + sml::event<event_probe> / effect_probe{}
      , sml::state<state_stream_ready> <= sml::state<state_stream_ready>
          + sml::event<event_error_probe> / effect_error_probe{}
      , sml::state<state_stream_ready> <= sml::state<state_stream_ready>
          + sml::event<emel::event::completion> / effect_record_completion{}
    );
    // clang-format on
  }
};

using stream_machine = emel::co_sm<stream_model, context_stream, stream_co_policy>;

struct stream_fixture {
  stream_pool pool{};
  stream_probe probe{};
  stream_machine machine;

  stream_fixture() : machine{context_stream{.probe = &probe}} {
    probe.scheduler = &machine.scheduler();
    probe.pool = &pool;
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

}  // namespace

TEST_CASE("co_sm_external_completion_required_drained_before_return") {
  stream_fixture fixture{};
  fixture.probe.worker_delay = true;  // force the dispatch coroutine to park

  CHECK(fixture.machine.process_event(event_require_via_worker{.count = 3}));

  CHECK(fixture.probe.log_count == 3u);
  CHECK_FALSE(fixture.machine.scheduler().has_required());
  for (std::size_t index = 0; index < 3; ++index) {
    CHECK(fixture.machine.scheduler().source(index).is_idle());
  }
}

TEST_CASE("co_sm_external_completion_delivers_on_dispatching_thread") {
  stream_fixture fixture{};
  fixture.probe.worker_delay = true;

  CHECK(fixture.machine.process_event(event_require_via_worker{.count = 1}));

  CHECK(fixture.probe.log_count == 1u);
  CHECK(fixture.probe.completion_thread == std::this_thread::get_id());
}

TEST_CASE("co_sm_external_completion_prefired_delivery_is_ascending") {
  stream_fixture fixture{};

  CHECK(fixture.machine.process_event(event_require_fired_inline{.count = 3}));

  CHECK(fixture.probe.log_count == 3u);
  CHECK(fixture.probe.log[0] == 0u);
  CHECK(fixture.probe.log[1] == 1u);
  CHECK(fixture.probe.log[2] == 2u);
}

TEST_CASE("co_sm_external_completion_background_fire_swept_before_next_trigger") {
  stream_fixture fixture{};

  CHECK(fixture.machine.process_event(event_hand_unrequired{.index = 5}));
  CHECK(fixture.probe.log_count == 0u);

  require_eventually("background source fires", [&]() {
    return fixture.machine.scheduler().source(5).is_fired();
  });

  CHECK(fixture.machine.process_event(event_probe{}));

  CHECK(fixture.probe.log_count == 2u);
  CHECK(fixture.probe.log[0] == 5u);
  CHECK(fixture.probe.log[1] == k_probe_marker);
  CHECK(fixture.machine.scheduler().source(5).is_idle());
}

TEST_CASE("co_sm_external_completion_dispatch_without_requires_never_parks") {
  stream_fixture fixture{};

  CHECK(fixture.machine.process_event(event_probe{}));

  CHECK(fixture.probe.log_count == 1u);
  CHECK(fixture.probe.log[0] == k_probe_marker);
}

TEST_CASE("co_sm_external_completion_async_wrapper_completes_inline") {
  stream_fixture fixture{};
  fixture.probe.worker_delay = true;

  emel::bool_task task =
      fixture.machine.process_event_async(event_require_via_worker{.count = 2});

  CHECK(task.await_ready());
  CHECK(task.result());
  CHECK(fixture.probe.log_count == 2u);
}

TEST_CASE("co_sm_external_completion_normalizes_error_out") {
  stream_fixture fixture{};
  int32_t error_value = 0;

  CHECK_FALSE(fixture.machine.process_event(event_error_probe{.error_out = &error_value}));
  CHECK(error_value == 7);
}

TEST_CASE("co_sm_external_completion_suspending_dispatch_does_not_allocate") {
  stream_fixture fixture{};
  fixture.probe.worker_delay = true;

  allocation_scope allocations{};

  CHECK(fixture.machine.process_event(event_require_via_worker{.count = 2}));
  CHECK(fixture.probe.log_count == 2u);
  CHECK(allocations.allocations() == 0u);
}

TEST_CASE("co_sm_external_completion_sources_reusable_across_dispatches") {
  stream_fixture fixture{};

  for (std::size_t round = 0; round < 3; ++round) {
    fixture.probe.log_count = 0;
    CHECK(fixture.machine.process_event(event_require_fired_inline{.count = 2}));
    CHECK(fixture.probe.log_count == 2u);
    CHECK(fixture.probe.log[0] == 0u);
    CHECK(fixture.probe.log[1] == 1u);
  }
}
