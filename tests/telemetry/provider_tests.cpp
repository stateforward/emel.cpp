#include "doctest/doctest.h"

#include "emel/emel.h"
#include "emel/telemetry/provider/sm.hpp"

namespace {

struct queue_state {
  emel::telemetry::record records[4] = {};
  int32_t count = 0;
  int32_t capacity = 4;
};

bool try_enqueue(void * queue_ctx, const emel::telemetry::record & value) noexcept {
  auto * state = static_cast<queue_state *>(queue_ctx);
  if (state == nullptr) {
    return false;
  }
  if (state->count >= state->capacity) {
    return false;
  }
  state->records[state->count] = value;
  state->count += 1;
  return true;
}

}  // namespace

TEST_CASE("provider configures, starts, and publishes") {
  emel::telemetry::provider::sm machine{};
  queue_state queue{};

  int32_t err = EMEL_OK;
  emel::telemetry::provider::event::configure configure{};
  configure.queue_ctx = &queue;
  configure.try_enqueue = try_enqueue;
  configure.max_batch = 4;
  configure.error_out = &err;
  CHECK(machine.process_event(configure));
  CHECK(err == EMEL_OK);

  emel::telemetry::provider::event::start start{};
  start.error_out = &err;
  CHECK(machine.process_event(start));
  CHECK(err == EMEL_OK);
  CHECK(machine.sessions_started() == 1);

  bool dropped = false;
  emel::telemetry::provider::event::publish publish{};
  publish.value.status = 7;
  publish.dropped_out = &dropped;
  publish.error_out = &err;
  CHECK(machine.process_event(publish));
  CHECK(err == EMEL_OK);
  CHECK(dropped == false);
  CHECK(queue.count == 1);
  CHECK(queue.records[0].status == 7);
  CHECK(machine.records_emitted() == 1);
  CHECK(machine.records_dropped() == 0);
}

TEST_CASE("provider publish drop does not error") {
  emel::telemetry::provider::sm machine{};
  queue_state queue{};
  queue.capacity = 0;

  int32_t err = EMEL_OK;
  emel::telemetry::provider::event::configure configure{};
  configure.queue_ctx = &queue;
  configure.try_enqueue = try_enqueue;
  configure.max_batch = 1;
  configure.error_out = &err;
  CHECK(machine.process_event(configure));
  CHECK(err == EMEL_OK);

  emel::telemetry::provider::event::start start{};
  start.error_out = &err;
  CHECK(machine.process_event(start));
  CHECK(err == EMEL_OK);

  bool dropped = false;
  emel::telemetry::provider::event::publish publish{};
  publish.value.status = 3;
  publish.dropped_out = &dropped;
  publish.error_out = &err;
  CHECK(machine.process_event(publish));
  CHECK(err == EMEL_OK);
  CHECK(dropped == true);
  CHECK(queue.count == 0);
  CHECK(machine.records_emitted() == 0);
  CHECK(machine.records_dropped() == 1);
}

TEST_CASE("provider start fails without configure and reset recovers") {
  emel::telemetry::provider::sm machine{};

  int32_t err = EMEL_OK;
  emel::telemetry::provider::event::start start{};
  start.error_out = &err;
  CHECK(!machine.process_event(start));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  emel::telemetry::provider::event::reset reset{};
  reset.error_out = &err;
  CHECK(machine.process_event(reset));
  CHECK(err == EMEL_OK);
}

TEST_CASE("provider unexpected publish transitions to error") {
  emel::telemetry::provider::sm machine{};

  int32_t err = EMEL_OK;
  bool dropped = false;
  emel::telemetry::provider::event::publish publish{};
  publish.dropped_out = &dropped;
  publish.error_out = &err;
  CHECK(!machine.process_event(publish));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(dropped == true);
}
