#include "doctest/doctest.h"

#include "emel/emel.h"
#include "emel/telemetry/exporter/sm.hpp"

namespace {

struct queue_state {
  emel::telemetry::record records[4] = {};
  int32_t count = 0;
  int32_t index = 0;
};

bool try_dequeue(void * queue_ctx, emel::telemetry::record * out_value) noexcept {
  auto * state = static_cast<queue_state *>(queue_ctx);
  if (state == nullptr || out_value == nullptr) {
    return false;
  }
  if (state->index >= state->count) {
    return false;
  }
  *out_value = state->records[state->index];
  state->index += 1;
  return true;
}

struct exporter_state {
  int32_t flushed_calls = 0;
  int32_t last_count = 0;
  bool should_fail = false;
};

bool flush_records(void * exporter_ctx,
                   const emel::telemetry::record *,
                   int32_t record_count,
                   int32_t * error_out) noexcept {
  auto * state = static_cast<exporter_state *>(exporter_ctx);
  if (state == nullptr) {
    if (error_out != nullptr) {
      *error_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  state->flushed_calls += 1;
  state->last_count = record_count;
  if (state->should_fail) {
    if (error_out != nullptr) {
      *error_out = EMEL_ERR_BACKEND;
    }
    return false;
  }
  if (error_out != nullptr) {
    *error_out = EMEL_OK;
  }
  return true;
}

}  // namespace

TEST_CASE("exporter configures, starts, and flushes batches") {
  emel::telemetry::exporter::sm machine{};
  queue_state queue{};
  exporter_state exporter{};
  queue.count = 2;

  int32_t err = EMEL_OK;
  emel::telemetry::exporter::event::configure configure{};
  configure.queue_ctx = &queue;
  configure.try_dequeue = try_dequeue;
  configure.exporter_ctx = &exporter;
  configure.flush_records = flush_records;
  configure.batch_capacity = 4;
  configure.error_out = &err;

  CHECK(machine.process_event(configure));
  CHECK(err == EMEL_OK);

  emel::telemetry::exporter::event::start start{};
  start.error_out = &err;
  CHECK(machine.process_event(start));
  CHECK(err == EMEL_OK);

  emel::telemetry::exporter::event::tick tick{};
  tick.max_records = 0;
  tick.error_out = &err;
  CHECK(machine.process_event(tick));
  CHECK(err == EMEL_OK);
  CHECK(exporter.flushed_calls == 1);
  CHECK(exporter.last_count == 2);
  CHECK(machine.flushed_records() == 2);
  CHECK(machine.dropped_records() == 0);
}

TEST_CASE("exporter tick with empty queue does not flush") {
  emel::telemetry::exporter::sm machine{};
  queue_state queue{};
  exporter_state exporter{};
  int32_t err = EMEL_OK;

  emel::telemetry::exporter::event::configure configure{};
  configure.queue_ctx = &queue;
  configure.try_dequeue = try_dequeue;
  configure.exporter_ctx = &exporter;
  configure.flush_records = flush_records;
  configure.batch_capacity = 4;
  configure.error_out = &err;
  CHECK(machine.process_event(configure));
  CHECK(err == EMEL_OK);

  emel::telemetry::exporter::event::start start{};
  start.error_out = &err;
  CHECK(machine.process_event(start));
  CHECK(err == EMEL_OK);

  emel::telemetry::exporter::event::tick tick{};
  tick.max_records = 0;
  tick.error_out = &err;
  CHECK(machine.process_event(tick));
  CHECK(err == EMEL_OK);
  CHECK(exporter.flushed_calls == 0);
}

TEST_CASE("exporter flush error triggers backoff and reports error") {
  emel::telemetry::exporter::sm machine{};
  queue_state queue{};
  exporter_state exporter{};
  exporter.should_fail = true;
  queue.count = 1;

  int32_t err = EMEL_OK;
  emel::telemetry::exporter::event::configure configure{};
  configure.queue_ctx = &queue;
  configure.try_dequeue = try_dequeue;
  configure.exporter_ctx = &exporter;
  configure.flush_records = flush_records;
  configure.batch_capacity = 4;
  configure.error_out = &err;
  CHECK(machine.process_event(configure));
  CHECK(err == EMEL_OK);

  emel::telemetry::exporter::event::start start{};
  start.error_out = &err;
  CHECK(machine.process_event(start));
  CHECK(err == EMEL_OK);

  emel::telemetry::exporter::event::tick tick{};
  tick.max_records = 0;
  tick.error_out = &err;
  CHECK(!machine.process_event(tick));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(machine.dropped_records() == 1);
  CHECK(machine.backoff_count() == 1);
}

TEST_CASE("exporter start fails without configure and reset clears error") {
  emel::telemetry::exporter::sm machine{};

  int32_t err = EMEL_OK;
  emel::telemetry::exporter::event::start start{};
  start.error_out = &err;
  CHECK(!machine.process_event(start));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  emel::telemetry::exporter::event::reset reset{};
  reset.error_out = &err;
  CHECK(machine.process_event(reset));
  CHECK(err == EMEL_OK);
}

TEST_CASE("exporter stop returns to initialized") {
  emel::telemetry::exporter::sm machine{};
  queue_state queue{};
  exporter_state exporter{};
  int32_t err = EMEL_OK;

  emel::telemetry::exporter::event::configure configure{};
  configure.queue_ctx = &queue;
  configure.try_dequeue = try_dequeue;
  configure.exporter_ctx = &exporter;
  configure.flush_records = flush_records;
  configure.batch_capacity = 4;
  configure.error_out = &err;
  CHECK(machine.process_event(configure));
  CHECK(err == EMEL_OK);

  emel::telemetry::exporter::event::start start{};
  start.error_out = &err;
  CHECK(machine.process_event(start));
  CHECK(err == EMEL_OK);

  emel::telemetry::exporter::event::stop stop{};
  stop.error_out = &err;
  CHECK(machine.process_event(stop));
  CHECK(err == EMEL_OK);
}
