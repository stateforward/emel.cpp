#pragma once

#include <cstdint>

#include "emel/telemetry/record.hpp"

namespace emel::telemetry::exporter::event {

struct configure {
  void * queue_ctx = nullptr;
  emel::telemetry::dequeue_record_fn try_dequeue = nullptr;
  void * exporter_ctx = nullptr;
  emel::telemetry::flush_records_fn flush_records = nullptr;
  int32_t batch_capacity = 64;
  int32_t * error_out = nullptr;
};

struct start {
  int32_t * error_out = nullptr;
};

struct tick {
  int32_t max_records = 0;
  int32_t * error_out = nullptr;
};

struct stop {
  int32_t * error_out = nullptr;
};

struct reset {
  int32_t * error_out = nullptr;
};

struct validate_config {
  int32_t * error_out = nullptr;
};

struct run_start {
  int32_t * error_out = nullptr;
};

struct collect_batch {
  int32_t * error_out = nullptr;
};

struct flush_batch {
  int32_t * error_out = nullptr;
};

struct run_backoff {
  int32_t * error_out = nullptr;
};

struct run_stop {
  int32_t * error_out = nullptr;
};

}  // namespace emel::telemetry::exporter::event

namespace emel::telemetry::exporter::events {

struct configure_done {
  const event::configure * request = nullptr;
};
struct configure_error {
  int32_t err = 0;
  const event::configure * request = nullptr;
};

struct start_done {
  const event::start * request = nullptr;
};
struct start_error {
  int32_t err = 0;
  const event::start * request = nullptr;
};

struct collect_batch_done {
  const event::tick * request = nullptr;
};
struct collect_batch_error {
  int32_t err = 0;
  const event::tick * request = nullptr;
};

struct flush_batch_done {
  const event::tick * request = nullptr;
};
struct flush_batch_error {
  int32_t err = 0;
  const event::tick * request = nullptr;
};

struct backoff_done {
  const event::tick * request = nullptr;
};
struct backoff_error {
  int32_t err = 0;
  const event::tick * request = nullptr;
};

struct stop_done {
  const event::stop * request = nullptr;
};
struct stop_error {
  int32_t err = 0;
  const event::stop * request = nullptr;
};

}  // namespace emel::telemetry::exporter::events
