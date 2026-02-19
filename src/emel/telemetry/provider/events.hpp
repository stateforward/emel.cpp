#pragma once

#include <cstdint>

#include "emel/telemetry/record.hpp"

namespace emel::telemetry::provider::event {

struct configure {
  void * queue_ctx = nullptr;
  emel::telemetry::enqueue_record_fn try_enqueue = nullptr;
  int32_t max_batch = 64;
  int32_t * error_out = nullptr;
};

struct start {
  int32_t * error_out = nullptr;
};

struct publish {
  emel::telemetry::record value = {};
  bool * dropped_out = nullptr;
  int32_t * error_out = nullptr;
};

struct stop {
  int32_t * error_out = nullptr;
};

struct reset {
  int32_t * error_out = nullptr;
};

}  // namespace emel::telemetry::provider::event

namespace emel::telemetry::provider::events {

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

struct publish_done {
  bool dropped = false;
  const event::publish * request = nullptr;
};

struct publish_error {
  int32_t err = 0;
  const event::publish * request = nullptr;
};

struct stop_done {
  const event::stop * request = nullptr;
};

struct stop_error {
  int32_t err = 0;
  const event::stop * request = nullptr;
};

struct reset_done {
  const event::reset * request = nullptr;
};

struct reset_error {
  int32_t err = 0;
  const event::reset * request = nullptr;
};

}  // namespace emel::telemetry::provider::events
