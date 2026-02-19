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

}  // namespace emel::telemetry::exporter::event
