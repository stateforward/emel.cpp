#pragma once

#include <array>
#include <cstdint>

#include "emel/emel.h"
#include "emel/telemetry/exporter/events.hpp"

namespace emel::telemetry::exporter::action {

inline constexpr int32_t k_max_batch_capacity = 256;

struct context {
  void * queue_ctx = nullptr;
  emel::telemetry::dequeue_record_fn try_dequeue = nullptr;
  void * exporter_ctx = nullptr;
  emel::telemetry::flush_records_fn flush_records = nullptr;
  int32_t batch_capacity = 64;
  int32_t tick_max_records = 0;

  std::array<emel::telemetry::record, k_max_batch_capacity> batch = {};
  int32_t batch_count = 0;
  uint64_t flushed_records = 0;
  uint64_t dropped_records = 0;
  uint32_t backoff_count = 0;

  int32_t * error_out = nullptr;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::telemetry::exporter::action
