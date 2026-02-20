#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/telemetry/provider/events.hpp"

namespace emel::telemetry::provider::action {

struct context {
  void * queue_ctx = nullptr;
  emel::telemetry::enqueue_record_fn try_enqueue = nullptr;
  int32_t max_batch = 64;

  uint64_t sessions_started = 0;
  uint64_t records_emitted = 0;
  uint64_t records_dropped = 0;

  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

}  // namespace emel::telemetry::provider::action
