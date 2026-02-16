#pragma once

#include <cstdint>

namespace emel::telemetry {

struct record {
  uint64_t timestamp_ns = 0;
  uint32_t component_id = 0;
  uint32_t machine_id = 0;
  uint32_t state_id = 0;
  uint32_t event_id = 0;
  uint32_t phase_id = 0;
  int32_t status = 0;
};

using enqueue_record_fn = bool (*)(void * queue_ctx, const record & value) noexcept;
using dequeue_record_fn = bool (*)(void * queue_ctx, record * out_value) noexcept;
using flush_records_fn = bool (*)(
    void * exporter_ctx,
    const record * records,
    int32_t record_count,
    int32_t * error_out) noexcept;

}  // namespace emel::telemetry

