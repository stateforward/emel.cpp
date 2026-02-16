#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/decoder/compute_executor/sm.hpp"

namespace {

TEST_CASE("compute_executor_sm_success_path_reports_outputs") {
  emel::decoder::compute_executor::sm machine{};
  int32_t err = EMEL_OK;
  int32_t outputs = 0;

  machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .kv_tokens = 0,
    .outputs_produced_out = &outputs,
    .error_out = &err,
  });
  CHECK(err != EMEL_OK);
}

TEST_CASE("compute_executor_sm_validation_error_path") {
  emel::decoder::compute_executor::sm machine{};
  int32_t err = EMEL_OK;

  machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = -1,
    .ubatch_size = 0,
    .kv_tokens = 0,
    .error_out = &err,
  });
  CHECK(err != EMEL_OK);
}

}  // namespace
