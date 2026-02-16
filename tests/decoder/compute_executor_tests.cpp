#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/decoder/compute_executor/sm.hpp"
#include "emel/emel.h"

TEST_CASE("compute_executor_starts_initialized") {
  emel::decoder::compute_executor::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::decoder::compute_executor::initialized>));
}

TEST_CASE("compute_executor_execute_success_path") {
  emel::decoder::compute_executor::sm machine{};
  int32_t outputs_produced = 0;
  int32_t error = EMEL_OK;

  CHECK(machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = 1,
    .ubatch_size = 4,
    .kv_tokens = 9,
    .outputs_produced_out = &outputs_produced,
    .error_out = &error,
  }));

  CHECK(error == EMEL_OK);
  CHECK(machine.outputs_produced() == 4);
  CHECK(outputs_produced == 4);
}

TEST_CASE("compute_executor_rejects_invalid_payload") {
  emel::decoder::compute_executor::sm machine{};
  int32_t error = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = -1,
    .ubatch_size = 1,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 0,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .kv_tokens = -1,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("compute_executor_handles_backend_and_extract_failures") {
  emel::decoder::compute_executor::sm machine{};
  int32_t error = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 2,
    .kv_tokens = 0,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_BACKEND);

  CHECK_FALSE(machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 2,
    .kv_tokens = 1,
    .error_out = &error,
  }));
  CHECK(error == EMEL_ERR_BACKEND);
}

TEST_CASE("compute_executor_runs_to_completion_on_execute") {
  emel::decoder::compute_executor::sm machine{};
  int32_t error = EMEL_OK;

  CHECK(machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = 0,
    .ubatch_size = 1,
    .kv_tokens = 1,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);
  CHECK(machine.is(boost::sml::state<emel::decoder::compute_executor::initialized>));

  error = EMEL_OK;
  CHECK(machine.process_event(emel::decoder::compute_executor::event::execute{
    .ubatch_index = 1,
    .ubatch_size = 1,
    .kv_tokens = 1,
    .error_out = &error,
  }));
  CHECK(error == EMEL_OK);
  CHECK(machine.is(boost::sml::state<emel::decoder::compute_executor::initialized>));
}
