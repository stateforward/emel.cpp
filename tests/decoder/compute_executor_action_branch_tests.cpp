#include <doctest/doctest.h>

#include "emel/decoder/compute_executor/actions.hpp"
#include "emel/decoder/compute_executor/events.hpp"

TEST_CASE("compute_executor_actions_return_early_when_error_out_is_null") {
  emel::decoder::compute_executor::action::context ctx{};

  emel::decoder::compute_executor::action::run_validate(
    emel::decoder::compute_executor::event::validate{}, ctx);
  emel::decoder::compute_executor::action::run_bind_inputs(
    emel::decoder::compute_executor::event::bind_inputs{}, ctx);
  emel::decoder::compute_executor::action::run_backend(
    emel::decoder::compute_executor::event::run_backend{}, ctx);
  emel::decoder::compute_executor::action::run_extract_outputs(
    emel::decoder::compute_executor::event::extract_outputs{}, ctx);
}

TEST_CASE("compute_executor_run_validate_reports_invalid_values") {
  emel::decoder::compute_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.ubatch_index = -1;
  ctx.ubatch_size = 0;
  ctx.kv_tokens = -1;
  emel::decoder::compute_executor::action::run_validate(
    emel::decoder::compute_executor::event::validate{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("compute_executor_run_backend_requires_tokens") {
  emel::decoder::compute_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.kv_tokens = 0;
  emel::decoder::compute_executor::action::run_backend(
    emel::decoder::compute_executor::event::run_backend{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("compute_executor_run_extract_outputs_checks_kv_tokens") {
  emel::decoder::compute_executor::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.kv_tokens = 1;
  ctx.ubatch_size = 2;
  emel::decoder::compute_executor::action::run_extract_outputs(
    emel::decoder::compute_executor::event::extract_outputs{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}
