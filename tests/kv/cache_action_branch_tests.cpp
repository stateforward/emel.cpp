#include <doctest/doctest.h>

#include "emel/kv/cache/actions.hpp"
#include "emel/kv/cache/events.hpp"
#include "emel/emel.h"

TEST_CASE("kv_cache_run_validate_prepare_reports_invalid") {
  emel::kv::cache::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::kv::cache::event::validate_prepare validate{
    .request = nullptr,
    .error_out = &err,
  };
  emel::kv::cache::action::run_validate_prepare(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  emel::kv::cache::event::prepare prepare{};
  validate.request = &prepare;
  err = EMEL_OK;
  emel::kv::cache::action::run_validate_prepare(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  prepare.ubatch_count = 1;
  prepare.requested_capacity = emel::kv::cache::action::MAX_KV_CELLS + 1;
  ctx.kv_size = 8;
  err = EMEL_OK;
  emel::kv::cache::action::run_validate_prepare(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  prepare.requested_capacity = 8;
  ctx.kv_size = 0;
  err = EMEL_OK;
  emel::kv::cache::action::run_validate_prepare(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("kv_cache_run_validate_apply_reports_invalid_and_valid") {
  emel::kv::cache::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::kv::cache::event::apply_ubatch apply{};
  emel::kv::cache::event::validate_apply validate{
    .request = &apply,
    .error_out = &err,
  };
  emel::kv::cache::action::run_validate_apply(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.planned_ubatch_count = 2;
  apply.ubatch_index = 2;
  err = EMEL_OK;
  emel::kv::cache::action::run_validate_apply(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  apply.ubatch_index = 1;
  ctx.applied_ubatches = 0;
  err = EMEL_OK;
  emel::kv::cache::action::run_validate_apply(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  apply.ubatch_index = 0;
  ctx.applied_ubatches = 0;
  err = EMEL_OK;
  emel::kv::cache::action::run_validate_apply(validate, ctx);
  CHECK(err == EMEL_OK);
}

TEST_CASE("kv_cache_run_validate_rollback_reports_invalid_and_valid") {
  emel::kv::cache::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::kv::cache::event::rollback rollback{};
  emel::kv::cache::event::validate_rollback validate{
    .request = nullptr,
    .error_out = &err,
  };
  emel::kv::cache::action::run_validate_rollback(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  validate.request = &rollback;
  ctx.applied_ubatches = 2;
  ctx.planned_ubatch_count = 2;
  rollback.from_ubatch_index = 3;
  err = EMEL_OK;
  emel::kv::cache::action::run_validate_rollback(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  rollback.from_ubatch_index = 1;
  err = EMEL_OK;
  emel::kv::cache::action::run_validate_rollback(validate, ctx);
  CHECK(err == EMEL_OK);
}

TEST_CASE("kv_cache_prepare_slots_reports_invalid_and_backend") {
  emel::kv::cache::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.ubatch_count = 1;
  ctx.ubatch_sizes[0] = 0;
  ctx.kv_size = 4;
  emel::kv::cache::action::run_prepare_slots(
    emel::kv::cache::event::prepare_slots{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.ubatch_sizes[0] = 4;
  ctx.kv_size = 2;
  err = EMEL_OK;
  emel::kv::cache::action::run_prepare_slots(
    emel::kv::cache::event::prepare_slots{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("kv_cache_prepare_slots_accepts_valid_input") {
  emel::kv::cache::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.ubatch_count = 1;
  ctx.ubatch_sizes[0] = 2;
  ctx.kv_size = 4;
  emel::kv::cache::action::run_prepare_slots(
    emel::kv::cache::event::prepare_slots{.error_out = &err}, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.planned_ubatch_count == 1);
}

TEST_CASE("kv_cache_run_apply_step_reports_errors") {
  emel::kv::cache::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::kv::cache::event::apply_step step{
    .request = nullptr,
    .error_out = &err,
  };
  emel::kv::cache::action::run_apply_step(
    step, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  emel::kv::cache::event::apply_ubatch apply{};
  ctx.planned_ubatch_count = 1;
  apply.ubatch_index = 1;
  step.request = &apply;
  err = EMEL_OK;
  emel::kv::cache::action::run_apply_step(
    step, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  apply.ubatch_index = 0;
  ctx.kv_size = 1;
  ctx.ubatch_sizes[0] = 2;
  ctx.slot_offsets[0] = 0;
  err = EMEL_OK;
  emel::kv::cache::action::run_apply_step(
    step, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("kv_cache_run_apply_step_accepts_valid_input") {
  emel::kv::cache::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::kv::cache::event::apply_ubatch apply{};
  emel::kv::cache::event::apply_step step{.request = &apply, .error_out = &err};
  ctx.planned_ubatch_count = 1;
  apply.ubatch_index = 0;
  ctx.applied_ubatches = 0;
  ctx.kv_size = 4;
  ctx.ubatch_sizes[0] = 1;
  ctx.slot_offsets[0] = 0;

  emel::kv::cache::action::run_apply_step(
    step, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.applied_ubatches == 1);
}

TEST_CASE("kv_cache_run_rollback_step_reports_errors") {
  emel::kv::cache::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::kv::cache::event::rollback rollback{};
  emel::kv::cache::event::rollback_step step{.request = nullptr, .error_out = &err};
  emel::kv::cache::action::run_rollback_step(
    step, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  rollback.from_ubatch_index = 0;
  step.request = &rollback;
  ctx.applied_ubatches = 1;
  ctx.planned_ubatch_count = 1;
  ctx.kv_size = 1;
  ctx.ubatch_sizes[0] = 2;
  ctx.slot_offsets[0] = 0;
  err = EMEL_OK;
  emel::kv::cache::action::run_rollback_step(
    step, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("kv_cache_run_rollback_step_accepts_valid_input") {
  emel::kv::cache::action::context ctx{};
  int32_t err = EMEL_OK;

  emel::kv::cache::event::rollback rollback{};
  emel::kv::cache::event::rollback_step step{.request = &rollback, .error_out = &err};
  rollback.from_ubatch_index = 0;
  ctx.applied_ubatches = 1;
  ctx.planned_ubatch_count = 1;
  ctx.kv_size = 4;
  ctx.ubatch_sizes[0] = 1;
  ctx.slot_offsets[0] = 0;
  ctx.cells[0] = 1;

  emel::kv::cache::action::run_rollback_step(
    step, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.applied_ubatches == 0);
}

TEST_CASE("kv_cache_on_done_and_error_reset_operation") {
  emel::kv::cache::action::context ctx{};

  ctx.applied_ubatches = 2;
  emel::kv::cache::action::on_kv_done(emel::kv::cache::events::kv_done{}, ctx);
  CHECK(ctx.applied_ubatches == 2);

  ctx.planned_ubatch_count = 3;
  emel::kv::cache::action::on_kv_error(emel::kv::cache::events::kv_error{}, ctx);
  CHECK(ctx.planned_ubatch_count == 3);
}

TEST_CASE("kv_cache_on_unexpected_sets_error_out") {
  emel::kv::cache::action::context ctx{};
  int32_t err = EMEL_OK;
  emel::kv::cache::event::prepare prepare{.error_out = &err};

  emel::kv::cache::action::on_unexpected{}(prepare, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("kv_cache_detail_helpers_cover_branches") {
  emel::kv::cache::action::context ctx{};
  CHECK(emel::kv::cache::action::count_used_cells(ctx) == 0);
  CHECK(emel::kv::cache::action::used_max_p1(ctx) == 0);

  ctx.kv_size = 4;
  ctx.cells[2] = 1;
  CHECK(emel::kv::cache::action::count_used_cells(ctx) == 1);
  CHECK(emel::kv::cache::action::used_max_p1(ctx) == 3);

  int32_t head_after = 0;
  CHECK(emel::kv::cache::action::find_contiguous_slot(
          ctx.cells, 0, 0, 1, 0, head_after) == -1);
  CHECK(emel::kv::cache::action::find_contiguous_slot(
          ctx.cells, 4, 0, 5, 0, head_after) == -1);

  head_after = 0;
  ctx.cells.fill(0);
  CHECK(emel::kv::cache::action::find_contiguous_slot(
          ctx.cells, 4, 3, 2, 0, head_after) == 0);
}
