#include <doctest/doctest.h>

#include "emel/kv/cache/actions.hpp"
#include "emel/kv/cache/events.hpp"
#include "emel/emel.h"

TEST_CASE("kv_cache_run_validate_covers_operation_branches") {
  emel::kv::cache::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.op = emel::kv::cache::action::operation::none;
  emel::kv::cache::action::run_validate(
    emel::kv::cache::event::validate{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.op = emel::kv::cache::action::operation::prepare;
  ctx.ubatch_count = 0;
  ctx.requested_capacity = 0;
  ctx.kv_size = 0;
  err = EMEL_OK;
  emel::kv::cache::action::run_validate(
    emel::kv::cache::event::validate{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.op = emel::kv::cache::action::operation::apply;
  ctx.planned_ubatch_count = 0;
  ctx.current_ubatch_index = 0;
  ctx.applied_ubatches = 0;
  err = EMEL_OK;
  emel::kv::cache::action::run_validate(
    emel::kv::cache::event::validate{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.op = emel::kv::cache::action::operation::rollback;
  ctx.current_ubatch_index = -1;
  ctx.applied_ubatches = 0;
  ctx.planned_ubatch_count = 0;
  err = EMEL_OK;
  emel::kv::cache::action::run_validate(
    emel::kv::cache::event::validate{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
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

  ctx.op = emel::kv::cache::action::operation::none;
  emel::kv::cache::action::run_apply_step(
    emel::kv::cache::event::apply_step{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.op = emel::kv::cache::action::operation::apply;
  ctx.planned_ubatch_count = 1;
  ctx.current_ubatch_index = 1;
  err = EMEL_OK;
  emel::kv::cache::action::run_apply_step(
    emel::kv::cache::event::apply_step{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.current_ubatch_index = 0;
  ctx.kv_size = 1;
  ctx.ubatch_sizes[0] = 2;
  ctx.slot_offsets[0] = 0;
  err = EMEL_OK;
  emel::kv::cache::action::run_apply_step(
    emel::kv::cache::event::apply_step{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("kv_cache_run_apply_step_accepts_valid_input") {
  emel::kv::cache::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.op = emel::kv::cache::action::operation::apply;
  ctx.planned_ubatch_count = 1;
  ctx.current_ubatch_index = 0;
  ctx.applied_ubatches = 0;
  ctx.kv_size = 4;
  ctx.ubatch_sizes[0] = 1;
  ctx.slot_offsets[0] = 0;

  emel::kv::cache::action::run_apply_step(
    emel::kv::cache::event::apply_step{.error_out = &err}, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.applied_ubatches == 1);
}

TEST_CASE("kv_cache_run_rollback_step_reports_errors") {
  emel::kv::cache::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.op = emel::kv::cache::action::operation::none;
  emel::kv::cache::action::run_rollback_step(
    emel::kv::cache::event::rollback_step{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.op = emel::kv::cache::action::operation::rollback;
  ctx.current_ubatch_index = 0;
  ctx.applied_ubatches = 1;
  ctx.planned_ubatch_count = 1;
  ctx.kv_size = 1;
  ctx.ubatch_sizes[0] = 2;
  ctx.slot_offsets[0] = 0;
  err = EMEL_OK;
  emel::kv::cache::action::run_rollback_step(
    emel::kv::cache::event::rollback_step{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("kv_cache_run_rollback_step_accepts_valid_input") {
  emel::kv::cache::action::context ctx{};
  int32_t err = EMEL_OK;

  ctx.op = emel::kv::cache::action::operation::rollback;
  ctx.current_ubatch_index = 0;
  ctx.applied_ubatches = 1;
  ctx.planned_ubatch_count = 1;
  ctx.kv_size = 4;
  ctx.ubatch_sizes[0] = 1;
  ctx.slot_offsets[0] = 0;
  ctx.cells[0] = 1;

  emel::kv::cache::action::run_rollback_step(
    emel::kv::cache::event::rollback_step{.error_out = &err}, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.applied_ubatches == 0);
}

TEST_CASE("kv_cache_on_done_and_error_reset_operation") {
  emel::kv::cache::action::context ctx{};

  ctx.op = emel::kv::cache::action::operation::prepare;
  ctx.current_ubatch_index = 4;
  emel::kv::cache::action::on_kv_done(emel::kv::cache::events::kv_done{}, ctx);
  CHECK(ctx.op == emel::kv::cache::action::operation::none);
  CHECK(ctx.current_ubatch_index == 0);

  ctx.op = emel::kv::cache::action::operation::apply;
  ctx.current_ubatch_index = 2;
  emel::kv::cache::action::on_kv_error(emel::kv::cache::events::kv_error{}, ctx);
  CHECK(ctx.op == emel::kv::cache::action::operation::none);
  CHECK(ctx.current_ubatch_index == 0);
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
