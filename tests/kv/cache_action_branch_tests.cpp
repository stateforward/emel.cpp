#include <array>
#include <memory>
#include <doctest/doctest.h>

#include "emel/kv/cache/actions.hpp"
#include "emel/kv/cache/events.hpp"
#include "emel/kv/cache/guards.hpp"
#include "emel/emel.h"

TEST_CASE("kv_cache_validate_prepare_guard_reports_invalid") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  emel::kv::cache::event::validate_prepare validate{
    .request = nullptr,
    .error_out = &err,
  };
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));
  emel::kv::cache::action::reject_invalid_prepare(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  std::array<int32_t, 1> sizes = {{1}};
  emel::kv::cache::event::prepare prepare{};
  validate.request = &prepare;
  err = EMEL_OK;
  emel::kv::cache::action::begin_prepare(prepare, ctx);
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));
  emel::kv::cache::action::reject_invalid_prepare(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  prepare.ubatch_sizes = sizes.data();
  prepare.ubatch_count = 1;
  prepare.requested_capacity = emel::kv::cache::action::MAX_KV_CELLS + 1;
  err = EMEL_OK;
  emel::kv::cache::action::begin_prepare(prepare, ctx);
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));
  emel::kv::cache::action::reject_invalid_prepare(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  prepare.requested_capacity = 8;
  err = EMEL_OK;
  emel::kv::cache::action::begin_prepare(prepare, ctx);
  CHECK(emel::kv::cache::guard::valid_prepare_request(validate, ctx));
}

TEST_CASE("kv_cache_validate_apply_guard_reports_invalid_and_valid") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  emel::kv::cache::event::apply_ubatch apply{};
  emel::kv::cache::event::validate_apply validate{
    .request = &apply,
    .error_out = &err,
  };
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_request(validate, ctx));
  emel::kv::cache::action::reject_invalid_apply(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.planned_ubatch_count = 2;
  apply.ubatch_index = 2;
  err = EMEL_OK;
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_request(validate, ctx));
  emel::kv::cache::action::reject_invalid_apply(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  apply.ubatch_index = 1;
  ctx.applied_ubatches = 0;
  err = EMEL_OK;
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_request(validate, ctx));
  emel::kv::cache::action::reject_invalid_apply(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  apply.ubatch_index = 0;
  ctx.applied_ubatches = 0;
  err = EMEL_OK;
  CHECK(emel::kv::cache::guard::valid_apply_request(validate, ctx));
}

TEST_CASE("kv_cache_validate_rollback_guard_reports_invalid_and_valid") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  emel::kv::cache::event::rollback rollback{};
  emel::kv::cache::event::validate_rollback validate{
    .request = nullptr,
    .error_out = &err,
  };
  CHECK_FALSE(emel::kv::cache::guard::valid_rollback_request(validate, ctx));
  emel::kv::cache::action::reject_invalid_rollback(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  validate.request = &rollback;
  ctx.applied_ubatches = 2;
  ctx.planned_ubatch_count = 2;
  rollback.from_ubatch_index = 3;
  err = EMEL_OK;
  CHECK_FALSE(emel::kv::cache::guard::valid_rollback_request(validate, ctx));
  emel::kv::cache::action::reject_invalid_rollback(validate, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  rollback.from_ubatch_index = 1;
  err = EMEL_OK;
  CHECK(emel::kv::cache::guard::valid_rollback_request(validate, ctx));
}

TEST_CASE("kv_cache_prepare_slots_reports_invalid_and_backend") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  ctx.ubatch_count = 1;
  ctx.ubatch_sizes[0] = 0;
  ctx.kv_size = 4;
  ctx.n_stream = 1;
  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_seq_ids[0] = 0;
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_slots_request(
      emel::kv::cache::event::prepare_slots{.error_out = &err}, ctx));
  emel::kv::cache::action::reject_invalid_prepare_slots(
      emel::kv::cache::event::prepare_slots{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.ubatch_sizes[0] = 2;
  ctx.kv_size = 2;
  err = EMEL_OK;
  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 0, 0);
  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 1, 1);
  emel::kv::cache::action::run_prepare_slots(
      emel::kv::cache::event::prepare_slots{.error_out = &err}, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("kv_cache_prepare_slots_accepts_valid_input") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  ctx.ubatch_count = 1;
  ctx.ubatch_sizes[0] = 2;
  ctx.kv_size = 4;
  ctx.n_stream = 1;
  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_seq_ids[0] = 0;
  emel::kv::cache::action::run_prepare_slots(
    emel::kv::cache::event::prepare_slots{.error_out = &err}, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.planned_ubatch_count == 1);
}

TEST_CASE("kv_cache_run_apply_step_reports_errors") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  emel::kv::cache::event::apply_step step{
    .request = nullptr,
    .error_out = &err,
  };
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_step_request(step, ctx));
  emel::kv::cache::action::reject_invalid_apply_step(step, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  emel::kv::cache::event::apply_ubatch apply{};
  ctx.planned_ubatch_count = 1;
  apply.ubatch_index = 1;
  step.request = &apply;
  err = EMEL_OK;
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_step_request(step, ctx));
  emel::kv::cache::action::reject_invalid_apply_step(step, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  apply.ubatch_index = 0;
  ctx.kv_size = 1;
  ctx.ubatch_sizes[0] = 2;
  ctx.slot_offsets[0] = 0;
  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_seq_ids[0] = 0;
  err = EMEL_OK;
  ctx.ubatch_sizes[0] = 1;
  ctx.kv_size = 2;
  ctx.slot_offsets[0] = 0;
  ctx.streams[0].head = 0;
  ctx.streams[0].pos[0] = 3;
  emel::kv::cache::action::run_apply_step(
    step, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("kv_cache_run_apply_step_accepts_valid_input") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  emel::kv::cache::event::apply_ubatch apply{};
  emel::kv::cache::event::apply_step step{.request = &apply, .error_out = &err};
  ctx.planned_ubatch_count = 1;
  apply.ubatch_index = 0;
  ctx.applied_ubatches = 0;
  ctx.kv_size = 4;
  ctx.ubatch_sizes[0] = 1;
  ctx.slot_offsets[0] = 0;
  ctx.n_stream = 1;
  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_seq_ids[0] = 0;

  emel::kv::cache::action::run_apply_step(
    step, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.applied_ubatches == 1);
}

TEST_CASE("kv_cache_run_rollback_step_reports_errors") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  emel::kv::cache::event::rollback rollback{};
  emel::kv::cache::event::rollback_step step{.request = nullptr, .error_out = &err};
  CHECK_FALSE(emel::kv::cache::guard::valid_rollback_step_request(step, ctx));
  emel::kv::cache::action::reject_invalid_rollback_step(step, ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  rollback.from_ubatch_index = 0;
  step.request = &rollback;
  ctx.applied_ubatches = 1;
  ctx.planned_ubatch_count = 1;
  ctx.kv_size = 4;
  ctx.ubatch_sizes[0] = 1;
  ctx.slot_offsets[0] = 0;
  ctx.n_stream = 1;
  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_seq_ids[0] = 0;
  err = EMEL_OK;
  CHECK(emel::kv::cache::guard::valid_rollback_step_request(step, ctx));
}

TEST_CASE("kv_cache_run_rollback_step_accepts_valid_input") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  emel::kv::cache::event::rollback rollback{};
  emel::kv::cache::event::rollback_step step{.request = &rollback, .error_out = &err};
  rollback.from_ubatch_index = 0;
  ctx.applied_ubatches = 1;
  ctx.planned_ubatch_count = 1;
  ctx.kv_size = 4;
  ctx.ubatch_sizes[0] = 1;
  ctx.slot_offsets[0] = 0;
  ctx.n_stream = 1;
  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_seq_ids[0] = 0;
  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 0, 0);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[0], 0, 0);

  emel::kv::cache::action::run_rollback_step(
    step, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.applied_ubatches == 0);
}

TEST_CASE("kv_cache_on_done_and_error_reset_operation") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;

  ctx.applied_ubatches = 2;
  emel::kv::cache::action::on_kv_done(emel::kv::cache::events::kv_done{}, ctx);
  CHECK(ctx.applied_ubatches == 2);

  ctx.planned_ubatch_count = 3;
  emel::kv::cache::action::on_kv_error(emel::kv::cache::events::kv_error{}, ctx);
  CHECK(ctx.planned_ubatch_count == 3);
}

TEST_CASE("kv_cache_on_unexpected_sets_error_out") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;
  emel::kv::cache::event::prepare prepare{.error_out = &err};

  emel::kv::cache::action::on_unexpected{}(prepare, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("kv_cache_seq_operations_cover_branches") {
  using emel::kv::cache::action::POS_NONE;
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  ctx.kv_size = 4;
  ctx.n_stream = 2;
  ctx.seq_to_stream[0] = 0;
  ctx.seq_to_stream[1] = 1;

  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 0, 0);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[0], 0, 0);
  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 1, 1);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[0], 1, 0);

  emel::kv::cache::event::seq_remove remove{
    .seq_id = 0,
    .pos_start = 0,
    .pos_end = 1,
    .error_out = &err,
  };
  emel::kv::cache::event::seq_remove_step remove_step{
    .request = &remove,
    .error_out = &err,
  };
  emel::kv::cache::action::run_seq_remove_step(remove_step, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.streams[0].pos[0] == POS_NONE);

  emel::kv::cache::event::seq_copy copy{
    .seq_id_src = 0,
    .seq_id_dst = 1,
    .pos_start = POS_NONE,
    .pos_end = POS_NONE,
    .error_out = &err,
  };
  emel::kv::cache::event::seq_copy_step copy_step{
    .request = &copy,
    .error_out = &err,
  };
  emel::kv::cache::action::run_seq_copy_step(copy_step, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.streams[1].pos[1] != POS_NONE);

  emel::kv::cache::event::seq_keep keep{
    .seq_id = 1,
    .error_out = &err,
  };
  emel::kv::cache::event::seq_keep_step keep_step{
    .request = &keep,
    .error_out = &err,
  };
  emel::kv::cache::action::run_seq_keep_step(keep_step, ctx);
  CHECK(err == EMEL_OK);

  emel::kv::cache::event::seq_add add{
    .seq_id = 1,
    .pos_start = POS_NONE,
    .pos_end = POS_NONE,
    .shift = 2,
    .error_out = &err,
  };
  emel::kv::cache::event::seq_add_step add_step{
    .request = &add,
    .error_out = &err,
  };
  emel::kv::cache::action::run_seq_add_step(add_step, ctx);
  CHECK(err == EMEL_OK);

  emel::kv::cache::event::seq_div div{
    .seq_id = 1,
    .pos_start = POS_NONE,
    .pos_end = POS_NONE,
    .divisor = 2,
    .error_out = &err,
  };
  emel::kv::cache::event::seq_div_step div_step{
    .request = &div,
    .error_out = &err,
  };
  emel::kv::cache::action::run_seq_div_step(div_step, ctx);
  CHECK(err == EMEL_OK);

  ctx.streams[1].has_shift = true;
  ctx.streams[1].shift[1] = 3;
  emel::kv::cache::event::apply_updates updates{
    .error_out = &err,
  };
  emel::kv::cache::event::apply_updates_step updates_step{
    .request = &updates,
    .error_out = &err,
  };
  emel::kv::cache::action::run_apply_updates(updates_step, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.streams[1].has_shift == false);
  CHECK(ctx.pending_copy_count == 0);
}

TEST_CASE("kv_cache_detail_helpers_cover_branches") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  CHECK(emel::kv::cache::action::count_used_cells(ctx.streams[0]) == 0);
  CHECK(emel::kv::cache::action::used_max_p1(ctx.streams[0]) == 0);

  ctx.kv_size = 4;
  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 2, 5);
  CHECK(emel::kv::cache::action::count_used_cells(ctx.streams[0]) == 1);
  CHECK(emel::kv::cache::action::used_max_p1(ctx.streams[0]) == 3);

  int32_t head_after = 0;
  CHECK(emel::kv::cache::action::find_contiguous_slot(
          ctx, ctx.streams[0], 0, 0, 1, 0, 0, 0, head_after) == -1);
  CHECK(emel::kv::cache::action::find_contiguous_slot(
          ctx, ctx.streams[0], 4, 0, 5, 0, 0, 0, head_after) == -1);

  head_after = 0;
  ctx.streams[0].pos.fill(emel::kv::cache::action::POS_NONE);
  CHECK(emel::kv::cache::action::find_contiguous_slot(
          ctx, ctx.streams[0], 4, 3, 2, 0, 0, 0, head_after) == 0);
}
