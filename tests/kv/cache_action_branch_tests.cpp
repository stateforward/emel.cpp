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
  ctx.slot_index_count = 1;
  ctx.slot_indices[0] = 7;
  ctx.slot_stream_ids[0] = 0;
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
  ctx.slot_index_count = 1;
  ctx.slot_indices[0] = 0;
  ctx.slot_stream_ids[0] = 0;
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
  ctx.slot_index_count = 1;
  ctx.slot_indices[0] = 0;
  ctx.slot_stream_ids[0] = 0;
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
  ctx.slot_index_count = 1;
  ctx.slot_indices[0] = 0;
  ctx.slot_stream_ids[0] = 0;
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
  auto stream_copy_ok = [](int32_t, int32_t, void *, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return true;
  };
  auto apply_shift_ok = [](int32_t, const int32_t *, int32_t, void *, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return true;
  };
  emel::kv::cache::event::apply_updates updates{
    .stream_copy = stream_copy_ok,
    .apply_shift = apply_shift_ok,
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

TEST_CASE("kv_cache_snapshot_helpers_cover_branches") {
  emel::kv::cache::action::stream_state stream{};
  emel::kv::cache::action::reset_stream(stream);

  emel::kv::cache::action::set_cell_pos(stream, 3, 10);
  emel::kv::cache::action::set_cell_pos(stream, 3, 10);
  stream.shift[3] = 2;
  stream.ext_x[3] = 1;
  stream.ext_y[3] = 4;
  stream.seq_count[3] = 1;
  emel::kv::cache::action::set_seq_bit(stream, 3, 5);

  emel::kv::cache::action::cell_snapshot cell_snap{};
  emel::kv::cache::action::snapshot_cell(stream, 3, cell_snap);
  stream.pos[3] = emel::kv::cache::action::POS_NONE;
  stream.shift[3] = 0;
  stream.ext_x[3] = 0;
  stream.ext_y[3] = 0;
  stream.seq_count[3] = 0;
  stream.seq_mask[3].fill(0);
  emel::kv::cache::action::restore_cell(stream, 3, cell_snap);

  CHECK(stream.pos[3] == 10);
  CHECK(stream.shift[3] == 2);
  CHECK(emel::kv::cache::action::cell_has_seq(stream, 3, 5));
  CHECK_FALSE(emel::kv::cache::action::cell_has_seq(stream, 3, -1));
  CHECK_FALSE(emel::kv::cache::action::cell_has_seq(
    stream, 3, emel::kv::cache::action::MAX_SEQ));
  emel::kv::cache::action::clear_seq_bit(stream, 3, 5);
  CHECK_FALSE(emel::kv::cache::action::cell_has_seq(stream, 3, 5));
  emel::kv::cache::action::set_seq_bit(stream, 3, 5);
  CHECK(emel::kv::cache::action::single_seq_id(stream, 3) == 5);
  stream.seq_count[3] = 2;
  CHECK(emel::kv::cache::action::single_seq_id(stream, 3) ==
        emel::kv::cache::action::POS_NONE);
  emel::kv::cache::action::add_seq_to_cell(stream, 3, 5);
  stream.seq_count[3] = 1;
  stream.seq_mask[3].fill(0);
  CHECK(emel::kv::cache::action::single_seq_id(stream, 3) ==
        emel::kv::cache::action::POS_NONE);

  emel::kv::cache::action::stream_snapshot stream_snap{};
  stream.head = 2;
  stream.used_count = 1;
  stream.used_max_p1 = 4;
  stream.has_shift = true;
  stream.seq_pos_min[5] = 10;
  stream.seq_pos_max[5] = 12;
  stream.seq_pos_min_count[5] = 1;
  stream.seq_pos_max_count[5] = 2;
  emel::kv::cache::action::snapshot_stream_state(stream, stream_snap);
  stream.head = 0;
  stream.used_count = 0;
  stream.used_max_p1 = 0;
  stream.has_shift = false;
  stream.seq_pos_min[5] = emel::kv::cache::action::POS_NONE;
  stream.seq_pos_max[5] = emel::kv::cache::action::POS_NONE;
  stream.seq_pos_min_count[5] = 0;
  stream.seq_pos_max_count[5] = 0;
  emel::kv::cache::action::restore_stream_state(stream, stream_snap);
  CHECK(stream.head == 2);
  CHECK(stream.used_count == 1);
  CHECK(stream.used_max_p1 == 4);
  CHECK(stream.has_shift);
  CHECK(stream.seq_pos_min[5] == 10);
  CHECK(stream.seq_pos_max[5] == 12);

  CHECK(emel::kv::cache::action::pad_to(5, 4) == 8);
  CHECK(emel::kv::cache::action::pad_to(5, 0) == 5);
  CHECK(emel::kv::cache::action::pad_to(5, -1) == 5);

  CHECK(emel::kv::cache::action::ranges_overlap(0, 2, 1, 2));
  CHECK_FALSE(emel::kv::cache::action::ranges_overlap(0, 1, 1, 1));

  CHECK(emel::kv::cache::action::pos_in_range(3, -1, -1));
  CHECK(emel::kv::cache::action::pos_in_range(3, -1, 4));
  CHECK(emel::kv::cache::action::pos_in_range(3, 3, -1));
  CHECK_FALSE(emel::kv::cache::action::pos_in_range(3, 4, -1));
  CHECK_FALSE(emel::kv::cache::action::pos_in_range(3, 0, 3));

  CHECK_FALSE(emel::kv::cache::action::is_full_copy_range(0, 0, 0));
  CHECK(emel::kv::cache::action::is_full_copy_range(0, 7, 8));
  CHECK_FALSE(emel::kv::cache::action::is_full_copy_range(1, 6, 8));

  emel::kv::cache::action::seq_pos_add(stream, emel::kv::cache::action::MAX_SEQ, 1);
  emel::kv::cache::action::seq_pos_add(stream, 2, -1);
  emel::kv::cache::action::seq_pos_remove(stream, emel::kv::cache::action::MAX_SEQ, 1);
  emel::kv::cache::action::seq_pos_remove(stream, 2, -1);

  emel::kv::cache::action::reset_stream(stream);
  emel::kv::cache::action::set_cell_pos(stream, 1, 5);
  emel::kv::cache::action::add_seq_to_cell(stream, 1, 2);
  emel::kv::cache::action::set_cell_pos(stream, 2, 7);
  emel::kv::cache::action::add_seq_to_cell(stream, 2, 2);
  emel::kv::cache::action::recompute_seq_pos_min(stream, 2);
  emel::kv::cache::action::recompute_seq_pos_max(stream, 2);
  CHECK(stream.seq_pos_min[2] == 5);
  CHECK(stream.seq_pos_max[2] == 7);

  stream.seq_pos_min.fill(emel::kv::cache::action::POS_NONE);
  stream.seq_pos_max.fill(emel::kv::cache::action::POS_NONE);
  stream.seq_pos_min_count.fill(0);
  stream.seq_pos_max_count.fill(0);
  emel::kv::cache::action::seq_pos_add(stream, 2, 5);
  emel::kv::cache::action::seq_pos_add(stream, 2, 5);
  emel::kv::cache::action::seq_pos_add(stream, 2, 7);
  CHECK(stream.seq_pos_min[2] == 5);
  CHECK(stream.seq_pos_min_count[2] == 2);
  CHECK(stream.seq_pos_max[2] == 7);
  CHECK(stream.seq_pos_max_count[2] == 1);
  emel::kv::cache::action::seq_pos_remove(stream, 2, 5);
  CHECK(stream.seq_pos_min_count[2] == 1);
  emel::kv::cache::action::clear_seq_bit(stream, 1, 2);
  stream.seq_count[1] = 0;
  emel::kv::cache::action::seq_pos_remove(stream, 2, 5);
  CHECK(stream.seq_pos_min[2] == 7);

  emel::kv::cache::action::reset_stream(stream);
  emel::kv::cache::action::set_cell_pos(stream, 0, 0);
  emel::kv::cache::action::add_seq_to_cell(stream, 0, 1);
  emel::kv::cache::action::set_cell_pos(stream, 3, 3);
  emel::kv::cache::action::add_seq_to_cell(stream, 3, 1);
  stream.used_count = 2;
  stream.used_max_p1 = 4;
  emel::kv::cache::action::set_cell_empty(stream, 3);
  CHECK(stream.pos[3] == emel::kv::cache::action::POS_NONE);
  CHECK(stream.used_count == 1);
  CHECK(stream.used_max_p1 == 1);
}

TEST_CASE("kv_cache_seq_copy_cross_stream_full_range") {
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

  emel::kv::cache::action::set_cell_pos(ctx.streams[1], 0, 0);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[1], 0, 1);

  emel::kv::cache::event::seq_copy copy{
    .seq_id_src = 0,
    .seq_id_dst = 1,
    .pos_start = emel::kv::cache::action::POS_NONE,
    .pos_end = emel::kv::cache::action::POS_NONE,
    .error_out = &err,
  };
  emel::kv::cache::event::seq_copy_step copy_step{
    .request = &copy,
    .error_out = &err,
  };
  emel::kv::cache::action::run_seq_copy_step(copy_step, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.pending_copy_count == 1);
  CHECK(ctx.pending_copy_src[0] == 0);
  CHECK(ctx.pending_copy_dst[0] == 1);
}

TEST_CASE("kv_cache_seq_noop_branches") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  ctx.kv_size = 4;
  ctx.n_stream = 1;
  ctx.seq_to_stream[0] = 0;

  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 0, 0);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[0], 0, 0);

  emel::kv::cache::event::seq_add add{
    .seq_id = 0,
    .pos_start = emel::kv::cache::action::POS_NONE,
    .pos_end = emel::kv::cache::action::POS_NONE,
    .shift = 0,
    .error_out = &err,
  };
  emel::kv::cache::event::seq_add_step add_step{
    .request = &add,
    .error_out = &err,
  };
  emel::kv::cache::action::run_seq_add_step(add_step, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.streams[0].pos[0] == 0);

  emel::kv::cache::event::seq_div div{
    .seq_id = 0,
    .pos_start = emel::kv::cache::action::POS_NONE,
    .pos_end = emel::kv::cache::action::POS_NONE,
    .divisor = 1,
    .error_out = &err,
  };
  emel::kv::cache::event::seq_div_step div_step{
    .request = &div,
    .error_out = &err,
  };
  emel::kv::cache::action::run_seq_div_step(div_step, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.streams[0].pos[0] == 0);
}

TEST_CASE("kv_cache_apply_updates_copy_branches") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  ctx.n_stream = 2;
  ctx.pending_copy_count = 1;
  ctx.pending_copy_src[0] = 0;
  ctx.pending_copy_dst[0] = 1;

  auto stream_copy_ok = [](int32_t, int32_t, void *, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_OK;
    }
    return true;
  };

  emel::kv::cache::event::apply_updates updates{
    .stream_copy = stream_copy_ok,
    .error_out = &err,
  };
  emel::kv::cache::event::apply_updates_step updates_step{
    .request = &updates,
    .error_out = &err,
  };
  emel::kv::cache::action::run_apply_updates(updates_step, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.pending_copy_count == 0);

  ctx.pending_copy_count = 1;
  ctx.pending_copy_src[0] = 0;
  ctx.pending_copy_dst[0] = 1;
  auto stream_copy_fail = [](int32_t, int32_t, void *, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_BACKEND;
    }
    return false;
  };
  updates.stream_copy = stream_copy_fail;
  err = EMEL_OK;
  emel::kv::cache::action::run_apply_updates(updates_step, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("kv_cache_run_apply_step_with_positions") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  ctx.kv_size = 8;
  ctx.n_stream = 1;
  ctx.planned_ubatch_count = 1;
  ctx.applied_ubatches = 0;
  ctx.ubatch_sizes[0] = 2;
  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_seq_ids[0] = 0;
  ctx.seq_to_stream[0] = 0;
  ctx.slot_offsets[0] = 0;
  ctx.slot_index_count = 2;
  ctx.slot_indices[0] = 1;
  ctx.slot_indices[1] = 2;
  ctx.slot_stream_ids[0] = 0;
  ctx.slot_stream_ids[1] = 0;

  std::array<int32_t, 2> positions = {{7, 3}};
  emel::kv::cache::event::apply_ubatch apply{
    .ubatch_index = 0,
    .positions = positions.data(),
    .positions_count = static_cast<int32_t>(positions.size()),
  };
  emel::kv::cache::event::apply_step step{
    .request = &apply,
    .error_out = &err,
  };
  emel::kv::cache::action::run_apply_step(step, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.streams[0].pos[1] == 7);
  CHECK(ctx.streams[0].pos[2] == 3);
}

TEST_CASE("kv_cache_seq_remove_all_sequences") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  ctx.kv_size = 4;
  ctx.n_stream = 1;
  ctx.seq_to_stream[0] = 0;
  ctx.seq_to_stream[1] = 0;

  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 0, 0);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[0], 0, 0);
  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 1, 1);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[0], 1, 1);

  emel::kv::cache::event::seq_remove remove{
    .seq_id = -1,
    .pos_start = emel::kv::cache::action::POS_NONE,
    .pos_end = emel::kv::cache::action::POS_NONE,
    .error_out = &err,
  };
  emel::kv::cache::event::seq_remove_step remove_step{
    .request = &remove,
    .error_out = &err,
  };
  emel::kv::cache::action::run_seq_remove_step(remove_step, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.streams[0].pos[0] == emel::kv::cache::action::POS_NONE);
  CHECK(ctx.streams[0].pos[1] == emel::kv::cache::action::POS_NONE);
}

TEST_CASE("kv_cache_apply_updates_shift_failure") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  ctx.n_stream = 1;
  ctx.streams[0].has_shift = true;
  ctx.streams[0].shift[0] = 1;

  auto apply_shift_fail = [](int32_t, const int32_t *, int32_t, void *, int32_t * err_out) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_BACKEND;
    }
    return false;
  };

  emel::kv::cache::event::apply_updates updates{
    .apply_shift = apply_shift_fail,
    .error_out = &err,
  };
  emel::kv::cache::event::apply_updates_step updates_step{
    .request = &updates,
    .error_out = &err,
  };
  emel::kv::cache::action::run_apply_updates(updates_step, ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("kv_cache_run_apply_step_overwrites_existing_cell") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  ctx.kv_size = 4;
  ctx.n_stream = 1;
  ctx.planned_ubatch_count = 1;
  ctx.applied_ubatches = 0;
  ctx.ubatch_sizes[0] = 1;
  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_seq_ids[0] = 1;
  ctx.seq_to_stream[0] = 0;
  ctx.seq_to_stream[1] = 0;
  ctx.slot_offsets[0] = 0;
  ctx.slot_index_count = 1;
  ctx.slot_indices[0] = 1;
  ctx.slot_stream_ids[0] = 0;

  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 1, 0);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[0], 1, 0);

  emel::kv::cache::event::apply_ubatch apply{.ubatch_index = 0};
  emel::kv::cache::event::apply_step step{.request = &apply, .error_out = &err};
  emel::kv::cache::action::run_apply_step(step, ctx);
  CHECK(err == EMEL_OK);
  CHECK(emel::kv::cache::action::cell_has_seq(ctx.streams[0], 1, 1));
  CHECK_FALSE(emel::kv::cache::action::cell_has_seq(ctx.streams[0], 1, 0));
}

TEST_CASE("kv_cache_prepare_slots_multi_stream_swa") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  ctx.kv_size = 6;
  ctx.n_stream = 2;
  ctx.n_swa = 2;
  ctx.swa_type = 1;
  ctx.ubatch_count = 2;
  ctx.ubatch_sizes[0] = 2;
  ctx.ubatch_sizes[1] = 2;
  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_stream_ids[1] = 1;
  ctx.ubatch_seq_ids[0] = 0;
  ctx.ubatch_seq_ids[1] = 1;
  ctx.seq_to_stream[0] = 0;
  ctx.seq_to_stream[1] = 1;

  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 0, 0);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[0], 0, 0);
  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 1, 1);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[0], 1, 0);
  emel::kv::cache::action::set_cell_pos(ctx.streams[1], 0, 0);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[1], 0, 1);
  emel::kv::cache::action::set_cell_pos(ctx.streams[1], 1, 1);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[1], 1, 1);

  emel::kv::cache::action::run_prepare_slots(
    emel::kv::cache::event::prepare_slots{.error_out = &err}, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.planned_ubatch_count == 2);
  CHECK(ctx.slot_index_count == 4);
}

TEST_CASE("kv_cache_run_prepare_slots_multiple_ubatches") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  ctx.kv_size = 8;
  ctx.n_stream = 1;
  ctx.ubatch_count = 2;
  ctx.ubatch_sizes[0] = 2;
  ctx.ubatch_sizes[1] = 1;
  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_stream_ids[1] = 0;
  ctx.ubatch_seq_ids[0] = 0;
  ctx.ubatch_seq_ids[1] = 0;
  ctx.seq_to_stream[0] = 0;

  emel::kv::cache::action::run_prepare_slots(
    emel::kv::cache::event::prepare_slots{.error_out = &err}, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.planned_ubatch_count == 2);
  CHECK(ctx.slot_offsets[0] == 0);
  CHECK(ctx.slot_offsets[1] == 2);
  CHECK(ctx.slot_index_count == 3);
}

TEST_CASE("kv_cache_run_rollback_step_multiple_ubatches") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  ctx.kv_size = 6;
  ctx.n_stream = 1;
  ctx.planned_ubatch_count = 2;
  ctx.applied_ubatches = 2;
  ctx.ubatch_sizes[0] = 1;
  ctx.ubatch_sizes[1] = 2;
  ctx.slot_offsets[0] = 0;
  ctx.slot_offsets[1] = 1;
  ctx.slot_index_count = 3;
  ctx.slot_indices[0] = 0;
  ctx.slot_indices[1] = 1;
  ctx.slot_indices[2] = 2;
  ctx.slot_stream_ids[0] = 0;
  ctx.slot_stream_ids[1] = 0;
  ctx.slot_stream_ids[2] = 0;
  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_stream_ids[1] = 0;
  ctx.ubatch_seq_ids[0] = 0;
  ctx.ubatch_seq_ids[1] = 0;
  ctx.seq_to_stream[0] = 0;

  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 0, 0);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[0], 0, 0);
  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 1, 1);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[0], 1, 0);
  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 2, 2);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[0], 2, 0);

  emel::kv::cache::event::rollback rollback{.from_ubatch_index = 0};
  emel::kv::cache::event::rollback_step step{
    .request = &rollback,
    .error_out = &err,
  };
  emel::kv::cache::action::run_rollback_step(step, ctx);
  CHECK(err == EMEL_OK);
  CHECK(ctx.applied_ubatches == 0);
  CHECK(ctx.streams[0].pos[0] == emel::kv::cache::action::POS_NONE);
  CHECK(ctx.streams[0].pos[1] == emel::kv::cache::action::POS_NONE);
  CHECK(ctx.streams[0].pos[2] == emel::kv::cache::action::POS_NONE);
}

TEST_CASE("kv_cache_internal_helpers_cover_more_branches") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  auto stream_storage = std::make_unique<emel::kv::cache::action::stream_state>();
  emel::kv::cache::action::stream_state & stream = *stream_storage;
  emel::kv::cache::action::reset_stream(stream);

  emel::kv::cache::action::remove_seq_from_cell(stream, 0, 0);
  emel::kv::cache::action::set_cell_pos(stream, 0, 1);
  emel::kv::cache::action::add_seq_to_cell(stream, 0, 0);
  emel::kv::cache::action::remove_seq_from_cell(stream, 0, 0);
  CHECK(stream.pos[0] == emel::kv::cache::action::POS_NONE);

  CHECK_FALSE(emel::kv::cache::action::pos_add_cell(stream, 1, 1));
  emel::kv::cache::action::set_cell_pos(stream, 1, 1);
  emel::kv::cache::action::add_seq_to_cell(stream, 1, 0);
  CHECK(emel::kv::cache::action::pos_add_cell(stream, 1, -5));
  emel::kv::cache::action::set_cell_pos(stream, 1, 2);
  emel::kv::cache::action::add_seq_to_cell(stream, 1, 0);
  CHECK_FALSE(emel::kv::cache::action::pos_add_cell(stream, 1, 1));
  CHECK(stream.pos[1] == 3);
  CHECK(stream.has_shift);

  emel::kv::cache::action::pos_div_cell(stream, 2, 2);
  emel::kv::cache::action::set_cell_pos(stream, 2, 4);
  emel::kv::cache::action::add_seq_to_cell(stream, 2, 0);
  emel::kv::cache::action::pos_div_cell(stream, 2, 2);
  CHECK(stream.pos[2] == 2);

  ctx.n_stream = 2;
  ctx.streams[0].used_max_p1 = 3;
  ctx.streams[1].used_max_p1 = 5;
  CHECK(emel::kv::cache::action::max_used_max_p1(ctx) == 5);

  ctx.kv_size = 8;
  ctx.n_pad = 4;
  ctx.streams[0].used_max_p1 = 3;
  ctx.streams[1].used_max_p1 = 6;
  CHECK(emel::kv::cache::action::compute_kv_tokens(ctx) == 8);
  ctx.kv_size = 0;
  CHECK(emel::kv::cache::action::compute_kv_tokens(ctx) == 0);

  CHECK_FALSE(emel::kv::cache::action::is_masked_swa(0, 1, 0, 4));
  CHECK_FALSE(emel::kv::cache::action::is_masked_swa(2, 0, 0, 4));
  CHECK_FALSE(emel::kv::cache::action::is_masked_swa(2, 1, 0, 0));
  CHECK(emel::kv::cache::action::is_masked_swa(2, 1, 0, 5));

  ctx.ubatch_stream_ids[0] = 0;
  ctx.slot_offsets[0] = 2;
  ctx.ubatch_sizes[0] = 2;
  CHECK(emel::kv::cache::action::range_overlaps_planned(ctx, 0, 3, 1, 1));
  CHECK_FALSE(emel::kv::cache::action::range_overlaps_planned(ctx, 0, 0, 1, 1));

  auto ctx_slots_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx_slots = *ctx_slots_storage;
  auto stream_slots_storage = std::make_unique<emel::kv::cache::action::stream_state>();
  emel::kv::cache::action::stream_state & stream_slots = *stream_slots_storage;
  emel::kv::cache::action::reset_stream(stream_slots);
  ctx_slots.kv_size = 4;
  int32_t head_after = 0;
  CHECK(emel::kv::cache::action::find_contiguous_slot(
          ctx_slots, stream_slots, 4, 10, 2, 0, 0, 0, head_after) == 0);

  ctx_slots.ubatch_stream_ids[0] = 0;
  ctx_slots.slot_offsets[0] = 1;
  ctx_slots.ubatch_sizes[0] = 2;
  CHECK(emel::kv::cache::action::find_contiguous_slot(
          ctx_slots, stream_slots, 4, 0, 2, 0, 0, 1, head_after) == -1);

  auto ctx_find_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx_find = *ctx_find_storage;
  ctx_find.kv_size = 4;
  ctx_find.n_swa = 2;
  ctx_find.swa_type = 1;
  auto stream_find_storage = std::make_unique<emel::kv::cache::action::stream_state>();
  emel::kv::cache::action::stream_state & stream_find = *stream_find_storage;
  emel::kv::cache::action::reset_stream(stream_find);
  emel::kv::cache::action::set_cell_pos(stream_find, 0, 0);
  emel::kv::cache::action::add_seq_to_cell(stream_find, 0, 0);
  stream_find.seq_pos_max[0] = 4;
  int32_t indices[1] = {0};
  CHECK(emel::kv::cache::action::find_slot_indices(
          ctx_find, stream_find, 4, 1, indices));
  CHECK(indices[0] == 0);

  ctx_find.next_pos[0] = 0;
  emel::kv::cache::action::ensure_next_pos_for_seq(ctx_find, 0, stream_find);
  CHECK(ctx_find.next_pos[0] == 5);
  emel::kv::cache::action::reset_next_pos_for_seq(ctx_find, 0, stream_find);
  CHECK(ctx_find.next_pos[0] == 5);
  stream_find.seq_pos_max[0] = emel::kv::cache::action::POS_NONE;
  emel::kv::cache::action::reset_next_pos_for_seq(ctx_find, 0, stream_find);
  CHECK(ctx_find.next_pos[0] == 0);

  ctx_find.pending_copy_count = 0;
  emel::kv::cache::action::add_pending_copy(ctx_find, 0, 0);
  CHECK(ctx_find.pending_copy_count == 0);
  emel::kv::cache::action::add_pending_copy(ctx_find, 0, 1);
  CHECK(ctx_find.pending_copy_count == 1);
  emel::kv::cache::action::add_pending_copy(ctx_find, 0, 1);
  CHECK(ctx_find.pending_copy_count == 1);

  auto ctx_apply_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx_apply = *ctx_apply_storage;
  ctx_apply.kv_size = 4;
  ctx_apply.n_stream = 1;
  ctx_apply.seq_to_stream[0] = 0;
  ctx_apply.seq_to_stream[1] = 0;
  ctx_apply.slot_index_count = 2;
  ctx_apply.slot_indices[0] = 0;
  ctx_apply.slot_indices[1] = 1;
  ctx_apply.slot_stream_ids[0] = 0;
  ctx_apply.slot_stream_ids[1] = 0;
  emel::kv::cache::action::set_cell_pos(ctx_apply.streams[0], 0, 1);
  emel::kv::cache::action::add_seq_to_cell(ctx_apply.streams[0], 0, 1);
  ctx_apply.streams[0].seq_pos_min[1] = 1;
  ctx_apply.streams[0].seq_pos_max[1] = 1;
  std::array<int32_t, 6> positions = {{2, 3, 10, 11, 20, 21}};
  CHECK(emel::kv::cache::action::apply_slots(
    ctx_apply, 0, 2, 0, positions.data(),
    static_cast<int32_t>(positions.size()), true));
  CHECK(ctx_apply.streams[0].ext_y[0] == 10);
  CHECK(ctx_apply.streams[0].ext_x[0] == 20);
  CHECK(ctx_apply.next_pos[0] == 4);
}

TEST_CASE("kv_cache_seq_copy_partial_range_same_stream") {
  auto ctx_storage = std::make_unique<emel::kv::cache::action::context>();
  emel::kv::cache::action::context & ctx = *ctx_storage;
  int32_t err = EMEL_OK;

  ctx.kv_size = 6;
  ctx.n_stream = 1;
  ctx.seq_to_stream[0] = 0;
  ctx.seq_to_stream[1] = 0;

  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 0, 0);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[0], 0, 0);
  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 1, 1);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[0], 1, 0);

  emel::kv::cache::event::seq_copy copy{
    .seq_id_src = 0,
    .seq_id_dst = 1,
    .pos_start = 0,
    .pos_end = 1,
    .error_out = &err,
  };
  emel::kv::cache::event::seq_copy_step copy_step{
    .request = &copy,
    .error_out = &err,
  };
  emel::kv::cache::action::run_seq_copy_step(copy_step, ctx);
  CHECK(err == EMEL_OK);
  CHECK(emel::kv::cache::action::cell_has_seq(ctx.streams[0], 0, 1));
}
