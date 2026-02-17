#include <array>
#include <memory>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/kv/cache/actions.hpp"
#include "emel/kv/cache/events.hpp"
#include "emel/kv/cache/guards.hpp"

namespace {

using context = emel::kv::cache::action::context;

std::unique_ptr<context> make_context() {
  auto ctx = std::make_unique<context>();
  ctx->kv_size = 8;
  ctx->n_stream = 1;
  ctx->ubatch_count = 1;
  ctx->planned_ubatch_count = 1;
  ctx->applied_ubatches = 0;
  ctx->ubatch_sizes[0] = 1;
  ctx->slot_offsets[0] = 0;
  ctx->ubatch_stream_ids[0] = 0;
  ctx->ubatch_seq_ids[0] = 0;
  ctx->seq_to_stream[0] = 0;
  return ctx;
}

}  // namespace

TEST_CASE("kv_cache_guard_valid_pos_range_cases") {
  CHECK(emel::kv::cache::guard::valid_pos_range(-1, -1));
  CHECK(emel::kv::cache::guard::valid_pos_range(-1, 4));
  CHECK(emel::kv::cache::guard::valid_pos_range(4, -1));
  CHECK(emel::kv::cache::guard::valid_pos_range(0, 0));
  CHECK_FALSE(emel::kv::cache::guard::valid_pos_range(5, 4));
}

TEST_CASE("kv_cache_guard_prepare_request_branches") {
  auto ctx_storage = make_context();
  context & ctx = *ctx_storage;

  std::array<int32_t, 1> sizes = {{1}};
  std::array<int32_t, 1> streams = {{0}};
  std::array<int32_t, 1> seqs = {{0}};
  std::array<int32_t, 1> seq_to_stream = {{0}};

  emel::kv::cache::event::prepare req{
    .ubatch_sizes = sizes.data(),
    .ubatch_count = 1,
    .requested_capacity = 4,
    .seq_to_stream = seq_to_stream.data(),
    .seq_to_stream_count = 1,
    .ubatch_stream_ids = streams.data(),
    .ubatch_stream_ids_count = 1,
    .ubatch_seq_ids = seqs.data(),
    .ubatch_seq_ids_count = 1,
  };

  emel::kv::cache::action::begin_prepare(req, ctx);
  emel::kv::cache::event::validate_prepare validate{.request = &req};
  CHECK(emel::kv::cache::guard::valid_prepare_request(validate, ctx));
  CHECK_FALSE(emel::kv::cache::guard::invalid_prepare_request(validate, ctx));

  emel::kv::cache::event::validate_prepare null_validate{.request = nullptr};
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(null_validate, ctx));

  req.ubatch_count = 0;
  emel::kv::cache::action::begin_prepare(req, ctx);
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));

  req.ubatch_count = emel::kv::cache::action::MAX_UBATCHES + 1;
  emel::kv::cache::action::begin_prepare(req, ctx);
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));

  req.ubatch_count = 1;
  req.ubatch_sizes = nullptr;
  emel::kv::cache::action::begin_prepare(req, ctx);
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));

  req.ubatch_sizes = sizes.data();
  req.requested_capacity = emel::kv::cache::action::MAX_KV_CELLS + 1;
  emel::kv::cache::action::begin_prepare(req, ctx);
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));

  req.requested_capacity = 4;
  ctx.n_stream = 0;
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));

  ctx.n_stream = emel::kv::cache::action::MAX_STREAMS + 1;
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));

  ctx.n_stream = 1;
  req.ubatch_stream_ids = streams.data();
  req.ubatch_stream_ids_count = 0;
  emel::kv::cache::action::begin_prepare(req, ctx);
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));

  req.ubatch_stream_ids_count = 1;
  req.ubatch_seq_ids = seqs.data();
  req.ubatch_seq_ids_count = 0;
  emel::kv::cache::action::begin_prepare(req, ctx);
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));

  req.ubatch_seq_ids_count = 1;
  seq_to_stream[0] = 2;
  emel::kv::cache::action::begin_prepare(req, ctx);
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));

  seq_to_stream[0] = 0;
  ctx.kv_size = 0;
  req.requested_capacity = 0;
  emel::kv::cache::action::begin_prepare(req, ctx);
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));

  ctx.kv_size = emel::kv::cache::action::MAX_KV_CELLS + 1;
  emel::kv::cache::action::begin_prepare(req, ctx);
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));

  ctx.kv_size = 4;
  sizes[0] = 0;
  req.requested_capacity = 4;
  emel::kv::cache::action::begin_prepare(req, ctx);
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));

  sizes[0] = 5;
  ctx.kv_size = 4;
  emel::kv::cache::action::begin_prepare(req, ctx);
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));

  sizes[0] = 1;
  ctx.kv_size = 4;
  ctx.ubatch_stream_ids[0] = 2;
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));

  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_seq_ids[0] = emel::kv::cache::action::MAX_SEQ;
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));

  ctx.ubatch_seq_ids[0] = 0;
  ctx.seq_to_stream[0] = 1;
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_request(validate, ctx));
}

TEST_CASE("kv_cache_guard_prepare_slots_request_branches") {
  auto ctx_storage = make_context();
  context & ctx = *ctx_storage;

  ctx.kv_size = 4;
  ctx.ubatch_count = 1;
  ctx.n_stream = 1;
  ctx.ubatch_sizes[0] = 1;
  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_seq_ids[0] = 0;
  ctx.seq_to_stream[0] = 0;

  CHECK(emel::kv::cache::guard::valid_prepare_slots_request(
      emel::kv::cache::event::prepare_slots{}, ctx));
  CHECK_FALSE(emel::kv::cache::guard::invalid_prepare_slots_request(
      emel::kv::cache::event::prepare_slots{}, ctx));

  ctx.kv_size = 0;
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_slots_request(
      emel::kv::cache::event::prepare_slots{}, ctx));

  ctx.kv_size = 4;
  ctx.ubatch_count = 0;
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_slots_request(
      emel::kv::cache::event::prepare_slots{}, ctx));

  ctx.ubatch_count = 1;
  ctx.n_stream = 0;
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_slots_request(
      emel::kv::cache::event::prepare_slots{}, ctx));

  ctx.n_stream = 1;
  ctx.ubatch_sizes[0] = 0;
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_slots_request(
      emel::kv::cache::event::prepare_slots{}, ctx));

  ctx.ubatch_sizes[0] = 5;
  ctx.kv_size = 4;
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_slots_request(
      emel::kv::cache::event::prepare_slots{}, ctx));

  ctx.ubatch_sizes[0] = 1;
  ctx.kv_size = 4;
  ctx.ubatch_stream_ids[0] = 2;
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_slots_request(
      emel::kv::cache::event::prepare_slots{}, ctx));

  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_seq_ids[0] = emel::kv::cache::action::MAX_SEQ;
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_slots_request(
      emel::kv::cache::event::prepare_slots{}, ctx));

  ctx.ubatch_seq_ids[0] = 0;
  ctx.seq_to_stream[0] = 1;
  CHECK_FALSE(emel::kv::cache::guard::valid_prepare_slots_request(
      emel::kv::cache::event::prepare_slots{}, ctx));
}

TEST_CASE("kv_cache_guard_apply_request_branches") {
  auto ctx_storage = make_context();
  context & ctx = *ctx_storage;

  emel::kv::cache::event::apply_ubatch apply{.ubatch_index = 0};
  emel::kv::cache::event::validate_apply validate{.request = &apply};
  CHECK(emel::kv::cache::guard::valid_apply_request(validate, ctx));
  CHECK_FALSE(emel::kv::cache::guard::invalid_apply_request(validate, ctx));

  emel::kv::cache::event::validate_apply null_validate{.request = nullptr};
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_request(null_validate, ctx));

  ctx.planned_ubatch_count = 0;
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_request(validate, ctx));

  ctx.planned_ubatch_count = 1;
  apply.ubatch_index = -1;
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_request(validate, ctx));

  apply.ubatch_index = 2;
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_request(validate, ctx));

  apply.ubatch_index = 0;
  ctx.applied_ubatches = 1;
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_request(validate, ctx));
}

TEST_CASE("kv_cache_guard_apply_step_request_branches") {
  auto ctx_storage = make_context();
  context & ctx = *ctx_storage;

  ctx.kv_size = 8;
  ctx.planned_ubatch_count = 1;
  ctx.applied_ubatches = 0;
  ctx.ubatch_sizes[0] = 2;
  ctx.slot_offsets[0] = 0;
  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_seq_ids[0] = 0;

  emel::kv::cache::event::apply_ubatch apply{.ubatch_index = 0};
  emel::kv::cache::event::apply_step step{.request = &apply};
  CHECK(emel::kv::cache::guard::valid_apply_step_request(step, ctx));
  CHECK_FALSE(emel::kv::cache::guard::invalid_apply_step_request(step, ctx));

  emel::kv::cache::event::apply_step null_step{.request = nullptr};
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_step_request(null_step, ctx));

  ctx.planned_ubatch_count = 0;
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_step_request(step, ctx));

  ctx.planned_ubatch_count = 1;
  apply.ubatch_index = -1;
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_step_request(step, ctx));

  apply.ubatch_index = 2;
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_step_request(step, ctx));

  apply.ubatch_index = 0;
  ctx.applied_ubatches = 1;
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_step_request(step, ctx));

  ctx.applied_ubatches = 0;
  ctx.ubatch_sizes[0] = 0;
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_step_request(step, ctx));

  ctx.ubatch_sizes[0] = 2;
  ctx.slot_offsets[0] = -1;
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_step_request(step, ctx));

  ctx.slot_offsets[0] = 7;
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_step_request(step, ctx));

  ctx.slot_offsets[0] = 0;
  ctx.ubatch_stream_ids[0] = 2;
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_step_request(step, ctx));

  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_seq_ids[0] = emel::kv::cache::action::MAX_SEQ;
  CHECK_FALSE(emel::kv::cache::guard::valid_apply_step_request(step, ctx));
}

TEST_CASE("kv_cache_guard_rollback_request_branches") {
  auto ctx_storage = make_context();
  context & ctx = *ctx_storage;
  ctx.planned_ubatch_count = 2;
  ctx.applied_ubatches = 1;

  emel::kv::cache::event::rollback rollback{.from_ubatch_index = 0};
  emel::kv::cache::event::validate_rollback validate{.request = &rollback};
  CHECK(emel::kv::cache::guard::valid_rollback_request(validate, ctx));
  CHECK_FALSE(emel::kv::cache::guard::invalid_rollback_request(validate, ctx));

  emel::kv::cache::event::validate_rollback null_validate{.request = nullptr};
  CHECK_FALSE(emel::kv::cache::guard::valid_rollback_request(null_validate, ctx));

  rollback.from_ubatch_index = -1;
  CHECK_FALSE(emel::kv::cache::guard::valid_rollback_request(validate, ctx));

  rollback.from_ubatch_index = 3;
  CHECK_FALSE(emel::kv::cache::guard::valid_rollback_request(validate, ctx));

  rollback.from_ubatch_index = 2;
  ctx.applied_ubatches = 1;
  CHECK_FALSE(emel::kv::cache::guard::valid_rollback_request(validate, ctx));
}

TEST_CASE("kv_cache_guard_rollback_step_request_branches") {
  auto ctx_storage = make_context();
  context & ctx = *ctx_storage;
  ctx.planned_ubatch_count = 1;
  ctx.applied_ubatches = 1;
  ctx.kv_size = 4;
  ctx.ubatch_sizes[0] = 1;
  ctx.slot_offsets[0] = 0;
  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_seq_ids[0] = 0;

  emel::kv::cache::event::rollback rollback{.from_ubatch_index = 0};
  emel::kv::cache::event::rollback_step step{.request = &rollback};
  CHECK(emel::kv::cache::guard::valid_rollback_step_request(step, ctx));
  CHECK_FALSE(emel::kv::cache::guard::invalid_rollback_step_request(step, ctx));

  emel::kv::cache::event::rollback_step null_step{.request = nullptr};
  CHECK_FALSE(emel::kv::cache::guard::valid_rollback_step_request(null_step, ctx));

  rollback.from_ubatch_index = -1;
  CHECK_FALSE(emel::kv::cache::guard::valid_rollback_step_request(step, ctx));

  rollback.from_ubatch_index = 2;
  CHECK_FALSE(emel::kv::cache::guard::valid_rollback_step_request(step, ctx));

  rollback.from_ubatch_index = 0;
  ctx.ubatch_sizes[0] = 0;
  CHECK_FALSE(emel::kv::cache::guard::valid_rollback_step_request(step, ctx));

  ctx.ubatch_sizes[0] = 1;
  ctx.slot_offsets[0] = -1;
  CHECK_FALSE(emel::kv::cache::guard::valid_rollback_step_request(step, ctx));

  ctx.slot_offsets[0] = 4;
  CHECK_FALSE(emel::kv::cache::guard::valid_rollback_step_request(step, ctx));

  ctx.slot_offsets[0] = 0;
  ctx.ubatch_stream_ids[0] = 2;
  CHECK_FALSE(emel::kv::cache::guard::valid_rollback_step_request(step, ctx));

  ctx.ubatch_stream_ids[0] = 0;
  ctx.ubatch_seq_ids[0] = emel::kv::cache::action::MAX_SEQ;
  CHECK_FALSE(emel::kv::cache::guard::valid_rollback_step_request(step, ctx));
}

TEST_CASE("kv_cache_guard_seq_requests") {
  auto ctx_storage = make_context();
  context & ctx = *ctx_storage;
  ctx.n_stream = 1;
  ctx.seq_to_stream[0] = 0;

  emel::kv::cache::event::seq_remove remove{
    .seq_id = 0,
    .pos_start = 0,
    .pos_end = 1,
  };
  emel::kv::cache::event::validate_seq_remove remove_validate{.request = &remove};
  emel::kv::cache::event::seq_remove_step remove_step{.request = &remove};
  CHECK(emel::kv::cache::guard::valid_seq_remove_request(remove_validate, ctx));
  CHECK(emel::kv::cache::guard::valid_seq_remove_step_request(remove_step, ctx));
  CHECK_FALSE(emel::kv::cache::guard::invalid_seq_remove_request(remove_validate, ctx));
  CHECK_FALSE(emel::kv::cache::guard::invalid_seq_remove_step_request(remove_step, ctx));

  emel::kv::cache::event::validate_seq_remove null_remove{.request = nullptr};
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_remove_request(null_remove, ctx));

  remove.seq_id = emel::kv::cache::action::MAX_SEQ;
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_remove_request(remove_validate, ctx));

  remove.seq_id = 0;
  ctx.seq_to_stream[0] = 2;
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_remove_request(remove_validate, ctx));

  ctx.seq_to_stream[0] = 0;
  remove.pos_start = 2;
  remove.pos_end = 1;
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_remove_request(remove_validate, ctx));

  emel::kv::cache::event::seq_keep keep{.seq_id = 0};
  emel::kv::cache::event::validate_seq_keep keep_validate{.request = &keep};
  emel::kv::cache::event::seq_keep_step keep_step{.request = &keep};
  CHECK(emel::kv::cache::guard::valid_seq_keep_request(keep_validate, ctx));
  CHECK(emel::kv::cache::guard::valid_seq_keep_step_request(keep_step, ctx));

  keep.seq_id = emel::kv::cache::action::MAX_SEQ;
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_keep_request(keep_validate, ctx));

  keep.seq_id = 0;
  ctx.seq_to_stream[0] = 3;
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_keep_request(keep_validate, ctx));

  ctx.seq_to_stream[0] = 0;
  emel::kv::cache::event::seq_add add{
    .seq_id = 0,
    .pos_start = 0,
    .pos_end = 1,
  };
  emel::kv::cache::event::validate_seq_add add_validate{.request = &add};
  emel::kv::cache::event::seq_add_step add_step{.request = &add};
  CHECK(emel::kv::cache::guard::valid_seq_add_request(add_validate, ctx));
  CHECK(emel::kv::cache::guard::valid_seq_add_step_request(add_step, ctx));

  add.seq_id = emel::kv::cache::action::MAX_SEQ;
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_add_request(add_validate, ctx));

  add.seq_id = 0;
  ctx.seq_to_stream[0] = 4;
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_add_request(add_validate, ctx));

  ctx.seq_to_stream[0] = 0;
  add.pos_start = 2;
  add.pos_end = 1;
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_add_request(add_validate, ctx));

  emel::kv::cache::event::seq_div div{
    .seq_id = 0,
    .pos_start = 0,
    .pos_end = 1,
    .divisor = 2,
  };
  emel::kv::cache::event::validate_seq_div div_validate{.request = &div};
  emel::kv::cache::event::seq_div_step div_step{.request = &div};
  CHECK(emel::kv::cache::guard::valid_seq_div_request(div_validate, ctx));
  CHECK(emel::kv::cache::guard::valid_seq_div_step_request(div_step, ctx));

  div.seq_id = emel::kv::cache::action::MAX_SEQ;
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_div_request(div_validate, ctx));

  div.seq_id = 0;
  ctx.seq_to_stream[0] = 5;
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_div_request(div_validate, ctx));

  ctx.seq_to_stream[0] = 0;
  div.divisor = 0;
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_div_request(div_validate, ctx));

  div.divisor = 2;
  div.pos_start = 3;
  div.pos_end = 1;
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_div_request(div_validate, ctx));
}

TEST_CASE("kv_cache_guard_seq_copy_requests") {
  auto ctx_storage = make_context();
  context & ctx = *ctx_storage;
  ctx.kv_size = 2;
  ctx.n_stream = 2;
  ctx.seq_to_stream[0] = 0;
  ctx.seq_to_stream[1] = 1;

  emel::kv::cache::event::seq_copy copy{
    .seq_id_src = 0,
    .seq_id_dst = 1,
    .pos_start = 0,
    .pos_end = 1,
  };
  emel::kv::cache::event::validate_seq_copy copy_validate{.request = &copy};
  emel::kv::cache::event::seq_copy_step copy_step{.request = &copy};

  emel::kv::cache::action::set_cell_pos(ctx.streams[0], 0, 0);
  emel::kv::cache::action::add_seq_to_cell(ctx.streams[0], 0, 0);
  CHECK(emel::kv::cache::guard::valid_seq_copy_request(copy_validate, ctx));
  CHECK(emel::kv::cache::guard::valid_seq_copy_step_request(copy_step, ctx));

  emel::kv::cache::event::validate_seq_copy null_validate{.request = nullptr};
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_copy_request(null_validate, ctx));

  copy.seq_id_src = emel::kv::cache::action::MAX_SEQ;
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_copy_request(copy_validate, ctx));

  copy.seq_id_src = 0;
  ctx.seq_to_stream[1] = 3;
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_copy_request(copy_validate, ctx));

  ctx.seq_to_stream[1] = 1;
  copy.pos_start = 2;
  copy.pos_end = 1;
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_copy_request(copy_validate, ctx));

  copy.pos_start = 0;
  copy.pos_end = 1;
  ctx.pending_copy_count = emel::kv::cache::action::MAX_STREAM_COPY;
  ctx.pending_copy_src[0] = 0;
  ctx.pending_copy_dst[0] = 0;
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_copy_request(copy_validate, ctx));

  ctx.pending_copy_count = 0;
  emel::kv::cache::action::set_cell_pos(ctx.streams[1], 0, 1);
  CHECK_FALSE(emel::kv::cache::guard::valid_seq_copy_request(copy_validate, ctx));

  ctx.streams[1].pos[0] = emel::kv::cache::action::POS_NONE;
  copy.seq_id_dst = 0;
  CHECK(emel::kv::cache::guard::valid_seq_copy_request(copy_validate, ctx));
}

TEST_CASE("kv_cache_guard_updates_requests") {
  auto ctx_storage = make_context();
  context & ctx = *ctx_storage;

  emel::kv::cache::event::apply_updates updates{};
  emel::kv::cache::event::validate_updates validate{.request = &updates};
  emel::kv::cache::event::apply_updates_step step{.request = &updates};

  CHECK(emel::kv::cache::guard::valid_updates_request(validate, ctx));
  CHECK(emel::kv::cache::guard::valid_updates_step_request(step, ctx));
  CHECK_FALSE(emel::kv::cache::guard::invalid_updates_request(validate, ctx));
  CHECK_FALSE(emel::kv::cache::guard::invalid_updates_step_request(step, ctx));

  emel::kv::cache::event::validate_updates null_validate{.request = nullptr};
  emel::kv::cache::event::apply_updates_step null_step{.request = nullptr};
  CHECK_FALSE(emel::kv::cache::guard::valid_updates_request(null_validate, ctx));
  CHECK_FALSE(emel::kv::cache::guard::valid_updates_step_request(null_step, ctx));
}
