#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/hybrid/actions.hpp"
#include "emel/memory/hybrid/guards.hpp"
#include "emel/memory/hybrid/sm.hpp"

TEST_CASE("hybrid_memory_allocate_rolls_back_kv_on_recurrent_failure") {
  emel::memory::hybrid::sm machine{};
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::memory::hybrid::event::reserve{
      .kv_size = 8,
      .recurrent_slot_capacity = 1,
      .n_stream = 1,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(machine.reserved());

  CHECK(machine.process_event(emel::memory::hybrid::event::allocate_sequence{
      .seq_id = 0,
      .slot_count = 2,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(machine.has_sequence(0));

  CHECK_FALSE(
      machine.process_event(emel::memory::hybrid::event::allocate_sequence{
          .seq_id = 1,
          .slot_count = 2,
          .error_out = &err,
      }));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK_FALSE(machine.has_sequence(1));
  CHECK_FALSE(machine.kv_memory().has_sequence(1));
  CHECK_FALSE(machine.recurrent_memory().has_sequence(1));
}

TEST_CASE("hybrid_memory_branch_rolls_back_kv_child_on_recurrent_failure") {
  emel::memory::hybrid::sm machine{};
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::memory::hybrid::event::reserve{
      .kv_size = 8,
      .recurrent_slot_capacity = 1,
      .n_stream = 1,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  CHECK(machine.process_event(emel::memory::hybrid::event::allocate_sequence{
      .seq_id = 0,
      .slot_count = 3,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  CHECK_FALSE(
      machine.process_event(emel::memory::hybrid::event::branch_sequence{
          .seq_id_src = 0,
          .seq_id_dst = 1,
          .error_out = &err,
      }));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK_FALSE(machine.has_sequence(1));
  CHECK_FALSE(machine.kv_memory().has_sequence(1));
  CHECK_FALSE(machine.recurrent_memory().has_sequence(1));
  CHECK(machine.has_sequence(0));
}

TEST_CASE("hybrid_memory_actions_cover_error_paths") {
  emel::memory::hybrid::action::context ctx{};
  int32_t err = EMEL_OK;

  CHECK(emel::memory::hybrid::action::normalize_child_error(true, EMEL_OK) ==
        EMEL_OK);
  CHECK(emel::memory::hybrid::action::normalize_child_error(false, EMEL_OK) ==
        EMEL_ERR_BACKEND);
  CHECK(emel::memory::hybrid::action::normalize_child_error(
            true, EMEL_ERR_INVALID_ARGUMENT) == EMEL_ERR_INVALID_ARGUMENT);

  emel::memory::hybrid::action::set_invalid_argument(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  emel::memory::hybrid::action::run_reserve_step(ctx, nullptr);
  ctx.reserve_request = emel::memory::hybrid::event::reserve{
      .kv_size = 8,
      .recurrent_slot_capacity = 0,
      .n_stream = 1,
  };
  emel::memory::hybrid::action::run_reserve_step(ctx, &err);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK_FALSE(ctx.reserved);

  ctx.reserve_request = emel::memory::hybrid::event::reserve{
      .kv_size = 0,
      .recurrent_slot_capacity = 1,
      .n_stream = 1,
  };
  emel::memory::hybrid::action::run_reserve_step(ctx, &err);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK_FALSE(ctx.reserved);

  ctx.reserve_request = emel::memory::hybrid::event::reserve{
      .kv_size = 8,
      .recurrent_slot_capacity = 2,
      .n_stream = 1,
  };
  emel::memory::hybrid::action::run_reserve_step(ctx, &err);
  CHECK(err == EMEL_OK);
  CHECK(ctx.reserved);

  emel::memory::hybrid::action::run_allocate_step(ctx, nullptr);
  ctx.allocate_request = emel::memory::hybrid::event::allocate_sequence{
      .seq_id = -1,
      .slot_count = 2,
  };
  emel::memory::hybrid::action::run_allocate_step(ctx, &err);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.allocate_request = emel::memory::hybrid::event::allocate_sequence{
      .seq_id = 0,
      .slot_count = 2,
  };
  emel::memory::hybrid::action::run_allocate_step(ctx, &err);
  CHECK(err == EMEL_OK);

  ctx.allocate_request = emel::memory::hybrid::event::allocate_sequence{
      .seq_id = 1,
      .slot_count = 2,
  };
  emel::memory::hybrid::action::run_allocate_step(ctx, &err);
  CHECK(err == EMEL_OK);

  ctx.allocate_request = emel::memory::hybrid::event::allocate_sequence{
      .seq_id = 2,
      .slot_count = 2,
  };
  emel::memory::hybrid::action::run_allocate_step(ctx, &err);
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK_FALSE(ctx.kv_memory.has_sequence(2));

  emel::memory::hybrid::action::run_branch_step(ctx, nullptr);
  ctx.branch_request = emel::memory::hybrid::event::branch_sequence{
      .seq_id_src = 99,
      .seq_id_dst = 100,
  };
  emel::memory::hybrid::action::run_branch_step(ctx, &err);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  emel::memory::hybrid::action::context rollback_ctx{};
  rollback_ctx.reserve_request = emel::memory::hybrid::event::reserve{
      .kv_size = 8,
      .recurrent_slot_capacity = 1,
      .n_stream = 1,
  };
  emel::memory::hybrid::action::run_reserve_step(rollback_ctx, &err);
  CHECK(err == EMEL_OK);
  rollback_ctx.allocate_request =
      emel::memory::hybrid::event::allocate_sequence{
          .seq_id = 0,
          .slot_count = 2,
      };
  emel::memory::hybrid::action::run_allocate_step(rollback_ctx, &err);
  CHECK(err == EMEL_OK);
  rollback_ctx.branch_request = emel::memory::hybrid::event::branch_sequence{
      .seq_id_src = 0,
      .seq_id_dst = 1,
  };
  emel::memory::hybrid::action::run_branch_step(rollback_ctx, &err);
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK_FALSE(rollback_ctx.kv_memory.has_sequence(1));

  emel::memory::hybrid::action::run_free_step(rollback_ctx, nullptr);
  rollback_ctx.free_request =
      emel::memory::hybrid::event::free_sequence{.seq_id = 0};
  emel::memory::hybrid::action::run_free_step(rollback_ctx, &err);
  CHECK(err == EMEL_OK);
  CHECK_FALSE(rollback_ctx.kv_memory.has_sequence(0));
  CHECK_FALSE(rollback_ctx.recurrent_memory.has_sequence(0));

  rollback_ctx.free_request =
      emel::memory::hybrid::event::free_sequence{.seq_id = 77};
  emel::memory::hybrid::action::run_free_step(rollback_ctx, &err);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  rollback_ctx.reserve_request = emel::memory::hybrid::event::reserve{
      .kv_size = 8,
      .recurrent_slot_capacity = 2,
      .n_stream = 1,
  };
  emel::memory::hybrid::action::run_reserve_step(rollback_ctx, &err);
  CHECK(err == EMEL_OK);
  int32_t kv_err = EMEL_OK;
  CHECK(rollback_ctx.kv_memory.process_event(
      emel::memory::kv::event::allocate_sequence{
          .seq_id = 9,
          .slot_count = 2,
          .error_out = &kv_err,
      }));
  CHECK(kv_err == EMEL_OK);
  rollback_ctx.free_request =
      emel::memory::hybrid::event::free_sequence{.seq_id = 9};
  emel::memory::hybrid::action::run_free_step(rollback_ctx, &err);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  rollback_ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
  rollback_ctx.phase_error = EMEL_OK;
  emel::memory::hybrid::action::ensure_last_error(rollback_ctx);
  CHECK(rollback_ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
  rollback_ctx.last_error = EMEL_OK;
  rollback_ctx.phase_error = EMEL_OK;
  emel::memory::hybrid::action::ensure_last_error(rollback_ctx);
  CHECK(rollback_ctx.last_error == EMEL_ERR_BACKEND);
  emel::memory::hybrid::action::mark_done(rollback_ctx);
  CHECK(rollback_ctx.last_error == EMEL_OK);
  CHECK(rollback_ctx.phase_error == EMEL_OK);

  rollback_ctx.allocate_request =
      emel::memory::hybrid::event::allocate_sequence{.seq_id = 11};
  emel::memory::hybrid::action::clear_request(rollback_ctx);
  CHECK(rollback_ctx.allocate_request.seq_id == 0);

  int32_t unexpected_err = EMEL_OK;
  emel::memory::hybrid::action::on_unexpected(
      emel::memory::hybrid::event::reserve{
          .error_out = &unexpected_err,
      },
      rollback_ctx);
  CHECK(unexpected_err == EMEL_ERR_BACKEND);
  CHECK(rollback_ctx.last_error == EMEL_ERR_BACKEND);
}

TEST_CASE("hybrid_memory_guards_cover_all_predicates") {
  emel::memory::hybrid::action::context ctx{};

  CHECK(emel::memory::hybrid::guard::phase_ok{}(ctx));
  CHECK_FALSE(emel::memory::hybrid::guard::phase_failed{}(ctx));
  ctx.phase_error = EMEL_ERR_BACKEND;
  CHECK_FALSE(emel::memory::hybrid::guard::phase_ok{}(ctx));
  CHECK(emel::memory::hybrid::guard::phase_failed{}(ctx));

  emel::memory::hybrid::action::context capacity_ctx{};
  CHECK_FALSE(emel::memory::hybrid::guard::has_capacity{}(capacity_ctx));
  CHECK(emel::memory::hybrid::guard::no_capacity{}(capacity_ctx));
  capacity_ctx.reserved = true;
  CHECK(emel::memory::hybrid::guard::has_capacity{}(capacity_ctx));
  CHECK_FALSE(emel::memory::hybrid::guard::no_capacity{}(capacity_ctx));

  emel::memory::hybrid::action::context reserve_ctx{};
  reserve_ctx.reserve_request = emel::memory::hybrid::event::reserve{
      .kv_size = 0,
      .recurrent_slot_capacity = 1,
      .n_stream = 1,
  };
  CHECK_FALSE(
      emel::memory::hybrid::guard::valid_reserve_context{}(reserve_ctx));
  reserve_ctx.reserve_request.kv_size = 8;
  reserve_ctx.reserve_request.recurrent_slot_capacity = 0;
  CHECK_FALSE(
      emel::memory::hybrid::guard::valid_reserve_context{}(reserve_ctx));
  reserve_ctx.reserve_request.recurrent_slot_capacity = 1;
  reserve_ctx.reserve_request.n_stream = 0;
  CHECK_FALSE(
      emel::memory::hybrid::guard::valid_reserve_context{}(reserve_ctx));
  reserve_ctx.reserve_request.n_stream = 1;
  CHECK(emel::memory::hybrid::guard::valid_reserve_context{}(reserve_ctx));
  CHECK_FALSE(
      emel::memory::hybrid::guard::invalid_reserve_context{}(reserve_ctx));

  emel::memory::hybrid::action::context lifecycle_ctx{};
  lifecycle_ctx.allocate_request =
      emel::memory::hybrid::event::allocate_sequence{
          .seq_id = -1,
          .slot_count = 1,
      };
  CHECK_FALSE(
      emel::memory::hybrid::guard::valid_allocate_context{}(lifecycle_ctx));
  lifecycle_ctx.allocate_request.seq_id = 0;
  lifecycle_ctx.allocate_request.slot_count = 0;
  CHECK_FALSE(
      emel::memory::hybrid::guard::valid_allocate_context{}(lifecycle_ctx));
  lifecycle_ctx.allocate_request.slot_count = 2;
  CHECK(emel::memory::hybrid::guard::valid_allocate_context{}(lifecycle_ctx));
  lifecycle_ctx.reserve_request = emel::memory::hybrid::event::reserve{
      .kv_size = 8,
      .recurrent_slot_capacity = 2,
      .n_stream = 1,
  };
  int32_t err = EMEL_OK;
  emel::memory::hybrid::action::run_reserve_step(lifecycle_ctx, &err);
  lifecycle_ctx.allocate_request =
      emel::memory::hybrid::event::allocate_sequence{
          .seq_id = 3,
          .slot_count = 2,
      };
  emel::memory::hybrid::action::run_allocate_step(lifecycle_ctx, &err);
  CHECK_FALSE(
      emel::memory::hybrid::guard::valid_allocate_context{}(lifecycle_ctx));
  CHECK(emel::memory::hybrid::guard::invalid_allocate_context{}(lifecycle_ctx));

  lifecycle_ctx.branch_request = emel::memory::hybrid::event::branch_sequence{
      .seq_id_src = -1,
      .seq_id_dst = 4,
  };
  CHECK_FALSE(
      emel::memory::hybrid::guard::valid_branch_context{}(lifecycle_ctx));
  lifecycle_ctx.branch_request = emel::memory::hybrid::event::branch_sequence{
      .seq_id_src = 4,
      .seq_id_dst = 4,
  };
  CHECK_FALSE(
      emel::memory::hybrid::guard::valid_branch_context{}(lifecycle_ctx));
  lifecycle_ctx.branch_request = emel::memory::hybrid::event::branch_sequence{
      .seq_id_src = 100,
      .seq_id_dst = 101,
  };
  CHECK_FALSE(
      emel::memory::hybrid::guard::valid_branch_context{}(lifecycle_ctx));
  lifecycle_ctx.branch_request = emel::memory::hybrid::event::branch_sequence{
      .seq_id_src = 3,
      .seq_id_dst = 5,
  };
  CHECK(emel::memory::hybrid::guard::valid_branch_context{}(lifecycle_ctx));
  emel::memory::hybrid::action::run_branch_step(lifecycle_ctx, &err);
  CHECK_FALSE(
      emel::memory::hybrid::guard::valid_branch_context{}(lifecycle_ctx));
  CHECK(emel::memory::hybrid::guard::invalid_branch_context{}(lifecycle_ctx));

  lifecycle_ctx.free_request =
      emel::memory::hybrid::event::free_sequence{.seq_id = -1};
  CHECK_FALSE(emel::memory::hybrid::guard::valid_free_context{}(lifecycle_ctx));
  lifecycle_ctx.free_request =
      emel::memory::hybrid::event::free_sequence{.seq_id = 42};
  CHECK_FALSE(emel::memory::hybrid::guard::valid_free_context{}(lifecycle_ctx));
  lifecycle_ctx.free_request =
      emel::memory::hybrid::event::free_sequence{.seq_id = 3};
  CHECK(emel::memory::hybrid::guard::valid_free_context{}(lifecycle_ctx));
  CHECK_FALSE(
      emel::memory::hybrid::guard::invalid_free_context{}(lifecycle_ctx));
}

TEST_CASE("hybrid_memory_unexpected_event_reports_backend") {
  emel::memory::hybrid::sm machine{};
  int32_t err = EMEL_OK;

  CHECK_FALSE(
      machine.process_event(emel::memory::hybrid::event::allocate_sequence{
          .seq_id = 1,
          .slot_count = 1,
          .error_out = &err,
      }));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(machine.last_error() == EMEL_ERR_BACKEND);
}
