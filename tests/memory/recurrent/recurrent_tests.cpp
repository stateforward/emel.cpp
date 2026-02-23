#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/recurrent/actions.hpp"
#include "emel/memory/recurrent/guards.hpp"
#include "emel/memory/recurrent/sm.hpp"

TEST_CASE("recurrent_memory_allocate_branch_free_lifecycle") {
  emel::memory::recurrent::sm machine{};
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::memory::recurrent::event::reserve{
      .slot_capacity = 2,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(machine.slot_capacity() == 2);

  CHECK(machine.process_event(emel::memory::recurrent::event::allocate_sequence{
      .seq_id = 10,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(machine.has_sequence(10));

  CHECK(machine.process_event(emel::memory::recurrent::event::branch_sequence{
      .seq_id_src = 10,
      .seq_id_dst = 11,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(machine.has_sequence(11));
  CHECK(machine.active_count() == 2);

  CHECK_FALSE(
      machine.process_event(emel::memory::recurrent::event::allocate_sequence{
          .seq_id = 12,
          .error_out = &err,
      }));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK_FALSE(machine.has_sequence(12));

  CHECK(machine.process_event(emel::memory::recurrent::event::free_sequence{
      .seq_id = 11,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK_FALSE(machine.has_sequence(11));
  CHECK(machine.has_sequence(10));
  CHECK(machine.active_count() == 1);

  CHECK(machine.process_event(emel::memory::recurrent::event::allocate_sequence{
      .seq_id = 12,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(machine.has_sequence(12));
}

TEST_CASE("recurrent_memory_actions_cover_error_paths") {
  emel::memory::recurrent::action::context ctx{};
  int32_t err = EMEL_OK;

  CHECK(emel::memory::recurrent::action::slot_for_sequence(ctx, -1) ==
        emel::memory::recurrent::action::SLOT_NONE);
  CHECK(emel::memory::recurrent::action::slot_for_sequence(
            ctx, emel::memory::recurrent::action::MAX_SEQ) ==
        emel::memory::recurrent::action::SLOT_NONE);

  ctx.slot_capacity = 1;
  ctx.seq_to_slot[3] = 2;
  ctx.slot_active[2] = 1;
  CHECK_FALSE(emel::memory::recurrent::action::sequence_exists(ctx, 3));

  emel::memory::recurrent::action::set_invalid_argument(ctx);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  emel::memory::recurrent::action::run_reserve_step(ctx, nullptr);
  ctx.reserve_request.slot_capacity = 0;
  emel::memory::recurrent::action::run_reserve_step(ctx, &err);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  ctx.reserve_request.slot_capacity = 2;
  emel::memory::recurrent::action::run_reserve_step(ctx, &err);
  CHECK(err == EMEL_OK);

  emel::memory::recurrent::action::run_allocate_step(ctx, nullptr);
  ctx.allocate_request.seq_id = -1;
  emel::memory::recurrent::action::run_allocate_step(ctx, &err);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.allocate_request.seq_id = 0;
  ctx.slot_capacity = 0;
  emel::memory::recurrent::action::run_allocate_step(ctx, &err);
  CHECK(err == EMEL_ERR_BACKEND);

  ctx.slot_capacity = 1;
  ctx.seq_to_slot.fill(emel::memory::recurrent::action::SLOT_NONE);
  ctx.slot_active.fill(0);
  ctx.seq_to_slot[0] = 0;
  ctx.slot_active[0] = 1;
  ctx.allocate_request.seq_id = 0;
  emel::memory::recurrent::action::run_allocate_step(ctx, &err);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.allocate_request.seq_id = 1;
  emel::memory::recurrent::action::run_allocate_step(ctx, &err);
  CHECK(err == EMEL_ERR_BACKEND);

  emel::memory::recurrent::action::run_branch_step(ctx, nullptr);
  ctx.branch_request = emel::memory::recurrent::event::branch_sequence{
      .seq_id_src = -1,
      .seq_id_dst = 0,
  };
  emel::memory::recurrent::action::run_branch_step(ctx, &err);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.branch_request = emel::memory::recurrent::event::branch_sequence{
      .seq_id_src = 2,
      .seq_id_dst = 3,
  };
  emel::memory::recurrent::action::run_branch_step(ctx, &err);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.slot_capacity = 1;
  ctx.seq_to_slot.fill(emel::memory::recurrent::action::SLOT_NONE);
  ctx.slot_active.fill(0);
  ctx.seq_to_slot[2] = 0;
  ctx.slot_active[0] = 1;
  ctx.branch_request = emel::memory::recurrent::event::branch_sequence{
      .seq_id_src = 2,
      .seq_id_dst = 3,
  };
  emel::memory::recurrent::action::run_branch_step(ctx, &err);
  CHECK(err == EMEL_ERR_BACKEND);

  emel::memory::recurrent::action::run_free_step(ctx, nullptr);
  ctx.free_request =
      emel::memory::recurrent::event::free_sequence{.seq_id = -1};
  emel::memory::recurrent::action::run_free_step(ctx, &err);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.free_request = emel::memory::recurrent::event::free_sequence{.seq_id = 4};
  emel::memory::recurrent::action::run_free_step(ctx, &err);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.slot_capacity = 2;
  ctx.seq_to_slot.fill(emel::memory::recurrent::action::SLOT_NONE);
  ctx.slot_active.fill(0);
  ctx.seq_to_slot[4] = 5;
  ctx.slot_active[5] = 1;
  emel::memory::recurrent::action::run_free_step(ctx, &err);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.seq_to_slot[6] = 1;
  ctx.slot_active[1] = 1;
  ctx.active_count = 1;
  ctx.free_request = emel::memory::recurrent::event::free_sequence{.seq_id = 6};
  emel::memory::recurrent::action::run_free_step(ctx, &err);
  CHECK(err == EMEL_OK);
  CHECK(ctx.active_count == 0);

  ctx.last_error = EMEL_ERR_INVALID_ARGUMENT;
  ctx.phase_error = EMEL_OK;
  emel::memory::recurrent::action::ensure_last_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  ctx.last_error = EMEL_OK;
  ctx.phase_error = EMEL_OK;
  emel::memory::recurrent::action::ensure_last_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  ctx.last_error = EMEL_OK;
  ctx.phase_error = EMEL_ERR_INVALID_ARGUMENT;
  emel::memory::recurrent::action::ensure_last_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);

  emel::memory::recurrent::action::mark_done(ctx);
  CHECK(ctx.phase_error == EMEL_OK);
  CHECK(ctx.last_error == EMEL_OK);

  ctx.allocate_request =
      emel::memory::recurrent::event::allocate_sequence{.seq_id = 3};
  emel::memory::recurrent::action::clear_request(ctx);
  CHECK(ctx.allocate_request.seq_id == 0);

  int32_t unexpected_err = EMEL_OK;
  emel::memory::recurrent::action::on_unexpected(
      emel::memory::recurrent::event::allocate_sequence{
          .seq_id = 1,
          .error_out = &unexpected_err,
      },
      ctx);
  CHECK(unexpected_err == EMEL_ERR_BACKEND);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
}

TEST_CASE("recurrent_memory_guards_cover_all_predicates") {
  emel::memory::recurrent::action::context ctx{};

  CHECK(emel::memory::recurrent::guard::phase_ok{}(ctx));
  CHECK_FALSE(emel::memory::recurrent::guard::phase_failed{}(ctx));
  ctx.phase_error = EMEL_ERR_BACKEND;
  CHECK_FALSE(emel::memory::recurrent::guard::phase_ok{}(ctx));
  CHECK(emel::memory::recurrent::guard::phase_failed{}(ctx));

  ctx = {};
  CHECK_FALSE(emel::memory::recurrent::guard::has_capacity{}(ctx));
  CHECK(emel::memory::recurrent::guard::no_capacity{}(ctx));
  ctx.slot_capacity = 1;
  CHECK(emel::memory::recurrent::guard::has_capacity{}(ctx));
  CHECK_FALSE(emel::memory::recurrent::guard::no_capacity{}(ctx));

  ctx = {};
  ctx.reserve_request.slot_capacity = 0;
  CHECK_FALSE(emel::memory::recurrent::guard::valid_reserve_context{}(ctx));
  CHECK(emel::memory::recurrent::guard::invalid_reserve_context{}(ctx));
  ctx.reserve_request.slot_capacity = 2;
  CHECK(emel::memory::recurrent::guard::valid_reserve_context{}(ctx));

  ctx = {};
  ctx.allocate_request.seq_id = -1;
  CHECK_FALSE(emel::memory::recurrent::guard::valid_allocate_context{}(ctx));
  ctx.allocate_request.seq_id = 1;
  CHECK_FALSE(emel::memory::recurrent::guard::valid_allocate_context{}(ctx));
  ctx.slot_capacity = 2;
  CHECK(emel::memory::recurrent::guard::valid_allocate_context{}(ctx));
  ctx.seq_to_slot[1] = 0;
  ctx.slot_active[0] = 1;
  CHECK_FALSE(emel::memory::recurrent::guard::valid_allocate_context{}(ctx));
  CHECK(emel::memory::recurrent::guard::invalid_allocate_context{}(ctx));

  ctx = {};
  ctx.branch_request = emel::memory::recurrent::event::branch_sequence{
      .seq_id_src = -1,
      .seq_id_dst = 2,
  };
  CHECK_FALSE(emel::memory::recurrent::guard::valid_branch_context{}(ctx));
  ctx.branch_request = emel::memory::recurrent::event::branch_sequence{
      .seq_id_src = 1,
      .seq_id_dst = 1,
  };
  CHECK_FALSE(emel::memory::recurrent::guard::valid_branch_context{}(ctx));
  ctx.branch_request = emel::memory::recurrent::event::branch_sequence{
      .seq_id_src = 1,
      .seq_id_dst = 2,
  };
  CHECK_FALSE(emel::memory::recurrent::guard::valid_branch_context{}(ctx));
  ctx.slot_capacity = 2;
  CHECK_FALSE(emel::memory::recurrent::guard::valid_branch_context{}(ctx));
  ctx.seq_to_slot[1] = 0;
  ctx.slot_active[0] = 1;
  CHECK(emel::memory::recurrent::guard::valid_branch_context{}(ctx));
  ctx.seq_to_slot[2] = 1;
  ctx.slot_active[1] = 1;
  CHECK_FALSE(emel::memory::recurrent::guard::valid_branch_context{}(ctx));
  CHECK(emel::memory::recurrent::guard::invalid_branch_context{}(ctx));

  ctx = {};
  ctx.free_request.seq_id = -1;
  CHECK_FALSE(emel::memory::recurrent::guard::valid_free_context{}(ctx));
  ctx.free_request.seq_id = 4;
  CHECK_FALSE(emel::memory::recurrent::guard::valid_free_context{}(ctx));
  ctx.slot_capacity = 2;
  ctx.seq_to_slot[4] = 1;
  ctx.slot_active[1] = 1;
  CHECK(emel::memory::recurrent::guard::valid_free_context{}(ctx));
  CHECK_FALSE(emel::memory::recurrent::guard::invalid_free_context{}(ctx));
}

TEST_CASE("recurrent_memory_unexpected_event_reports_backend") {
  emel::memory::recurrent::sm machine{};
  int32_t err = EMEL_OK;

  CHECK_FALSE(
      machine.process_event(emel::memory::recurrent::event::allocate_sequence{
          .seq_id = 42,
          .error_out = &err,
      }));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(machine.last_error() == EMEL_ERR_BACKEND);
}
