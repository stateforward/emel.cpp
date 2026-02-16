#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/kv/cache/actions.hpp"
#include "emel/kv/cache/sm.hpp"

TEST_CASE("kv_cache_starts_initialized") {
  emel::kv::cache::sm machine{};
}

TEST_CASE("kv_cache_prepare_plans_slots_and_apply_reports_progressive_n_kv") {
  emel::kv::cache::sm machine{};
  std::array<int32_t, 3> ubatch_sizes = {{2, 2, 1}};
  std::array<int32_t, 8> slot_offsets = {{0, 0, 0, 0, 0, 0, 0, 0}};
  int32_t ubatch_count = 0;
  int32_t kv_tokens = 0;

  CHECK(machine.process_event(emel::kv::cache::event::prepare{
    .ubatch_sizes = ubatch_sizes.data(),
    .ubatch_count = static_cast<int32_t>(ubatch_sizes.size()),
    .requested_capacity = 16,
    .slot_offsets_out = slot_offsets.data(),
    .slot_offsets_capacity = static_cast<int32_t>(slot_offsets.size()),
    .ubatch_count_out = &ubatch_count,
  }));

  CHECK(ubatch_count == 3);
  CHECK(slot_offsets[0] == 0);
  CHECK(slot_offsets[1] == 2);
  CHECK(slot_offsets[2] == 4);

  CHECK(machine.process_event(emel::kv::cache::event::apply_ubatch{
    .ubatch_index = 0,
    .kv_tokens_out = &kv_tokens,
  }));
  CHECK(kv_tokens == 2);

  CHECK(machine.process_event(emel::kv::cache::event::apply_ubatch{
    .ubatch_index = 1,
    .kv_tokens_out = &kv_tokens,
  }));
  CHECK(kv_tokens == 4);

  CHECK(machine.process_event(emel::kv::cache::event::apply_ubatch{
    .ubatch_index = 2,
    .kv_tokens_out = &kv_tokens,
  }));
  CHECK(kv_tokens == 5);
}

TEST_CASE("kv_cache_prepare_fails_when_no_contiguous_slot_fits") {
  emel::kv::cache::sm machine{};
  std::array<int32_t, 2> ubatch_sizes = {{3, 3}};

  CHECK(machine.process_event(emel::kv::cache::event::prepare{
    .ubatch_sizes = ubatch_sizes.data(),
    .ubatch_count = static_cast<int32_t>(ubatch_sizes.size()),
    .requested_capacity = 4,
  }));
}

TEST_CASE("kv_cache_apply_requires_sequential_ubatch_order") {
  emel::kv::cache::sm machine{};
  std::array<int32_t, 3> ubatch_sizes = {{2, 2, 1}};
  int32_t kv_tokens = 0;

  CHECK(machine.process_event(emel::kv::cache::event::prepare{
    .ubatch_sizes = ubatch_sizes.data(),
    .ubatch_count = static_cast<int32_t>(ubatch_sizes.size()),
    .requested_capacity = 16,
  }));

  CHECK(machine.process_event(emel::kv::cache::event::apply_ubatch{
    .ubatch_index = 1,
    .kv_tokens_out = &kv_tokens,
  }));
  CHECK(kv_tokens == 0);
}

TEST_CASE("kv_cache_rollback_clears_planned_ranges_from_index") {
  emel::kv::cache::sm machine{};
  std::array<int32_t, 3> ubatch_sizes = {{2, 2, 1}};
  int32_t kv_tokens = 0;

  CHECK(machine.process_event(emel::kv::cache::event::prepare{
    .ubatch_sizes = ubatch_sizes.data(),
    .ubatch_count = static_cast<int32_t>(ubatch_sizes.size()),
    .requested_capacity = 16,
  }));

  CHECK(machine.process_event(emel::kv::cache::event::apply_ubatch{
    .ubatch_index = 0,
    .kv_tokens_out = &kv_tokens,
  }));
  CHECK(kv_tokens == 2);

  CHECK(machine.process_event(emel::kv::cache::event::apply_ubatch{
    .ubatch_index = 1,
    .kv_tokens_out = &kv_tokens,
  }));
  CHECK(kv_tokens == 4);

  CHECK(machine.process_event(emel::kv::cache::event::rollback{
    .from_ubatch_index = 0,
  }));

  CHECK(machine.process_event(emel::kv::cache::event::apply_ubatch{
    .ubatch_index = 0,
    .kv_tokens_out = &kv_tokens,
  }));
  CHECK(kv_tokens == 2);
}

TEST_CASE("kv_cache_reports_validation_errors") {
  emel::kv::cache::sm machine{};
  std::array<int32_t, 2> ubatch_sizes = {{1, 1}};

  CHECK(machine.process_event(emel::kv::cache::event::prepare{
    .ubatch_sizes = ubatch_sizes.data(),
    .ubatch_count = 0,
    .requested_capacity = 16,
  }));

  CHECK(machine.process_event(emel::kv::cache::event::prepare{
    .ubatch_sizes = ubatch_sizes.data(),
    .ubatch_count = static_cast<int32_t>(ubatch_sizes.size()),
    .requested_capacity = 16,
  }));

  CHECK(machine.process_event(emel::kv::cache::event::apply_ubatch{
    .ubatch_index = -1,
  }));

  CHECK(machine.process_event(emel::kv::cache::event::rollback{
    .from_ubatch_index = 3,
  }));
}

TEST_CASE("kv_cache_action_helpers_cover_validate_branches") {
  using emel::kv::cache::action::MAX_KV_CELLS;
  using emel::kv::cache::action::context;
  using emel::kv::cache::action::operation;

  context ctx{};
  int32_t error_out = EMEL_OK;

  ctx.op = operation::none;
  emel::kv::cache::action::run_validate(
      emel::kv::cache::event::validate{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  ctx.op = operation::prepare;
  ctx.ubatch_count = 1;
  ctx.requested_capacity = MAX_KV_CELLS + 1;
  ctx.kv_size = 8;
  emel::kv::cache::action::run_validate(
      emel::kv::cache::event::validate{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  ctx.requested_capacity = 8;
  ctx.kv_size = 0;
  emel::kv::cache::action::run_validate(
      emel::kv::cache::event::validate{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  ctx.op = operation::apply;
  ctx.planned_ubatch_count = 0;
  ctx.current_ubatch_index = 0;
  ctx.applied_ubatches = 0;
  emel::kv::cache::action::run_validate(
      emel::kv::cache::event::validate{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  ctx.planned_ubatch_count = 2;
  ctx.current_ubatch_index = 2;
  emel::kv::cache::action::run_validate(
      emel::kv::cache::event::validate{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  ctx.current_ubatch_index = 1;
  ctx.applied_ubatches = 0;
  emel::kv::cache::action::run_validate(
      emel::kv::cache::event::validate{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  ctx.current_ubatch_index = 0;
  ctx.applied_ubatches = 0;
  emel::kv::cache::action::run_validate(
      emel::kv::cache::event::validate{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_OK);

  ctx.op = operation::rollback;
  ctx.current_ubatch_index = 3;
  ctx.applied_ubatches = 2;
  ctx.planned_ubatch_count = 2;
  emel::kv::cache::action::run_validate(
      emel::kv::cache::event::validate{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("kv_cache_action_helpers_cover_slot_planning_apply_rollback_and_publish_edges") {
  using emel::kv::cache::action::context;
  using emel::kv::cache::action::operation;

  context ctx{};
  int32_t error_out = EMEL_OK;

  ctx.kv_size = 4;
  ctx.head = 4;
  emel::kv::cache::action::begin_prepare(
      emel::kv::cache::event::prepare{
          .ubatch_sizes = nullptr,
          .ubatch_count = 1,
          .requested_capacity = 4,
      },
      ctx);
  CHECK(ctx.head == 0);

  ctx.op = operation::prepare;
  ctx.ubatch_count = 1;
  ctx.ubatch_sizes[0] = 0;
  emel::kv::cache::action::run_prepare_slots(
      emel::kv::cache::event::prepare_slots{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  ctx.ubatch_sizes[0] = 5;
  emel::kv::cache::action::run_prepare_slots(
      emel::kv::cache::event::prepare_slots{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_BACKEND);

  ctx.kv_size = 8;
  ctx.head = 7;
  ctx.cells.fill(0);
  ctx.cells[1] = 1;
  ctx.ubatch_sizes[0] = 2;
  emel::kv::cache::action::run_prepare_slots(
      emel::kv::cache::event::prepare_slots{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_OK);
  CHECK(ctx.slot_offsets[0] == 2);

  ctx.op = operation::apply;
  ctx.current_ubatch_index = 0;
  ctx.planned_ubatch_count = 1;
  ctx.applied_ubatches = 0;
  ctx.ubatch_sizes[0] = 0;
  ctx.slot_offsets[0] = 0;
  ctx.op = operation::prepare;
  emel::kv::cache::action::run_apply_step(
      emel::kv::cache::event::apply_step{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  ctx.op = operation::apply;
  ctx.current_ubatch_index = 1;
  emel::kv::cache::action::run_apply_step(
      emel::kv::cache::event::apply_step{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  ctx.current_ubatch_index = 0;
  emel::kv::cache::action::run_apply_step(
      emel::kv::cache::event::apply_step{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_BACKEND);

  ctx.ubatch_sizes[0] = 1;
  ctx.slot_offsets[0] = 8;
  emel::kv::cache::action::run_apply_step(
      emel::kv::cache::event::apply_step{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_BACKEND);

  ctx.slot_offsets[0] = 7;
  ctx.cells[7] = 9;
  emel::kv::cache::action::run_apply_step(
      emel::kv::cache::event::apply_step{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_BACKEND);

  ctx.cells[7] = 0;
  ctx.kv_tokens = 0;
  emel::kv::cache::action::run_apply_step(
      emel::kv::cache::event::apply_step{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_OK);
  CHECK(ctx.head == 0);
  CHECK(ctx.kv_tokens >= 1);

  ctx.op = operation::rollback;
  ctx.current_ubatch_index = 0;
  ctx.applied_ubatches = 1;
  ctx.head = 8;
  emel::kv::cache::action::run_rollback_step(
      emel::kv::cache::event::rollback_step{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_OK);
  CHECK(ctx.applied_ubatches == 0);
  CHECK(ctx.head == 7);

  ctx.op = operation::apply;
  emel::kv::cache::action::run_rollback_step(
      emel::kv::cache::event::rollback_step{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_INVALID_ARGUMENT);

  ctx.op = operation::rollback;
  ctx.current_ubatch_index = 0;
  ctx.applied_ubatches = 1;
  ctx.ubatch_sizes[0] = 2;
  ctx.slot_offsets[0] = 7;
  emel::kv::cache::action::run_rollback_step(
      emel::kv::cache::event::rollback_step{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_ERR_BACKEND);

  ctx.op = operation::prepare;
  ctx.planned_ubatch_count = 2;
  ctx.slot_offsets[0] = 1;
  ctx.slot_offsets[1] = 3;
  emel::kv::cache::action::run_publish(
      emel::kv::cache::event::publish{
          .error_out = &error_out,
      },
      ctx);
  CHECK(error_out == EMEL_OK);

  emel::kv::cache::action::on_kv_done(emel::kv::cache::events::kv_done{}, ctx);
  CHECK(ctx.op == operation::none);
  ctx.current_ubatch_index = 42;
  emel::kv::cache::action::on_kv_error(
      emel::kv::cache::events::kv_error{
          .err = EMEL_ERR_BACKEND,
      },
      ctx);
  CHECK(ctx.op == operation::none);
  CHECK(ctx.current_ubatch_index == 0);
}
