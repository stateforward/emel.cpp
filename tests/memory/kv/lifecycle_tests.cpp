#include <cstdint>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/events.hpp"
#include "emel/memory/kv/sm.hpp"

namespace {

using kv_sm = emel::memory::kv::sm;
using namespace emel::memory::kv;

}  // namespace

TEST_CASE("memory_kv_lifecycle_reserve_success_and_failure") {
  kv_sm machine{};
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(event::reserve{
    .max_sequences = 8,
    .max_blocks = 16,
    .block_tokens = 4,
    .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  err = EMEL_OK;
  CHECK_FALSE(machine.process_event(event::reserve{
    .max_sequences = 999999,
    .max_blocks = 16,
    .block_tokens = 4,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("memory_kv_lifecycle_allocate_sequence_idempotent") {
  kv_sm machine{};
  int32_t err = EMEL_OK;

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 8,
    .max_blocks = 8,
    .block_tokens = 2,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 3,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 3,
    .error_out = &err,
  }));

  CHECK(machine.is_sequence_active(3));
  CHECK(machine.sequence_length(3) == 0);
}

TEST_CASE("memory_kv_lifecycle_block_oom") {
  kv_sm machine{};
  int32_t err = EMEL_OK;

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 4,
    .max_blocks = 2,
    .block_tokens = 1,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 0,
    .error_out = &err,
  }));

  CHECK_FALSE(machine.process_event(event::allocate_slots{
    .seq_id = 0,
    .token_count = 3,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_BACKEND);
}

TEST_CASE("memory_kv_lifecycle_branch_refcounts_and_free_pool") {
  kv_sm machine{};
  int32_t err = EMEL_OK;

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 8,
    .max_blocks = 8,
    .block_tokens = 2,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 1,
    .error_out = &err,
  }));

  int32_t allocated_blocks = 0;
  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 1,
    .token_count = 4,
    .block_count_out = &allocated_blocks,
    .error_out = &err,
  }));
  REQUIRE(allocated_blocks == 2);

  const int32_t parent_block_0 = machine.lookup_kv_block(1, 0);
  const int32_t parent_block_1 = machine.lookup_kv_block(1, 2);
  REQUIRE(parent_block_0 >= 0);
  REQUIRE(parent_block_1 >= 0);
  REQUIRE(parent_block_0 != parent_block_1);

  REQUIRE(machine.process_event(event::branch_sequence{
    .parent_seq_id = 1,
    .child_seq_id = 2,
    .error_out = &err,
  }));

  CHECK(machine.lookup_kv_block(2, 0) == parent_block_0);
  CHECK(machine.lookup_kv_block(2, 2) == parent_block_1);

  REQUIRE(machine.process_event(event::free_sequence{
    .seq_id = 2,
    .error_out = &err,
  }));
  CHECK(machine.is_sequence_active(1));

  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 3,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 3,
    .token_count = 2,
    .error_out = &err,
  }));
  const int32_t fresh_block = machine.lookup_kv_block(3, 0);
  CHECK(fresh_block != parent_block_0);
  CHECK(fresh_block != parent_block_1);

  REQUIRE(machine.process_event(event::free_sequence{
    .seq_id = 1,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 4,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 4,
    .token_count = 4,
    .error_out = &err,
  }));

  const int32_t recycled_block_0 = machine.lookup_kv_block(4, 0);
  const int32_t recycled_block_1 = machine.lookup_kv_block(4, 2);
  CHECK((recycled_block_0 == parent_block_0 || recycled_block_0 == parent_block_1));
  CHECK((recycled_block_1 == parent_block_0 || recycled_block_1 == parent_block_1));
  CHECK(recycled_block_0 != recycled_block_1);
}

TEST_CASE("memory_kv_lifecycle_mapping_order_is_deterministic") {
  kv_sm machine{};
  int32_t err = EMEL_OK;

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 8,
    .max_blocks = 6,
    .block_tokens = 1,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 0,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 1,
    .error_out = &err,
  }));

  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 0,
    .token_count = 1,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 1,
    .token_count = 1,
    .error_out = &err,
  }));

  CHECK(machine.lookup_kv_block(0, 0) == 5);
  CHECK(machine.lookup_kv_block(1, 0) == 4);
}

TEST_CASE("memory_kv_lifecycle_append_and_rollback_use_partial_tail_capacity") {
  kv_sm machine{};
  int32_t err = EMEL_OK;

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 2,
    .max_blocks = 4,
    .block_tokens = 16,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 0,
    .error_out = &err,
  }));

  int32_t block_delta = -1;
  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 0,
    .token_count = 1,
    .block_count_out = &block_delta,
    .error_out = &err,
  }));
  REQUIRE(block_delta == 1);

  const int32_t first_block = machine.lookup_kv_block(0, 0);
  REQUIRE(first_block >= 0);

  block_delta = -1;
  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 0,
    .token_count = 1,
    .block_count_out = &block_delta,
    .error_out = &err,
  }));
  CHECK(block_delta == 0);
  CHECK(machine.lookup_kv_block(0, 1) == first_block);
  CHECK(machine.sequence_length(0) == 2);

  block_delta = -1;
  REQUIRE(machine.process_event(event::rollback_slots{
    .seq_id = 0,
    .token_count = 1,
    .block_count_out = &block_delta,
    .error_out = &err,
  }));
  CHECK(block_delta == 0);
  CHECK(machine.lookup_kv_block(0, 0) == first_block);
  CHECK(machine.sequence_length(0) == 1);
}

TEST_CASE("memory_kv_lifecycle_validation_and_unexpected_event_paths") {
  kv_sm machine{};
  int32_t err = EMEL_OK;

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 4,
    .max_blocks = 4,
    .block_tokens = 2,
    .error_out = &err,
  }));

  CHECK_FALSE(machine.process_event(event::allocate_sequence{
    .seq_id = -1,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(machine.process_event(event::allocate_slots{
    .seq_id = 1,
    .token_count = 1,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(machine.process_event(event::branch_sequence{
    .parent_seq_id = 0,
    .child_seq_id = 0,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(machine.process_event(event::free_sequence{
    .seq_id = -1,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(machine.process_event(event::rollback_slots{
    .seq_id = 0,
    .token_count = 1,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  CHECK(machine.process_event(emel::memory::events::free_sequence_done{}));
  CHECK(machine.last_error() == EMEL_ERR_BACKEND);
}

TEST_CASE("memory_kv_action_direct_failure_and_helper_paths") {
  action::context ctx{};
  int32_t err = EMEL_OK;

  action::run_reserve_phase(event::reserve{
                              .max_sequences = 4,
                              .max_blocks = 4,
                              .block_tokens = 2,
                              .error_out = &err,
                            },
                            ctx);
  REQUIRE(err == EMEL_OK);

  CHECK(action::required_blocks(ctx, 0) == 0);
  CHECK(action::sequence_length_value(ctx, 0) == 0);
  CHECK(action::lookup_block_at_pos(ctx, 0, 0) == action::INVALID_INDEX);

  action::run_allocate_sequence_phase(event::allocate_sequence{
                                        .seq_id = 0,
                                        .error_out = &err,
                                      },
                                      ctx);
  REQUIRE(err == EMEL_OK);
  CHECK(action::lookup_block_at_pos(ctx, 0, 1) == action::INVALID_INDEX);
  ctx.block_tokens = 1;
  ctx.sequence_active[0] = true;
  ctx.sequence_length[0] = 1;
  ctx.sequence_block_count[0] = 0;
  CHECK(action::lookup_block_at_pos(ctx, 0, 0) == action::INVALID_INDEX);

  action::run_allocate_slots_phase(event::allocate_slots{
                                     .seq_id = 0,
                                     .token_count = 0,
                                     .error_out = &err,
                                   },
                                   ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.block_tokens = 0;
  action::run_allocate_slots_phase(event::allocate_slots{
                                     .seq_id = 0,
                                     .token_count = 1,
                                     .error_out = &err,
                                   },
                                   ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  ctx.block_tokens = 1;

  ctx.free_count = 1;
  ctx.free_stack[0] = static_cast<uint16_t>(action::MAX_BLOCKS + 9);
  action::run_allocate_slots_phase(event::allocate_slots{
                                     .seq_id = 0,
                                     .token_count = 1,
                                     .error_out = &err,
                                   },
                                   ctx);
  CHECK(err == EMEL_ERR_BACKEND);

  action::run_allocate_sequence_phase(event::allocate_sequence{
                                        .seq_id = 1,
                                        .error_out = &err,
                                      },
                                      ctx);
  REQUIRE(err == EMEL_OK);
  action::run_branch_sequence_phase(event::branch_sequence{
                                      .parent_seq_id = 0,
                                      .child_seq_id = 1,
                                      .error_out = &err,
                                    },
                                    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.sequence_active[0] = true;
  ctx.sequence_active[1] = false;
  ctx.sequence_block_count[0] = 1;
  ctx.seq_to_blocks[0][0] = static_cast<uint16_t>(action::MAX_BLOCKS + 11);
  action::run_branch_sequence_phase(event::branch_sequence{
                                      .parent_seq_id = 0,
                                      .child_seq_id = 1,
                                      .error_out = &err,
                                    },
                                    ctx);
  CHECK(err == EMEL_ERR_BACKEND);

  ctx.sequence_active[2] = false;
  action::run_free_sequence_phase(event::free_sequence{
                                    .seq_id = 2,
                                    .error_out = &err,
                                  },
                                  ctx);
  CHECK(err == EMEL_OK);

  ctx.sequence_active[0] = true;
  ctx.sequence_block_count[0] = 1;
  ctx.seq_to_blocks[0][0] = static_cast<uint16_t>(action::MAX_BLOCKS + 13);
  action::run_free_sequence_phase(event::free_sequence{
                                    .seq_id = 0,
                                    .error_out = &err,
                                  },
                                  ctx);
  CHECK(err == EMEL_ERR_BACKEND);

  ctx.sequence_active[0] = true;
  ctx.sequence_block_count[0] = 1;
  ctx.sequence_length[0] = 1;
  ctx.block_tokens = 1;
  ctx.seq_to_blocks[0][0] = static_cast<uint16_t>(action::MAX_BLOCKS + 15);
  action::run_rollback_slots_phase(event::rollback_slots{
                                     .seq_id = 0,
                                     .token_count = 1,
                                     .error_out = &err,
                                   },
                                   ctx);
  CHECK(err == EMEL_ERR_BACKEND);

  ctx.last_error = EMEL_OK;
  action::ensure_last_error(ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);

  action::on_unexpected(event::reserve{
                          .error_out = &err,
                        },
                        ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}
