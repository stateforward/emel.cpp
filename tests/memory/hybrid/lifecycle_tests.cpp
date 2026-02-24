#include <cstdint>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/events.hpp"
#include "emel/memory/hybrid/sm.hpp"

namespace {

using hybrid_sm = emel::memory::hybrid::sm;
using namespace emel::memory::hybrid;

struct copy_probe {
  bool succeed = true;
  int32_t callback_error = EMEL_OK;
};

bool copy_state_cb(const int32_t, const int32_t, void * user_data, int32_t * error_out) {
  const auto * probe = static_cast<const copy_probe *>(user_data);
  if (error_out != nullptr) {
    *error_out = probe != nullptr ? probe->callback_error : EMEL_ERR_BACKEND;
  }
  return probe != nullptr && probe->succeed;
}

}  // namespace

TEST_CASE("memory_hybrid_lifecycle_allocate_rolls_back_on_recurrent_failure") {
  hybrid_sm machine{};
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
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 1,
    .error_out = &err,
  }));

  CHECK_FALSE(machine.process_event(event::allocate_sequence{
    .seq_id = 2,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK_FALSE(machine.is_sequence_active(2));
}

TEST_CASE("memory_hybrid_lifecycle_branch_rolls_back_kv_when_recurrent_fails") {
  hybrid_sm machine{};
  int32_t err = EMEL_OK;

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 8,
    .max_blocks = 8,
    .block_tokens = 2,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 0,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 0,
    .token_count = 4,
    .error_out = &err,
  }));

  CHECK_FALSE(machine.process_event(event::branch_sequence{
    .parent_seq_id = 0,
    .child_seq_id = 1,
    .copy_state = nullptr,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK_FALSE(machine.is_sequence_active(1));
  CHECK(machine.lookup_kv_block(1, 0) == -1);
}

TEST_CASE("memory_hybrid_lifecycle_free_consistent_across_kv_and_recurrent") {
  hybrid_sm machine{};
  int32_t err = EMEL_OK;
  copy_probe probe{};

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 8,
    .max_blocks = 8,
    .block_tokens = 2,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 0,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 0,
    .token_count = 4,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::branch_sequence{
    .parent_seq_id = 0,
    .child_seq_id = 1,
    .copy_state = &copy_state_cb,
    .copy_state_user_data = &probe,
    .error_out = &err,
  }));

  REQUIRE(machine.process_event(event::free_sequence{
    .seq_id = 1,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::free_sequence{
    .seq_id = 0,
    .error_out = &err,
  }));

  CHECK_FALSE(machine.is_sequence_active(0));
  CHECK_FALSE(machine.is_sequence_active(1));
  CHECK(machine.lookup_kv_block(0, 0) == -1);
  CHECK(machine.lookup_kv_block(1, 0) == -1);
  CHECK(machine.lookup_recurrent_slot(0) == -1);
  CHECK(machine.lookup_recurrent_slot(1) == -1);
}

TEST_CASE("memory_hybrid_lifecycle_validation_and_unexpected_event_paths") {
  hybrid_sm machine{};
  int32_t err = EMEL_OK;

  CHECK_FALSE(machine.process_event(event::reserve{
    .max_sequences = 999999,
    .max_blocks = 8,
    .block_tokens = 2,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 4,
    .max_blocks = 2,
    .block_tokens = 2,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 0,
    .error_out = &err,
  }));

  CHECK_FALSE(machine.process_event(event::branch_sequence{
    .parent_seq_id = 0,
    .child_seq_id = 0,
    .copy_state = &copy_state_cb,
    .copy_state_user_data = nullptr,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(machine.process_event(event::rollback_slots{
    .seq_id = -1,
    .token_count = 1,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  CHECK(machine.process_event(emel::memory::events::rollback_slots_done{}));
  CHECK(machine.last_error() == EMEL_ERR_BACKEND);
}

TEST_CASE("memory_hybrid_action_direct_failure_and_helper_paths") {
  CHECK(action::normalize_error(true, EMEL_OK) == EMEL_OK);
  CHECK(action::normalize_error(false, EMEL_OK) == EMEL_ERR_BACKEND);

  int32_t err = EMEL_OK;

  action::context seq_ctx{};
  action::run_reserve_phase(event::reserve{
                              .max_sequences = 4,
                              .max_blocks = 4,
                              .block_tokens = 1,
                              .error_out = &err,
                            },
                            seq_ctx);
  REQUIRE(err == EMEL_OK);
  action::run_allocate_sequence_phase(event::allocate_sequence{
                                        .seq_id = -1,
                                        .error_out = &err,
                                      },
                                      seq_ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  action::run_allocate_sequence_phase(event::allocate_sequence{
                                        .seq_id = 0,
                                        .error_out = &err,
                                      },
                                      seq_ctx);
  REQUIRE(err == EMEL_OK);

  int32_t recurrent_side_err = EMEL_OK;
  REQUIRE(seq_ctx.recurrent.process_event(emel::memory::recurrent::event::free_sequence{
    .seq_id = 0,
    .error_out = &recurrent_side_err,
  }));
  action::run_allocate_slots_phase(event::allocate_slots{
                                     .seq_id = 0,
    .token_count = 1,
                                     .error_out = &err,
                                   },
                                   seq_ctx);
  CHECK(err != EMEL_OK);

  action::context success_alloc_ctx{};
  action::run_reserve_phase(event::reserve{
                              .max_sequences = 4,
                              .max_blocks = 8,
                              .block_tokens = 1,
                              .error_out = &err,
                            },
                            success_alloc_ctx);
  REQUIRE(err == EMEL_OK);
  action::run_allocate_sequence_phase(event::allocate_sequence{
                                        .seq_id = 0,
                                        .error_out = &err,
                                      },
                                      success_alloc_ctx);
  REQUIRE(err == EMEL_OK);
  int32_t allocated_blocks = -1;
  action::run_allocate_slots_phase(event::allocate_slots{
                                     .seq_id = 0,
                                     .token_count = 2,
                                     .block_count_out = &allocated_blocks,
                                     .error_out = &err,
                                   },
                                   success_alloc_ctx);
  CHECK(err == EMEL_OK);
  CHECK(allocated_blocks == 2);

  action::run_free_sequence_phase(event::free_sequence{
                                    .seq_id = -1,
                                    .error_out = &err,
                                  },
                                  success_alloc_ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  action::context rollback_ctx{};
  action::run_reserve_phase(event::reserve{
                              .max_sequences = 4,
                              .max_blocks = 8,
                              .block_tokens = 1,
                              .error_out = &err,
                            },
                            rollback_ctx);
  REQUIRE(err == EMEL_OK);
  action::run_allocate_sequence_phase(event::allocate_sequence{
                                        .seq_id = 0,
                                        .error_out = &err,
                                      },
                                      rollback_ctx);
  REQUIRE(err == EMEL_OK);
  action::run_allocate_slots_phase(event::allocate_slots{
                                     .seq_id = 0,
                                     .token_count = 2,
                                     .error_out = &err,
                                   },
                                   rollback_ctx);
  REQUIRE(err == EMEL_OK);
  REQUIRE(rollback_ctx.recurrent.process_event(emel::memory::recurrent::event::free_sequence{
    .seq_id = 0,
    .error_out = &recurrent_side_err,
  }));
  action::run_rollback_slots_phase(event::rollback_slots{
                                     .seq_id = 0,
                                     .token_count = 1,
                                     .error_out = &err,
                                   },
                                   rollback_ctx);
  CHECK(err != EMEL_OK);

  action::context rollback_success_ctx{};
  action::run_reserve_phase(event::reserve{
                              .max_sequences = 4,
                              .max_blocks = 8,
                              .block_tokens = 1,
                              .error_out = &err,
                            },
                            rollback_success_ctx);
  REQUIRE(err == EMEL_OK);
  action::run_allocate_sequence_phase(event::allocate_sequence{
                                        .seq_id = 0,
                                        .error_out = &err,
                                      },
                                      rollback_success_ctx);
  REQUIRE(err == EMEL_OK);
  action::run_allocate_slots_phase(event::allocate_slots{
                                     .seq_id = 0,
                                     .token_count = 2,
                                     .error_out = &err,
                                   },
                                   rollback_success_ctx);
  REQUIRE(err == EMEL_OK);

  int32_t rolled_back_blocks = -1;
  action::run_rollback_slots_phase(event::rollback_slots{
                                     .seq_id = 0,
                                     .token_count = 1,
                                     .block_count_out = &rolled_back_blocks,
                                     .error_out = &err,
                                   },
                                   rollback_success_ctx);
  CHECK(err == EMEL_OK);
  CHECK(rolled_back_blocks == 1);

  rollback_success_ctx.last_error = EMEL_OK;
  action::ensure_last_error(rollback_success_ctx);
  CHECK(rollback_success_ctx.last_error == EMEL_ERR_BACKEND);

  action::on_unexpected(event::reserve{
                          .error_out = &err,
                        },
                        rollback_success_ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}
