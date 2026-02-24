#include <cstdint>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/events.hpp"
#include "emel/memory/recurrent/sm.hpp"

namespace {

using recurrent_sm = emel::memory::recurrent::sm;
using namespace emel::memory::recurrent;

struct copy_probe {
  int32_t calls = 0;
  int32_t src_slot = -1;
  int32_t dst_slot = -1;
  bool succeed = true;
  int32_t callback_error = EMEL_OK;
};

bool copy_state_cb(const int32_t src_slot, const int32_t dst_slot, void * user_data,
                   int32_t * error_out) {
  auto * probe = static_cast<copy_probe *>(user_data);
  if (probe != nullptr) {
    probe->calls += 1;
    probe->src_slot = src_slot;
    probe->dst_slot = dst_slot;
  }
  if (error_out != nullptr) {
    *error_out = probe != nullptr ? probe->callback_error : EMEL_ERR_BACKEND;
  }
  return probe != nullptr && probe->succeed;
}

}  // namespace

TEST_CASE("memory_recurrent_lifecycle_slot_oom_and_reuse_determinism") {
  recurrent_sm machine{};
  int32_t err = EMEL_OK;

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 4,
    .max_blocks = 2,
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

  REQUIRE(machine.process_event(event::free_sequence{
    .seq_id = 0,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 2,
    .error_out = &err,
  }));
  CHECK(machine.lookup_recurrent_slot(2) == 0);
}

TEST_CASE("memory_recurrent_lifecycle_branch_invokes_copy_callback_once") {
  recurrent_sm machine{};
  int32_t err = EMEL_OK;
  copy_probe probe{};

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 8,
    .max_blocks = 8,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 0,
    .error_out = &err,
  }));

  REQUIRE(machine.process_event(event::branch_sequence{
    .parent_seq_id = 0,
    .child_seq_id = 1,
    .copy_state = &copy_state_cb,
    .copy_state_user_data = &probe,
    .error_out = &err,
  }));

  CHECK(probe.calls == 1);
  CHECK(probe.src_slot == machine.lookup_recurrent_slot(0));
  CHECK(probe.dst_slot == machine.lookup_recurrent_slot(1));
}

TEST_CASE("memory_recurrent_lifecycle_branch_callback_failure_rolls_back") {
  recurrent_sm machine{};
  int32_t err = EMEL_OK;
  copy_probe probe{};
  probe.succeed = false;
  probe.callback_error = EMEL_ERR_BACKEND;

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 8,
    .max_blocks = 8,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 0,
    .error_out = &err,
  }));

  CHECK_FALSE(machine.process_event(event::branch_sequence{
    .parent_seq_id = 0,
    .child_seq_id = 1,
    .copy_state = &copy_state_cb,
    .copy_state_user_data = &probe,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK_FALSE(machine.is_sequence_active(1));
  CHECK(machine.lookup_recurrent_slot(1) == -1);

  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 1,
    .error_out = &err,
  }));
  CHECK(machine.lookup_recurrent_slot(1) == 1);
}

TEST_CASE("memory_recurrent_lifecycle_validation_and_unexpected_event_paths") {
  recurrent_sm machine{};
  int32_t err = EMEL_OK;

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 4,
    .max_blocks = 2,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 0,
    .error_out = &err,
  }));

  CHECK_FALSE(machine.process_event(event::allocate_slots{
    .seq_id = 2,
    .token_count = 0,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(machine.process_event(event::branch_sequence{
    .parent_seq_id = 0,
    .child_seq_id = 1,
    .copy_state = nullptr,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(machine.process_event(event::free_sequence{
    .seq_id = -1,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  CHECK_FALSE(machine.process_event(event::rollback_slots{
    .seq_id = 1,
    .token_count = 1,
    .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  CHECK(machine.process_event(emel::memory::events::branch_sequence_done{}));
  CHECK(machine.last_error() == EMEL_ERR_BACKEND);
}

TEST_CASE("memory_recurrent_action_direct_failure_and_helper_paths") {
  action::context ctx{};
  int32_t err = EMEL_OK;

  action::run_reserve_phase(event::reserve{
                              .max_sequences = 4,
                              .max_blocks = 2,
                              .error_out = &err,
                            },
                            ctx);
  REQUIRE(err == EMEL_OK);

  CHECK(action::sequence_length_value(ctx, 3) == 0);
  CHECK_FALSE(action::activate_slot(ctx, -1));
  CHECK_FALSE(action::deactivate_slot(ctx, -1));

  action::run_reserve_phase(event::reserve{
                              .max_sequences = 999999,
                              .max_blocks = 1,
                              .error_out = &err,
                            },
                            ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  action::run_reserve_phase(event::reserve{
                              .max_sequences = 4,
                              .max_blocks = 2,
                              .error_out = &err,
                            },
                            ctx);
  REQUIRE(err == EMEL_OK);

  action::run_allocate_sequence_phase(event::allocate_sequence{
                                        .seq_id = -1,
                                        .error_out = &err,
                                      },
                                      ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  action::run_allocate_sequence_phase(event::allocate_sequence{
                                        .seq_id = 0,
                                        .error_out = &err,
                                      },
                                      ctx);
  REQUIRE(err == EMEL_OK);
  action::run_allocate_sequence_phase(event::allocate_sequence{
                                        .seq_id = 0,
                                        .error_out = &err,
                                      },
                                      ctx);
  CHECK(err == EMEL_OK);

  int32_t block_count = -1;
  action::run_allocate_slots_phase(event::allocate_slots{
                                     .seq_id = 0,
                                     .token_count = 2,
                                     .block_count_out = &block_count,
                                     .error_out = &err,
                                   },
                                   ctx);
  CHECK(err == EMEL_OK);
  CHECK(block_count == 0);

  action::run_allocate_sequence_phase(event::allocate_sequence{
                                        .seq_id = 1,
                                        .error_out = &err,
                                      },
                                      ctx);
  REQUIRE(err == EMEL_OK);

  copy_probe probe{};
  action::run_branch_sequence_phase(event::branch_sequence{
                                      .parent_seq_id = 0,
                                      .child_seq_id = 1,
                                      .copy_state = &copy_state_cb,
                                      .copy_state_user_data = &probe,
                                      .error_out = &err,
                                    },
                                    ctx);
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  action::context oom_ctx{};
  action::run_reserve_phase(event::reserve{
                              .max_sequences = 2,
                              .max_blocks = 1,
                              .error_out = &err,
                            },
                            oom_ctx);
  REQUIRE(err == EMEL_OK);
  action::run_allocate_sequence_phase(event::allocate_sequence{
                                        .seq_id = 0,
                                        .error_out = &err,
                                      },
                                      oom_ctx);
  REQUIRE(err == EMEL_OK);
  action::run_branch_sequence_phase(event::branch_sequence{
                                      .parent_seq_id = 0,
                                      .child_seq_id = 1,
                                      .copy_state = &copy_state_cb,
                                      .copy_state_user_data = &probe,
                                      .error_out = &err,
                                    },
                                    oom_ctx);
  CHECK(err == EMEL_ERR_BACKEND);

  action::context bad_free_ctx{};
  action::run_reserve_phase(event::reserve{
                              .max_sequences = 4,
                              .max_blocks = 2,
                              .error_out = &err,
                            },
                            bad_free_ctx);
  REQUIRE(err == EMEL_OK);
  bad_free_ctx.seq_to_slot[0] = 0;
  bad_free_ctx.slot_owner_seq[0] = 0;
  bad_free_ctx.slots.storage().active[0] = 0;
  action::run_free_sequence_phase(event::free_sequence{
                                    .seq_id = 0,
                                    .error_out = &err,
                                  },
                                  bad_free_ctx);
  CHECK(err == EMEL_ERR_BACKEND);

  int32_t rollback_block_count = -1;
  action::run_rollback_slots_phase(event::rollback_slots{
                                     .seq_id = 0,
                                     .token_count = 1,
                                     .block_count_out = &rollback_block_count,
                                     .error_out = &err,
                                   },
                                   ctx);
  CHECK(err == EMEL_OK);
  CHECK(rollback_block_count == 0);

  bad_free_ctx.last_error = EMEL_OK;
  action::ensure_last_error(bad_free_ctx);
  CHECK(bad_free_ctx.last_error == EMEL_ERR_BACKEND);

  action::on_unexpected(event::reserve{
                          .error_out = &err,
                        },
                        bad_free_ctx);
  CHECK(err == EMEL_ERR_BACKEND);
}
