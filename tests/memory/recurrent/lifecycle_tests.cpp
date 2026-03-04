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
  int32_t callback_error = static_cast<int32_t>(emel::error::cast(emel::memory::recurrent::error::none));
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
    *error_out = probe != nullptr ? probe->callback_error : static_cast<int32_t>(emel::error::cast(emel::memory::recurrent::error::backend_error));
  }
  return probe != nullptr && probe->succeed;
}

}  // namespace

TEST_CASE("memory_recurrent_lifecycle_slot_oom_reuse_and_rollback") {
  recurrent_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::memory::recurrent::error::none));

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
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::memory::recurrent::error::backend_error)));

  REQUIRE(machine.process_event(event::free_sequence{
    .seq_id = 0,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 2,
    .error_out = &err,
  }));
  CHECK(machine.view().lookup_recurrent_slot(2) == 0);

  int32_t block_delta = -1;
  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 2,
    .token_count = 3,
    .block_count_out = &block_delta,
    .error_out = &err,
  }));
  CHECK(block_delta == 0);
  CHECK(machine.view().sequence_length(2) == 3);

  block_delta = -1;
  REQUIRE(machine.process_event(event::rollback_slots{
    .seq_id = 2,
    .token_count = 2,
    .block_count_out = &block_delta,
    .error_out = &err,
  }));
  CHECK(block_delta == 0);
  CHECK(machine.view().sequence_length(2) == 1);
}

TEST_CASE("memory_recurrent_lifecycle_branch_invokes_copy_callback_once") {
  recurrent_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::memory::recurrent::error::none));
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
  CHECK(probe.src_slot == machine.view().lookup_recurrent_slot(0));
  CHECK(probe.dst_slot == machine.view().lookup_recurrent_slot(1));
}

TEST_CASE("memory_recurrent_lifecycle_branch_callback_failure_rolls_back") {
  recurrent_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::memory::recurrent::error::none));
  copy_probe probe{};
  probe.succeed = false;
  probe.callback_error = static_cast<int32_t>(emel::error::cast(emel::memory::recurrent::error::backend_error));

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
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::memory::recurrent::error::backend_error)));
  CHECK_FALSE(machine.view().is_sequence_active(1));
  CHECK(machine.view().lookup_recurrent_slot(1) == -1);

  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 1,
    .error_out = &err,
  }));
  CHECK(machine.view().lookup_recurrent_slot(1) == 1);
}

TEST_CASE("memory_recurrent_lifecycle_validation_and_unexpected_event_paths") {
  recurrent_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::memory::recurrent::error::none));

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
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::memory::recurrent::error::invalid_request)));

  CHECK_FALSE(machine.process_event(event::branch_sequence{
    .parent_seq_id = 0,
    .child_seq_id = 1,
    .copy_state = nullptr,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::memory::recurrent::error::invalid_request)));

  CHECK_FALSE(machine.process_event(event::free_sequence{
    .seq_id = -1,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::memory::recurrent::error::invalid_request)));

  CHECK_FALSE(machine.process_event(event::rollback_slots{
    .seq_id = 1,
    .token_count = 1,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::memory::recurrent::error::invalid_request)));

  CHECK(machine.process_event(emel::memory::events::branch_sequence_done{}));

  err = static_cast<int32_t>(emel::error::cast(emel::memory::recurrent::error::none));
  CHECK(machine.process_event(event::reserve{
    .max_sequences = 4,
    .max_blocks = 2,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::memory::recurrent::error::none)));
}

TEST_CASE("memory_recurrent_view_snapshot_tracks_state") {
  recurrent_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::memory::recurrent::error::none));

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 8,
    .max_blocks = 8,
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

  emel::memory::view::snapshot snapshot{};
  emel::error::type view_err = emel::error::cast(error::none);
  REQUIRE(machine.try_view(snapshot, view_err));
  CHECK(view_err == emel::error::cast(error::none));
  CHECK(snapshot.is_sequence_active(0));
  CHECK(snapshot.sequence_length(0) == 4);
  CHECK(snapshot.lookup_kv_block(0, 0) == -1);
  CHECK(snapshot.lookup_recurrent_slot(0) >= 0);
}
