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
  int32_t callback_error = static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::none));
};

bool copy_state_cb(const int32_t, const int32_t, void * user_data, int32_t * error_out) {
  const auto * probe = static_cast<const copy_probe *>(user_data);
  if (error_out != nullptr) {
    *error_out = probe != nullptr ? probe->callback_error : static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::backend_error));
  }
  return probe != nullptr && probe->succeed;
}

}  // namespace

TEST_CASE("memory_hybrid_lifecycle_allocate_rolls_back_on_recurrent_failure") {
  hybrid_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::none));

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
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::backend_error)));
  CHECK_FALSE(machine.view().is_sequence_active(2));
}

TEST_CASE("memory_hybrid_lifecycle_branch_rolls_back_kv_when_recurrent_fails") {
  hybrid_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::none));

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
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::invalid_request)));
  CHECK_FALSE(machine.view().is_sequence_active(1));
  CHECK(machine.view().lookup_kv_block(1, 0) == -1);
}

TEST_CASE("memory_hybrid_lifecycle_free_consistent_across_kv_and_recurrent") {
  hybrid_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::none));
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

  CHECK_FALSE(machine.view().is_sequence_active(0));
  CHECK_FALSE(machine.view().is_sequence_active(1));
  CHECK(machine.view().lookup_kv_block(0, 0) == -1);
  CHECK(machine.view().lookup_kv_block(1, 0) == -1);
  CHECK(machine.view().lookup_recurrent_slot(0) == -1);
  CHECK(machine.view().lookup_recurrent_slot(1) == -1);
}

TEST_CASE("memory_hybrid_lifecycle_validation_and_unexpected_event_paths") {
  hybrid_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::none));

  CHECK_FALSE(machine.process_event(event::reserve{
    .max_sequences = 999999,
    .max_blocks = 8,
    .block_tokens = 2,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::invalid_request)));

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
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::invalid_request)));

  CHECK_FALSE(machine.process_event(event::rollback_slots{
    .seq_id = -1,
    .token_count = 1,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::invalid_request)));

  CHECK(machine.process_event(emel::memory::events::rollback_slots_done{}));

  err = static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::none));
  CHECK(machine.process_event(event::reserve{
    .max_sequences = 4,
    .max_blocks = 2,
    .block_tokens = 2,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::none)));
}

TEST_CASE("memory_hybrid_view_snapshot_tracks_combined_state") {
  hybrid_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::none));

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 4,
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

  emel::memory::view::snapshot view{};
  emel::error::type view_err = emel::error::cast(error::none);
  REQUIRE(machine.try_view(view, view_err));
  CHECK(view_err == emel::error::cast(error::none));
  CHECK(view.is_sequence_active(0));
  CHECK(view.sequence_length(0) == 4);
  CHECK(view.lookup_kv_block(0, 0) >= 0);
  CHECK(view.lookup_recurrent_slot(0) >= 0);
}
