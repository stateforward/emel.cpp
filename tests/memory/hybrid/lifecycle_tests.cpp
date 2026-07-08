#include <cstdint>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/events.hpp"
#include "emel/memory/hybrid/sm.hpp"
#include "../recording_kv_actor.hpp"

namespace {

using hybrid_sm = emel::memory::hybrid::sm;
using namespace emel::memory::hybrid;

struct copy_probe {
  bool succeed = true;
  int32_t callback_error =
      static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::none));
};

struct route_probe_sm : hybrid_sm {
  using hybrid_sm::hybrid_sm;

  void reject_bound_allocate_slots() noexcept {
    this->context_.kv_actor.dispatch_allocate_slots =
        &emel::memory::hybrid::reject_kv_allocate_slots;
  }

  void reject_bound_capture_view() noexcept {
    this->context_.kv_actor.dispatch_capture_view =
        &emel::memory::hybrid::reject_kv_capture_view;
  }
};

bool copy_state_cb(const int32_t, const int32_t, void * user_data, int32_t * error_out) {
  const auto * probe = static_cast<const copy_probe *>(user_data);
  if (error_out != nullptr) {
    *error_out =
        probe != nullptr
            ? probe->callback_error
            : static_cast<int32_t>(
                  emel::error::cast(emel::memory::hybrid::error::backend_error));
  }
  return probe != nullptr && probe->succeed;
}

}  // namespace

TEST_CASE("memory_hybrid_uses_injected_kv_actor") {
  emel::memory::test::recording_kv_actor kv{};
  hybrid_sm machine{emel::memory::hybrid::bind_kv_actor(kv)};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::none));

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 4,
    .max_blocks = 4,
    .block_tokens = 2,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 0,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 0,
    .token_count = 2,
    .error_out = &err,
  }));

  emel::memory::view::snapshot view{};
  emel::error::type view_err = emel::error::cast(error::none);
  REQUIRE(machine.try_view(view, view_err));

  CHECK(kv.reserve_count == 1);
  CHECK(kv.allocate_sequence_count == 1);
  CHECK(kv.allocate_slots_count == 1);
  CHECK(kv.capture_view_count == 1);
  CHECK(view.is_sequence_active(0));
  CHECK(view.lookup_kv_block(0, 0) >= 0);
}

TEST_CASE("memory_hybrid_owned_kv_allocate_slots_uses_direct_dispatch") {
  route_probe_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::none));
  int32_t block_count = 0;

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 4,
    .max_blocks = 4,
    .block_tokens = 2,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 0,
    .error_out = &err,
  }));

  machine.reject_bound_allocate_slots();

  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 0,
    .token_count = 2,
    .block_count_out = &block_count,
    .error_out = &err,
  }));

  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::none)));
  CHECK(block_count == 1);
  CHECK(machine.view().lookup_kv_block(0, 0) >= 0);
}

TEST_CASE("memory_hybrid_owned_kv_capture_view_uses_direct_dispatch") {
  route_probe_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::none));

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 4,
    .max_blocks = 4,
    .block_tokens = 2,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_sequence{
    .seq_id = 0,
    .error_out = &err,
  }));
  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 0,
    .token_count = 2,
    .error_out = &err,
  }));

  machine.reject_bound_capture_view();

  emel::memory::view::snapshot view{};
  emel::error::type view_err = emel::error::cast(error::none);
  REQUIRE(machine.try_view(view, view_err));

  CHECK(view_err == emel::error::cast(emel::memory::hybrid::error::none));
  CHECK(view.lookup_kv_block(0, 0) >= 0);
}

TEST_CASE("memory_hybrid_rejects_partial_kv_actor_binding") {
  int actor = 0;
  hybrid_sm machine{emel::memory::hybrid::kv_binding{.actor = &actor}};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::none));

  CHECK_FALSE(machine.process_event(event::reserve{
    .max_sequences = 4,
    .max_blocks = 4,
    .block_tokens = 2,
    .error_out = &err,
  }));
  CHECK(err == static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::backend_error)));
}

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

TEST_CASE("memory_hybrid_interleaved_sequences_isolate_and_recycle_blocks") {
  hybrid_sm machine{};
  int32_t err = static_cast<int32_t>(emel::error::cast(emel::memory::hybrid::error::none));

  REQUIRE(machine.process_event(event::reserve{
    .max_sequences = 3,
    .max_blocks = 4,
    .block_tokens = 2,
    .error_out = &err,
  }));

  // Interleaved growth: seq 0 takes two blocks, seq 1 takes one, and the
  // block sets plus recurrent slots stay disjoint.
  REQUIRE(machine.process_event(event::allocate_sequence{.seq_id = 0, .error_out = &err}));
  REQUIRE(machine.process_event(event::allocate_sequence{.seq_id = 1, .error_out = &err}));
  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 0, .token_count = 3, .error_out = &err}));
  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 1, .token_count = 2, .error_out = &err}));
  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 0, .token_count = 1, .error_out = &err}));

  emel::memory::view::snapshot view{};
  emel::error::type view_err = emel::error::cast(error::none);
  REQUIRE(machine.try_view(view, view_err));
  CHECK(view.sequence_length(0) == 4);
  CHECK(view.sequence_length(1) == 2);
  const int32_t seq0_block0 = view.lookup_kv_block(0, 0);
  const int32_t seq0_block1 = view.lookup_kv_block(0, 2);
  const int32_t seq1_block0 = view.lookup_kv_block(1, 0);
  CHECK(seq0_block0 == 0);
  CHECK(seq0_block1 == 1);
  CHECK(seq1_block0 == 2);
  CHECK(view.lookup_recurrent_slot(0) != view.lookup_recurrent_slot(1));

  // Freeing seq 0 recycles its blocks without disturbing seq 1; a new
  // sequence reuses the reclaimed ids in logical order so repeated runs keep
  // the maintained flash-identity contract when no other sequence intervenes.
  REQUIRE(machine.process_event(event::free_sequence{.seq_id = 0, .error_out = &err}));
  REQUIRE(machine.process_event(event::allocate_sequence{.seq_id = 2, .error_out = &err}));
  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 2, .token_count = 4, .error_out = &err}));

  REQUIRE(machine.try_view(view, view_err));
  CHECK_FALSE(view.is_sequence_active(0));
  CHECK(view.sequence_length(1) == 2);
  CHECK(view.lookup_kv_block(1, 0) == seq1_block0);
  const int32_t seq2_block0 = view.lookup_kv_block(2, 0);
  const int32_t seq2_block1 = view.lookup_kv_block(2, 2);
  CHECK(seq2_block0 != seq1_block0);
  CHECK(seq2_block1 != seq1_block0);
  CHECK(seq2_block0 != seq2_block1);
  CHECK(seq2_block0 == 0);
  CHECK(seq2_block1 == 1);
}
