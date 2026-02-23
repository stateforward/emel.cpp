#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/coordinator/any.hpp"

TEST_CASE("memory_coordinator_any_routes_lifecycle_to_kv_variant") {
  emel::memory::coordinator::any machine{};
  machine.set_kind(emel::memory::coordinator::coordinator_kind::kv);

  int32_t err = EMEL_OK;
  CHECK(machine.process_event(emel::memory::coordinator::event::reserve{
      .kv_size = 12,
      .recurrent_slot_capacity = 2,
      .n_stream = 1,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  CHECK(
      machine.process_event(emel::memory::coordinator::event::allocate_sequence{
          .seq_id = 7,
          .slot_count = 3,
          .error_out = &err,
      }));
  CHECK(err == EMEL_OK);
  CHECK(machine.has_sequence(7));

  CHECK(machine.process_event(emel::memory::coordinator::event::free_sequence{
      .seq_id = 7,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK_FALSE(machine.has_sequence(7));
}

TEST_CASE(
    "memory_coordinator_any_hybrid_branch_rollback_preserves_consistency") {
  emel::memory::coordinator::any machine{};
  machine.set_kind(emel::memory::coordinator::coordinator_kind::hybrid);

  int32_t err = EMEL_OK;
  CHECK(machine.process_event(emel::memory::coordinator::event::reserve{
      .kv_size = 10,
      .recurrent_slot_capacity = 1,
      .n_stream = 1,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  CHECK(
      machine.process_event(emel::memory::coordinator::event::allocate_sequence{
          .seq_id = 0,
          .slot_count = 2,
          .error_out = &err,
      }));
  CHECK(err == EMEL_OK);

  CHECK_FALSE(
      machine.process_event(emel::memory::coordinator::event::branch_sequence{
          .seq_id_src = 0,
          .seq_id_dst = 1,
          .error_out = &err,
      }));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(machine.has_sequence(0));
  CHECK_FALSE(machine.has_sequence(1));
  CHECK(machine.last_error() == EMEL_ERR_BACKEND);
}

TEST_CASE("memory_coordinator_any_routes_lifecycle_to_recurrent_variant") {
  emel::memory::coordinator::any machine{};
  machine.set_kind(emel::memory::coordinator::coordinator_kind::recurrent);

  int32_t err = EMEL_OK;
  CHECK(machine.process_event(emel::memory::coordinator::event::reserve{
      .recurrent_slot_capacity = 2,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  CHECK(
      machine.process_event(emel::memory::coordinator::event::allocate_sequence{
          .seq_id = 5,
          .slot_count = 1,
          .error_out = &err,
      }));
  CHECK(err == EMEL_OK);
  CHECK(machine.has_sequence(5));

  CHECK(machine.process_event(emel::memory::coordinator::event::branch_sequence{
      .seq_id_src = 5,
      .seq_id_dst = 6,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(machine.has_sequence(6));

  CHECK(machine.process_event(emel::memory::coordinator::event::free_sequence{
      .seq_id = 6,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK_FALSE(machine.has_sequence(6));
}

TEST_CASE("memory_coordinator_any_routes_branch_and_free_for_all_variants") {
  emel::memory::coordinator::any machine{};
  int32_t err = EMEL_OK;

  machine.set_kind(emel::memory::coordinator::coordinator_kind::kv);
  CHECK(machine.process_event(emel::memory::coordinator::event::reserve{
      .kv_size = 10,
      .n_stream = 1,
      .error_out = &err,
  }));
  CHECK(
      machine.process_event(emel::memory::coordinator::event::allocate_sequence{
          .seq_id = 1,
          .slot_count = 3,
          .error_out = &err,
      }));
  CHECK(machine.process_event(emel::memory::coordinator::event::branch_sequence{
      .seq_id_src = 1,
      .seq_id_dst = 2,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(machine.has_sequence(2));

  machine.set_kind(emel::memory::coordinator::coordinator_kind::hybrid);
  CHECK(machine.process_event(emel::memory::coordinator::event::reserve{
      .kv_size = 8,
      .recurrent_slot_capacity = 2,
      .n_stream = 1,
      .error_out = &err,
  }));
  CHECK(
      machine.process_event(emel::memory::coordinator::event::allocate_sequence{
          .seq_id = 3,
          .slot_count = 2,
          .error_out = &err,
      }));
  CHECK(machine.process_event(emel::memory::coordinator::event::free_sequence{
      .seq_id = 3,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK_FALSE(machine.has_sequence(3));
}

TEST_CASE("memory_coordinator_any_last_error_uses_core_for_prepare_events") {
  emel::memory::coordinator::any machine{};
  machine.set_kind(emel::memory::coordinator::coordinator_kind::recurrent);

  int32_t err = EMEL_ERR_BACKEND;
  emel::memory::coordinator::event::memory_status status =
      emel::memory::coordinator::event::memory_status::failed_prepare;
  CHECK(machine.process_event(emel::memory::coordinator::event::prepare_update{
      .optimize = false,
      .status_out = &status,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(machine.last_error() == EMEL_OK);
}

TEST_CASE("memory_coordinator_any_invalid_kind_falls_back_to_backend_error") {
  emel::memory::coordinator::any machine{};
  machine.set_kind(
      static_cast<emel::memory::coordinator::coordinator_kind>(255));

  int32_t err = EMEL_OK;
  CHECK_FALSE(machine.process_event(emel::memory::coordinator::event::reserve{
      .kv_size = 8,
      .recurrent_slot_capacity = 1,
      .n_stream = 1,
      .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(machine.last_error() == EMEL_ERR_BACKEND);

  CHECK_FALSE(
      machine.process_event(emel::memory::coordinator::event::allocate_sequence{
          .seq_id = 1,
          .slot_count = 2,
          .error_out = &err,
      }));
  CHECK(err == EMEL_ERR_BACKEND);

  CHECK_FALSE(
      machine.process_event(emel::memory::coordinator::event::branch_sequence{
          .seq_id_src = 1,
          .seq_id_dst = 2,
          .error_out = &err,
      }));
  CHECK(err == EMEL_ERR_BACKEND);

  CHECK_FALSE(
      machine.process_event(emel::memory::coordinator::event::free_sequence{
          .seq_id = 1,
          .error_out = &err,
      }));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK_FALSE(machine.has_sequence(1));
}

TEST_CASE("memory_coordinator_any_kind_switch_resets_lifecycle_variant_state") {
  emel::memory::coordinator::any machine{};
  int32_t err = EMEL_OK;

  machine.set_kind(emel::memory::coordinator::coordinator_kind::kv);
  CHECK(machine.process_event(emel::memory::coordinator::event::reserve{
      .kv_size = 8,
      .n_stream = 1,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(
      machine.process_event(emel::memory::coordinator::event::allocate_sequence{
          .seq_id = 4,
          .slot_count = 2,
          .error_out = &err,
      }));
  CHECK(err == EMEL_OK);
  CHECK(machine.has_sequence(4));

  machine.set_kind(emel::memory::coordinator::coordinator_kind::recurrent);
  CHECK(machine.process_event(emel::memory::coordinator::event::reserve{
      .recurrent_slot_capacity = 2,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  machine.set_kind(emel::memory::coordinator::coordinator_kind::kv);
  CHECK_FALSE(machine.has_sequence(4));
}
