#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/kv/sm.hpp"

TEST_CASE("kv_lifecycle_allocate_branch_free_sequence") {
  emel::memory::kv::sm machine{};
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::memory::kv::event::reserve{
      .kv_size = 16,
      .n_stream = 1,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  CHECK(machine.process_event(emel::memory::kv::event::allocate_sequence{
      .seq_id = 0,
      .slot_count = 4,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(machine.has_sequence(0));
  CHECK(machine.sequence_token_count(0) == 4);

  CHECK(machine.process_event(emel::memory::kv::event::branch_sequence{
      .seq_id_src = 0,
      .seq_id_dst = 1,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(machine.has_sequence(1));
  CHECK(machine.sequence_token_count(1) == 4);

  CHECK(machine.process_event(emel::memory::kv::event::free_sequence{
      .seq_id = 1,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK_FALSE(machine.has_sequence(1));

  CHECK(machine.process_event(emel::memory::kv::event::free_sequence{
      .seq_id = 0,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK_FALSE(machine.has_sequence(0));
}

TEST_CASE("kv_lifecycle_reports_capacity_errors_without_partial_leaks") {
  emel::memory::kv::sm machine{};
  int32_t err = EMEL_OK;

  CHECK(machine.process_event(emel::memory::kv::event::reserve{
      .kv_size = 2,
      .n_stream = 1,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);

  CHECK(machine.process_event(emel::memory::kv::event::allocate_sequence{
      .seq_id = 0,
      .slot_count = 2,
      .error_out = &err,
  }));
  CHECK(err == EMEL_OK);
  CHECK(machine.has_sequence(0));

  CHECK_FALSE(machine.process_event(emel::memory::kv::event::allocate_sequence{
      .seq_id = 1,
      .slot_count = 1,
      .error_out = &err,
  }));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK_FALSE(machine.has_sequence(1));
  CHECK(machine.has_sequence(0));
}
