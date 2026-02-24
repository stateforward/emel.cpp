#include <cstdint>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/memory/coordinator/any.hpp"
#include "emel/memory/view.hpp"

namespace {

using coordinator_any = emel::memory::coordinator::any;
using coordinator_kind = emel::memory::coordinator::coordinator_kind;
using namespace emel::memory::coordinator;

}  // namespace

TEST_CASE("memory_coordinator_any_variant_switch_and_invalid_kind_normalization") {
  coordinator_any machine{};
  CHECK(machine.kind() == coordinator_kind::recurrent);

  machine.set_kind(static_cast<coordinator_kind>(99));
  CHECK(machine.kind() == coordinator_kind::recurrent);

  machine.set_kind(coordinator_kind::kv);
  CHECK(machine.kind() == coordinator_kind::kv);
}

TEST_CASE("memory_coordinator_any_routes_events_to_active_variant_only") {
  coordinator_any machine{};
  int32_t err = EMEL_OK;

  machine.set_kind(coordinator_kind::kv);
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
  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 3,
    .token_count = 2,
    .error_out = &err,
  }));
  auto kv_view = machine.view();
  CHECK(kv_view.lookup_kv_block(3, 0) >= 0);
  CHECK(kv_view.lookup_recurrent_slot(3) == -1);

  machine.set_kind(coordinator_kind::recurrent);
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
  auto recurrent_view = machine.view();
  CHECK(recurrent_view.lookup_recurrent_slot(3) >= 0);
  CHECK(recurrent_view.lookup_kv_block(3, 0) == -1);
}

TEST_CASE("memory_coordinator_any_forwards_view_queries") {
  coordinator_any machine{};
  int32_t err = EMEL_OK;

  machine.set_kind(coordinator_kind::hybrid);
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
  REQUIRE(machine.process_event(event::allocate_slots{
    .seq_id = 1,
    .token_count = 2,
    .error_out = &err,
  }));

  const auto view = machine.view();
  CHECK(view.is_sequence_active(1));
  CHECK(view.sequence_length(1) == 2);
  CHECK(view.lookup_kv_block(1, 0) >= 0);
  CHECK(view.lookup_recurrent_slot(1) >= 0);
}

TEST_CASE("memory_view_any_defaults_when_unbound") {
  const emel::memory::view::any view = {};
  CHECK_FALSE(view.is_sequence_active(0));
  CHECK(view.sequence_length(0) == 0);
  CHECK(view.lookup_kv_block(0, 0) == -1);
  CHECK(view.lookup_recurrent_slot(0) == -1);
}
