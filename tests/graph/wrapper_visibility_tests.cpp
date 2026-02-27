#include <doctest/doctest.h>

#include <concepts>
#include <type_traits>

#include "emel/graph/allocator/sm.hpp"
#include "emel/graph/assembler/sm.hpp"
#include "emel/graph/processor/sm.hpp"
#include "emel/graph/sm.hpp"

namespace {

template <class machine_type, class event_type>
concept has_public_process_event = requires(machine_type & machine, const event_type & ev) {
  { machine.process_event(ev) } -> std::convertible_to<bool>;
};

}  // namespace

TEST_CASE("graph_wrappers_hide_internal_event_dispatch_entrypoints") {
  CHECK_FALSE((has_public_process_event<emel::graph::sm, emel::graph::event::reserve_graph>));
  CHECK_FALSE((has_public_process_event<emel::graph::sm, emel::graph::event::compute_graph>));

  CHECK_FALSE(
      (has_public_process_event<emel::graph::assembler::sm, emel::graph::assembler::event::reserve_graph>));
  CHECK_FALSE(
      (has_public_process_event<emel::graph::assembler::sm, emel::graph::assembler::event::assemble_graph>));

  CHECK_FALSE(
      (has_public_process_event<emel::graph::allocator::sm, emel::graph::allocator::event::allocate_graph_plan>));

  CHECK_FALSE(
      (has_public_process_event<emel::graph::processor::sm, emel::graph::processor::event::execute_step>));
}
