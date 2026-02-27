#include <doctest/doctest.h>

#include <cstdint>
#include <limits>

#include "emel/error/error.hpp"
#include "emel/graph/assembler/actions.hpp"
#include "emel/graph/assembler/assemble_alloc_pass/actions.hpp"
#include "emel/graph/assembler/assemble_alloc_pass/guards.hpp"
#include "emel/graph/assembler/assemble_build_pass/actions.hpp"
#include "emel/graph/assembler/assemble_build_pass/guards.hpp"
#include "emel/graph/assembler/assemble_validate_pass/actions.hpp"
#include "emel/graph/assembler/assemble_validate_pass/guards.hpp"
#include "emel/graph/assembler/errors.hpp"
#include "emel/graph/assembler/events.hpp"
#include "emel/graph/assembler/guards.hpp"
#include "emel/graph/assembler/reserve_alloc_pass/actions.hpp"
#include "emel/graph/assembler/reserve_alloc_pass/guards.hpp"
#include "emel/graph/assembler/reserve_build_pass/actions.hpp"
#include "emel/graph/assembler/reserve_build_pass/guards.hpp"
#include "emel/graph/assembler/reserve_validate_pass/actions.hpp"
#include "emel/graph/assembler/reserve_validate_pass/guards.hpp"
#include "emel/graph/assembler/reuse_decision_pass/actions.hpp"
#include "emel/graph/assembler/reuse_decision_pass/guards.hpp"

namespace {

struct reserve_dispatch_state {
  bool done_called = false;
  bool error_called = false;
  emel::graph::assembler::event::reserve_output done_output = {};
  emel::graph::assembler::event::reserve_output error_output = {};
  int32_t error_code = 0;

  static bool on_done(void * owner,
                      const emel::graph::assembler::events::reserve_done & ev) noexcept {
    auto * self = static_cast<reserve_dispatch_state *>(owner);
    self->done_called = true;
    self->done_output = ev.output;
    return true;
  }

  static bool on_error(void * owner,
                       const emel::graph::assembler::events::reserve_error & ev) noexcept {
    auto * self = static_cast<reserve_dispatch_state *>(owner);
    self->error_called = true;
    self->error_output = ev.output;
    self->error_code = ev.err;
    return true;
  }
};

struct assemble_dispatch_state {
  bool done_called = false;
  bool error_called = false;
  emel::graph::assembler::event::assemble_output done_output = {};
  emel::graph::assembler::event::assemble_output error_output = {};
  int32_t error_code = 0;

  static bool on_done(void * owner,
                      const emel::graph::assembler::events::assemble_done & ev) noexcept {
    auto * self = static_cast<assemble_dispatch_state *>(owner);
    self->done_called = true;
    self->done_output = ev.output;
    return true;
  }

  static bool on_error(void * owner,
                       const emel::graph::assembler::events::assemble_error & ev) noexcept {
    auto * self = static_cast<assemble_dispatch_state *>(owner);
    self->error_called = true;
    self->error_output = ev.output;
    self->error_code = ev.err;
    return true;
  }
};

emel::graph::assembler::event::reserve make_valid_reserve(
    emel::graph::assembler::event::reserve_output * output,
    reserve_dispatch_state * state) {
  return emel::graph::assembler::event::reserve{
    .model_topology = reinterpret_cast<const void *>(0xAA11),
    .output_out = output,
    .max_node_count = 6u,
    .max_tensor_count = 8u,
    .bytes_per_tensor = 16u,
    .workspace_capacity_bytes = 256u,
    .dispatch_done = {state, reserve_dispatch_state::on_done},
    .dispatch_error = {state, reserve_dispatch_state::on_error},
  };
}

emel::graph::assembler::event::assemble make_valid_assemble(
    emel::graph::assembler::event::assemble_output * output,
    assemble_dispatch_state * state) {
  return emel::graph::assembler::event::assemble{
    .step_plan = reinterpret_cast<const void *>(0xBB22),
    .output_out = output,
    .node_count_hint = 6u,
    .tensor_count_hint = 8u,
    .bytes_per_tensor = 16u,
    .workspace_capacity_bytes = 256u,
    .dispatch_done = {state, assemble_dispatch_state::on_done},
    .dispatch_error = {state, assemble_dispatch_state::on_error},
  };
}

}  // namespace

TEST_CASE("graph_assembler_action_and_guard_branches") {
  namespace action = emel::graph::assembler::action;
  namespace event = emel::graph::assembler::event;
  namespace guard = emel::graph::assembler::guard;
  using assembler_error = emel::graph::assembler::error;

  action::context machine_ctx{};

  reserve_dispatch_state reserve_state{};
  event::reserve_output reserve_output{};
  event::reserve reserve_request = make_valid_reserve(&reserve_output, &reserve_state);
  event::reserve_ctx reserve_ctx{};
  event::reserve_graph reserve_ev{reserve_request, reserve_ctx};

  CHECK(guard::valid_reserve{}(reserve_ev, machine_ctx));
  CHECK_FALSE(guard::invalid_reserve{}(reserve_ev, machine_ctx));
  reserve_request.model_topology = nullptr;
  CHECK(guard::invalid_reserve{}(reserve_ev, machine_ctx));
  CHECK(guard::invalid_reserve_with_dispatchable_output{}(reserve_ev, machine_ctx));
  reserve_request.dispatch_error = {};
  CHECK(guard::invalid_reserve_with_output_only{}(reserve_ev, machine_ctx));
  reserve_request.output_out = nullptr;
  CHECK(guard::invalid_reserve_without_output{}(reserve_ev, machine_ctx));

  reserve_request = make_valid_reserve(&reserve_output, &reserve_state);
  reserve_ctx = {};
  action::begin_reserve(reserve_ev, machine_ctx);
  CHECK(reserve_ctx.err == emel::error::cast(assembler_error::none));

  reserve_state = {};
  action::reject_invalid_reserve_with_dispatch(reserve_ev, machine_ctx);
  CHECK(reserve_state.error_called);
  CHECK(reserve_output.graph_topology == nullptr);

  reserve_output = {.graph_topology = reinterpret_cast<const void *>(0x1), .node_count = 1u};
  action::reject_invalid_reserve_with_output_only(reserve_ev, machine_ctx);
  CHECK(reserve_output.graph_topology == nullptr);
  action::reject_invalid_reserve_without_output(reserve_ev, machine_ctx);
  CHECK(reserve_ctx.err == emel::error::cast(assembler_error::invalid_request));

  reserve_ctx.assembled_node_count = 6u;
  reserve_ctx.assembled_tensor_count = 8u;
  reserve_ctx.alloc_plan.required_buffer_bytes = 128u;
  action::commit_reserve_result(reserve_ev, machine_ctx);
  CHECK(machine_ctx.reserved_topology == reserve_request.model_topology);
  CHECK(machine_ctx.reserved_node_count == 6u);
  CHECK(machine_ctx.reserved_tensor_count == 8u);
  CHECK(machine_ctx.reserved_required_buffer_bytes == 128u);
  CHECK(machine_ctx.topology_version == 1u);

  reserve_state = {};
  action::dispatch_reserve_done(reserve_ev, machine_ctx);
  CHECK(reserve_state.done_called);
  reserve_state = {};
  reserve_ctx.err = emel::error::cast(assembler_error::capacity);
  action::dispatch_reserve_error(reserve_ev, machine_ctx);
  CHECK(reserve_state.error_called);
  CHECK(reserve_state.error_code ==
        static_cast<int32_t>(emel::error::cast(assembler_error::capacity)));

  assemble_dispatch_state assemble_state{};
  event::assemble_output assemble_output{};
  event::assemble assemble_request = make_valid_assemble(&assemble_output, &assemble_state);
  event::assemble_ctx assemble_ctx{};
  event::assemble_graph assemble_ev{assemble_request, assemble_ctx};

  CHECK(guard::valid_assemble{}(assemble_ev, machine_ctx));
  CHECK_FALSE(guard::invalid_assemble{}(assemble_ev, machine_ctx));
  assemble_request.step_plan = nullptr;
  CHECK(guard::invalid_assemble{}(assemble_ev, machine_ctx));
  CHECK(guard::invalid_assemble_with_dispatchable_output{}(assemble_ev, machine_ctx));
  assemble_request.dispatch_error = {};
  CHECK(guard::invalid_assemble_with_output_only{}(assemble_ev, machine_ctx));
  assemble_request.output_out = nullptr;
  CHECK(guard::invalid_assemble_without_output{}(assemble_ev, machine_ctx));

  assemble_request = make_valid_assemble(&assemble_output, &assemble_state);
  assemble_ctx = {};
  action::begin_assemble(assemble_ev, machine_ctx);
  CHECK(assemble_ctx.err == emel::error::cast(assembler_error::none));

  assemble_state = {};
  action::reject_invalid_assemble_with_dispatch(assemble_ev, machine_ctx);
  CHECK(assemble_state.error_called);
  action::reject_invalid_assemble_with_output_only(assemble_ev, machine_ctx);
  action::reject_invalid_assemble_without_output(assemble_ev, machine_ctx);

  action::commit_assemble_reuse_result(assemble_ev, machine_ctx);
  CHECK(assemble_output.reused_topology == 1u);

  assemble_ctx.assembled_node_count = 9u;
  assemble_ctx.assembled_tensor_count = 10u;
  assemble_ctx.alloc_plan.required_buffer_bytes = 320u;
  action::commit_assemble_rebuild_result(assemble_ev, machine_ctx);
  CHECK(assemble_output.reused_topology == 0u);
  CHECK(machine_ctx.reserved_topology == assemble_request.step_plan);

  assemble_state = {};
  action::dispatch_assemble_done(assemble_ev, machine_ctx);
  CHECK(assemble_state.done_called);
  assemble_state = {};
  assemble_ctx.err = emel::error::cast(assembler_error::internal_error);
  action::dispatch_assemble_error(assemble_ev, machine_ctx);
  CHECK(assemble_state.error_called);

  reserve_ctx.err = emel::error::cast(assembler_error::none);
  reserve_ctx.validate_outcome = emel::graph::assembler::reserve_validate_pass::events::phase_outcome::done;
  reserve_ctx.build_outcome = emel::graph::assembler::reserve_build_pass::events::phase_outcome::done;
  reserve_ctx.alloc_outcome = emel::graph::assembler::reserve_alloc_pass::events::phase_outcome::done;
  CHECK(guard::reserve_validate_done{}(reserve_ev, machine_ctx));
  CHECK(guard::reserve_build_done{}(reserve_ev, machine_ctx));
  CHECK(guard::reserve_alloc_done{}(reserve_ev, machine_ctx));
  CHECK(guard::reserve_phase_ok{}(reserve_ev, machine_ctx));
  reserve_ctx.err = emel::error::cast(assembler_error::capacity);
  CHECK(guard::reserve_validate_failed{}(reserve_ev, machine_ctx));
  CHECK(guard::reserve_build_failed{}(reserve_ev, machine_ctx));
  CHECK(guard::reserve_alloc_failed{}(reserve_ev, machine_ctx));
  CHECK(guard::reserve_phase_failed{}(reserve_ev, machine_ctx));

  assemble_ctx.err = emel::error::cast(assembler_error::none);
  assemble_ctx.validate_outcome = emel::graph::assembler::assemble_validate_pass::events::phase_outcome::done;
  assemble_ctx.reuse_outcome = emel::graph::assembler::reuse_decision_pass::events::phase_outcome::reused;
  assemble_ctx.build_outcome = emel::graph::assembler::assemble_build_pass::events::phase_outcome::done;
  assemble_ctx.alloc_outcome = emel::graph::assembler::assemble_alloc_pass::events::phase_outcome::done;
  CHECK(guard::assemble_validate_done{}(assemble_ev, machine_ctx));
  CHECK(guard::reuse_decision_reused{}(assemble_ev, machine_ctx));
  assemble_ctx.reuse_outcome = emel::graph::assembler::reuse_decision_pass::events::phase_outcome::rebuild;
  CHECK(guard::reuse_decision_rebuild{}(assemble_ev, machine_ctx));
  CHECK(guard::assemble_build_done{}(assemble_ev, machine_ctx));
  CHECK(guard::assemble_alloc_done{}(assemble_ev, machine_ctx));
  CHECK(guard::assemble_phase_ok{}(assemble_ev, machine_ctx));
  assemble_ctx.err = emel::error::cast(assembler_error::internal_error);
  CHECK(guard::assemble_validate_failed{}(assemble_ev, machine_ctx));
  CHECK(guard::reuse_decision_failed{}(assemble_ev, machine_ctx));
  CHECK(guard::assemble_build_failed{}(assemble_ev, machine_ctx));
  CHECK(guard::assemble_alloc_failed{}(assemble_ev, machine_ctx));
  CHECK(guard::assemble_phase_failed{}(assemble_ev, machine_ctx));

  action::on_unexpected(reserve_ev, machine_ctx);
  action::on_unexpected(assemble_ev, machine_ctx);
}

TEST_CASE("graph_assembler_pass_action_and_guard_branches") {
  namespace event = emel::graph::assembler::event;
  using assembler_error = emel::graph::assembler::error;

  emel::graph::assembler::action::context machine_ctx{};

  reserve_dispatch_state reserve_state{};
  event::reserve_output reserve_output{};
  event::reserve reserve_request = make_valid_reserve(&reserve_output, &reserve_state);
  event::reserve_ctx reserve_ctx{};
  event::reserve_graph reserve_ev{reserve_request, reserve_ctx};

  // reserve_validate
  reserve_ctx.err = emel::error::cast(assembler_error::none);
  CHECK(emel::graph::assembler::reserve_validate_pass::guard::phase_done{}(reserve_ev, machine_ctx));
  reserve_request.model_topology = nullptr;
  CHECK(emel::graph::assembler::reserve_validate_pass::guard::phase_invalid_request{}(reserve_ev, machine_ctx));
  reserve_request = make_valid_reserve(&reserve_output, &reserve_state);
  emel::graph::assembler::reserve_validate_pass::action::mark_done(reserve_ev, machine_ctx);
  emel::graph::assembler::reserve_validate_pass::action::mark_failed_invalid_request(reserve_ev, machine_ctx);
  emel::graph::assembler::reserve_validate_pass::action::on_unexpected(reserve_ev, machine_ctx);
  emel::graph::assembler::reserve_validate_pass::action::on_unexpected(0, machine_ctx);

  // reserve_build
  reserve_ctx = {};
  reserve_ctx.err = emel::error::cast(assembler_error::none);
  reserve_ctx.validate_outcome = emel::graph::assembler::reserve_validate_pass::events::phase_outcome::done;
  CHECK(emel::graph::assembler::reserve_build_pass::guard::phase_done{}(reserve_ev, machine_ctx));
  reserve_ctx.validate_outcome = emel::graph::assembler::reserve_validate_pass::events::phase_outcome::failed;
  CHECK(emel::graph::assembler::reserve_build_pass::guard::phase_prereq_failed{}(reserve_ev, machine_ctx));
  reserve_ctx.validate_outcome = emel::graph::assembler::reserve_validate_pass::events::phase_outcome::done;
  reserve_request.workspace_capacity_bytes = 1u;
  CHECK(emel::graph::assembler::reserve_build_pass::guard::phase_capacity_exceeded{}(reserve_ev, machine_ctx));
  reserve_request.max_node_count = 0u;
  CHECK(emel::graph::assembler::reserve_build_pass::guard::phase_invalid_request{}(reserve_ev, machine_ctx));
  reserve_request = make_valid_reserve(&reserve_output, &reserve_state);
  emel::graph::assembler::reserve_build_pass::action::mark_done(reserve_ev, machine_ctx);
  emel::graph::assembler::reserve_build_pass::action::mark_failed_prereq(reserve_ev, machine_ctx);
  emel::graph::assembler::reserve_build_pass::action::mark_failed_capacity(reserve_ev, machine_ctx);
  emel::graph::assembler::reserve_build_pass::action::mark_failed_invalid_request(reserve_ev, machine_ctx);
  emel::graph::assembler::reserve_build_pass::action::on_unexpected(reserve_ev, machine_ctx);
  emel::graph::assembler::reserve_build_pass::action::on_unexpected(0, machine_ctx);

  // reserve_alloc
  reserve_ctx = {};
  reserve_ctx.err = emel::error::cast(assembler_error::none);
  reserve_ctx.build_outcome = emel::graph::assembler::reserve_build_pass::events::phase_outcome::done;
  reserve_ctx.assembled_node_count = 6u;
  reserve_ctx.assembled_tensor_count = 8u;
  CHECK(emel::graph::assembler::reserve_alloc_pass::guard::phase_request_allocator{}(reserve_ev, machine_ctx));
  emel::graph::assembler::reserve_alloc_pass::action::request_allocator_plan(reserve_ev, machine_ctx);
  CHECK(reserve_ctx.alloc_outcome == emel::graph::assembler::reserve_alloc_pass::events::phase_outcome::done);

  reserve_ctx = {};
  reserve_ctx.err = emel::error::cast(assembler_error::none);
  reserve_ctx.build_outcome = emel::graph::assembler::reserve_build_pass::events::phase_outcome::done;
  reserve_ctx.assembled_node_count = 6u;
  reserve_ctx.assembled_tensor_count = 8u;
  reserve_request.workspace_capacity_bytes = 1u;
  emel::graph::assembler::reserve_alloc_pass::action::request_allocator_plan(reserve_ev, machine_ctx);
  CHECK(reserve_ctx.alloc_outcome == emel::graph::assembler::reserve_alloc_pass::events::phase_outcome::failed);
  CHECK(reserve_ctx.err != emel::error::cast(assembler_error::none));

  reserve_ctx = {};
  reserve_ctx.err = emel::error::cast(assembler_error::none);
  reserve_ctx.build_outcome = emel::graph::assembler::reserve_build_pass::events::phase_outcome::failed;
  CHECK(emel::graph::assembler::reserve_alloc_pass::guard::phase_prereq_failed{}(reserve_ev, machine_ctx));
  reserve_ctx.build_outcome = emel::graph::assembler::reserve_build_pass::events::phase_outcome::done;
  reserve_ctx.assembled_node_count = 0u;
  CHECK(emel::graph::assembler::reserve_alloc_pass::guard::phase_invalid_request{}(reserve_ev, machine_ctx));
  emel::graph::assembler::reserve_alloc_pass::action::mark_failed_prereq(reserve_ev, machine_ctx);
  emel::graph::assembler::reserve_alloc_pass::action::mark_failed_invalid_request(reserve_ev, machine_ctx);
  emel::graph::assembler::reserve_alloc_pass::action::on_unexpected(reserve_ev, machine_ctx);
  emel::graph::assembler::reserve_alloc_pass::action::on_unexpected(0, machine_ctx);

  // assemble_* passes
  assemble_dispatch_state assemble_state{};
  event::assemble_output assemble_output{};
  event::assemble assemble_request = make_valid_assemble(&assemble_output, &assemble_state);
  event::assemble_ctx assemble_ctx{};
  event::assemble_graph assemble_ev{assemble_request, assemble_ctx};

  assemble_ctx.err = emel::error::cast(assembler_error::none);
  CHECK(emel::graph::assembler::assemble_validate_pass::guard::phase_done{}(assemble_ev, machine_ctx));
  assemble_request.step_plan = nullptr;
  CHECK(emel::graph::assembler::assemble_validate_pass::guard::phase_invalid_request{}(assemble_ev, machine_ctx));
  assemble_request = make_valid_assemble(&assemble_output, &assemble_state);
  emel::graph::assembler::assemble_validate_pass::action::mark_done(assemble_ev, machine_ctx);
  emel::graph::assembler::assemble_validate_pass::action::mark_failed_invalid_request(assemble_ev, machine_ctx);
  emel::graph::assembler::assemble_validate_pass::action::on_unexpected(assemble_ev, machine_ctx);
  emel::graph::assembler::assemble_validate_pass::action::on_unexpected(0, machine_ctx);

  assemble_ctx = {};
  assemble_ctx.err = emel::error::cast(assembler_error::none);
  assemble_ctx.validate_outcome = emel::graph::assembler::assemble_validate_pass::events::phase_outcome::done;
  machine_ctx.reserved_topology = reinterpret_cast<const void *>(0xCC33);
  machine_ctx.reserved_node_count = 6u;
  machine_ctx.reserved_tensor_count = 8u;
  machine_ctx.reserved_required_buffer_bytes = 128u;
  machine_ctx.has_reserved_topology = 1u;
  assemble_request.node_count_hint = 6u;
  assemble_request.tensor_count_hint = 8u;
  CHECK(emel::graph::assembler::reuse_decision_pass::guard::phase_reuse{}(assemble_ev, machine_ctx));
  emel::graph::assembler::reuse_decision_pass::action::mark_reuse(assemble_ev, machine_ctx);
  CHECK(assemble_ctx.reused_topology == 1u);

  assemble_request.node_count_hint = 7u;
  assemble_request.tensor_count_hint = 9u;
  CHECK(emel::graph::assembler::reuse_decision_pass::guard::phase_rebuild{}(assemble_ev, machine_ctx));
  emel::graph::assembler::reuse_decision_pass::action::mark_rebuild(assemble_ev, machine_ctx);
  CHECK(assemble_ctx.reused_topology == 0u);

  assemble_ctx.validate_outcome = emel::graph::assembler::assemble_validate_pass::events::phase_outcome::failed;
  CHECK(emel::graph::assembler::reuse_decision_pass::guard::phase_prereq_failed{}(assemble_ev, machine_ctx));
  assemble_ctx.validate_outcome = emel::graph::assembler::assemble_validate_pass::events::phase_outcome::done;
  assemble_request.node_count_hint = 0u;
  CHECK(emel::graph::assembler::reuse_decision_pass::guard::phase_invalid_request{}(assemble_ev, machine_ctx));
  emel::graph::assembler::reuse_decision_pass::action::mark_failed_prereq(assemble_ev, machine_ctx);
  emel::graph::assembler::reuse_decision_pass::action::mark_failed_invalid_request(assemble_ev, machine_ctx);
  emel::graph::assembler::reuse_decision_pass::action::on_unexpected(assemble_ev, machine_ctx);
  emel::graph::assembler::reuse_decision_pass::action::on_unexpected(0, machine_ctx);

  assemble_ctx = {};
  assemble_ctx.err = emel::error::cast(assembler_error::none);
  assemble_ctx.reuse_outcome = emel::graph::assembler::reuse_decision_pass::events::phase_outcome::rebuild;
  assemble_ctx.assembled_node_count = 6u;
  assemble_ctx.assembled_tensor_count = 8u;
  assemble_request.bytes_per_tensor = 16u;
  assemble_request.workspace_capacity_bytes = 256u;
  CHECK(emel::graph::assembler::assemble_build_pass::guard::phase_done{}(assemble_ev, machine_ctx));
  emel::graph::assembler::assemble_build_pass::action::mark_done(assemble_ev, machine_ctx);
  assemble_ctx.reuse_outcome = emel::graph::assembler::reuse_decision_pass::events::phase_outcome::failed;
  CHECK(emel::graph::assembler::assemble_build_pass::guard::phase_prereq_failed{}(assemble_ev, machine_ctx));
  assemble_ctx.reuse_outcome = emel::graph::assembler::reuse_decision_pass::events::phase_outcome::rebuild;
  assemble_request.workspace_capacity_bytes = 1u;
  CHECK(emel::graph::assembler::assemble_build_pass::guard::phase_capacity_exceeded{}(assemble_ev, machine_ctx));
  assemble_request.bytes_per_tensor = 0u;
  CHECK(emel::graph::assembler::assemble_build_pass::guard::phase_invalid_request{}(assemble_ev, machine_ctx));
  emel::graph::assembler::assemble_build_pass::action::mark_failed_prereq(assemble_ev, machine_ctx);
  emel::graph::assembler::assemble_build_pass::action::mark_failed_capacity(assemble_ev, machine_ctx);
  emel::graph::assembler::assemble_build_pass::action::mark_failed_invalid_request(assemble_ev, machine_ctx);
  emel::graph::assembler::assemble_build_pass::action::on_unexpected(assemble_ev, machine_ctx);
  emel::graph::assembler::assemble_build_pass::action::on_unexpected(0, machine_ctx);

  assemble_ctx = {};
  assemble_ctx.err = emel::error::cast(assembler_error::none);
  assemble_ctx.build_outcome = emel::graph::assembler::assemble_build_pass::events::phase_outcome::done;
  assemble_ctx.assembled_node_count = 6u;
  assemble_ctx.assembled_tensor_count = 8u;
  assemble_request = make_valid_assemble(&assemble_output, &assemble_state);
  CHECK(emel::graph::assembler::assemble_alloc_pass::guard::phase_request_allocator{}(assemble_ev, machine_ctx));
  emel::graph::assembler::assemble_alloc_pass::action::request_allocator_plan(assemble_ev, machine_ctx);
  CHECK(assemble_ctx.alloc_outcome == emel::graph::assembler::assemble_alloc_pass::events::phase_outcome::done);

  assemble_ctx = {};
  assemble_ctx.err = emel::error::cast(assembler_error::none);
  assemble_ctx.build_outcome = emel::graph::assembler::assemble_build_pass::events::phase_outcome::done;
  assemble_ctx.assembled_node_count = 6u;
  assemble_ctx.assembled_tensor_count = 8u;
  assemble_request.workspace_capacity_bytes = 1u;
  emel::graph::assembler::assemble_alloc_pass::action::request_allocator_plan(assemble_ev, machine_ctx);
  CHECK(assemble_ctx.alloc_outcome == emel::graph::assembler::assemble_alloc_pass::events::phase_outcome::failed);

  assemble_ctx = {};
  assemble_ctx.err = emel::error::cast(assembler_error::none);
  assemble_ctx.build_outcome = emel::graph::assembler::assemble_build_pass::events::phase_outcome::failed;
  CHECK(emel::graph::assembler::assemble_alloc_pass::guard::phase_prereq_failed{}(assemble_ev, machine_ctx));
  assemble_ctx.build_outcome = emel::graph::assembler::assemble_build_pass::events::phase_outcome::done;
  assemble_ctx.assembled_node_count = 0u;
  CHECK(emel::graph::assembler::assemble_alloc_pass::guard::phase_invalid_request{}(assemble_ev, machine_ctx));
  emel::graph::assembler::assemble_alloc_pass::action::mark_failed_prereq(assemble_ev, machine_ctx);
  emel::graph::assembler::assemble_alloc_pass::action::mark_failed_invalid_request(assemble_ev, machine_ctx);
  emel::graph::assembler::assemble_alloc_pass::action::on_unexpected(assemble_ev, machine_ctx);
  emel::graph::assembler::assemble_alloc_pass::action::on_unexpected(0, machine_ctx);

  // exercise overflow helpers
  CHECK(emel::graph::assembler::reserve_build_pass::guard::product_overflows_u64(
      std::numeric_limits<uint64_t>::max(), 2u));
  CHECK(emel::graph::assembler::assemble_build_pass::guard::product_overflows_u64(
      std::numeric_limits<uint64_t>::max(), 2u));
}
