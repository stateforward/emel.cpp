#include <doctest/doctest.h>

#include <cstdint>

#include "emel/error/error.hpp"
#include "emel/graph/errors.hpp"
#include "emel/graph/events.hpp"
#include "emel/graph/guards.hpp"
#include "emel/graph/sm.hpp"

namespace {

using execute_t = emel::graph::processor::event::execute;

bool validate_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool prepare_graph_reuse(const execute_t &, bool * reused_out, int32_t * err_out) {
  if (reused_out != nullptr) {
    *reused_out = true;
  }
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool alloc_graph_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool bind_inputs_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool run_kernel_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool extract_outputs_ok(const execute_t &, int32_t * outputs_out, int32_t * err_out) {
  if (outputs_out != nullptr) {
    *outputs_out = 1;
  }
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

struct reserve_callbacks {
  bool done_called = false;
  bool error_called = false;
  emel::graph::event::reserve_output done_output = {};
  int32_t error_code = 0;

  static bool on_done(void * owner,
                      const emel::graph::events::reserve_done & ev) noexcept {
    auto * self = static_cast<reserve_callbacks *>(owner);
    self->done_called = true;
    self->done_output = ev.output;
    return true;
  }

  static bool on_error(void * owner,
                       const emel::graph::events::reserve_error & ev) noexcept {
    auto * self = static_cast<reserve_callbacks *>(owner);
    self->error_called = true;
    self->error_code = ev.err;
    return true;
  }
};

struct compute_callbacks {
  bool done_called = false;
  bool error_called = false;
  emel::graph::event::compute_output done_output = {};
  int32_t error_code = 0;

  static bool on_done(void * owner,
                      const emel::graph::events::compute_done & ev) noexcept {
    auto * self = static_cast<compute_callbacks *>(owner);
    self->done_called = true;
    self->done_output = ev.output;
    return true;
  }

  static bool on_error(void * owner,
                       const emel::graph::events::compute_error & ev) noexcept {
    auto * self = static_cast<compute_callbacks *>(owner);
    self->error_called = true;
    self->error_code = ev.err;
    return true;
  }
};

}  // namespace

TEST_CASE("graph_machine_reserve_then_compute_success_path") {
  emel::graph::sm machine{};

  emel::graph::event::reserve_output reserve_output{};
  reserve_callbacks reserve_cb{};
  const emel::graph::event::reserve reserve_request{
    .model_topology = reinterpret_cast<const void *>(0xA5),
    .output_out = &reserve_output,
    .max_node_count = 4u,
    .max_tensor_count = 5u,
    .bytes_per_tensor = 8u,
    .workspace_capacity_bytes = 64u,
    .dispatch_done = {&reserve_cb, reserve_callbacks::on_done},
    .dispatch_error = {&reserve_cb, reserve_callbacks::on_error},
  };
  REQUIRE(machine.process_event(reserve_request));
  REQUIRE(reserve_cb.done_called);
  REQUIRE_FALSE(reserve_cb.error_called);

  emel::graph::event::compute_output compute_output{};
  compute_callbacks compute_cb{};
  const emel::graph::event::compute compute_request{
    .step_plan = reinterpret_cast<const void *>(0xB6),
    .output_out = &compute_output,
    .node_count_hint = reserve_output.node_count,
    .tensor_count_hint = reserve_output.tensor_count,
    .bytes_per_tensor = 8u,
    .workspace_capacity_bytes = 64u,
    .step_index = 0,
    .step_size = 1,
    .kv_tokens = 1,
    .expected_outputs = 1,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_kernel = run_kernel_ok,
    .extract_outputs = extract_outputs_ok,
    .dispatch_done = {&compute_cb, compute_callbacks::on_done},
    .dispatch_error = {&compute_cb, compute_callbacks::on_error},
  };

  CHECK(machine.process_event(compute_request));
  CHECK(compute_cb.done_called);
  CHECK_FALSE(compute_cb.error_called);
  CHECK(compute_output.graph_topology == reserve_output.graph_topology);
  CHECK(compute_output.node_count == reserve_output.node_count);
  CHECK(compute_output.tensor_count == reserve_output.tensor_count);
  CHECK(compute_output.required_buffer_bytes == reserve_output.required_buffer_bytes);
  CHECK(compute_output.version == reserve_output.version);
  CHECK(compute_output.reused_topology == 1u);
  CHECK(compute_output.outputs_produced == 1);
  CHECK(compute_output.graph_reused == 1u);
}

TEST_CASE("graph_machine_dispatches_invalid_compute_error") {
  emel::graph::sm machine{};

  emel::graph::event::reserve_output reserve_output{};
  reserve_callbacks reserve_cb{};
  const emel::graph::event::reserve reserve_request{
    .model_topology = reinterpret_cast<const void *>(0xA5),
    .output_out = &reserve_output,
    .max_node_count = 4u,
    .max_tensor_count = 5u,
    .bytes_per_tensor = 8u,
    .workspace_capacity_bytes = 64u,
    .dispatch_done = {&reserve_cb, reserve_callbacks::on_done},
    .dispatch_error = {&reserve_cb, reserve_callbacks::on_error},
  };
  REQUIRE(machine.process_event(reserve_request));
  REQUIRE(reserve_cb.done_called);

  emel::graph::event::compute_output compute_output{
    .graph_topology = reinterpret_cast<const void *>(0xFE),
    .node_count = 9u,
    .tensor_count = 9u,
    .required_buffer_bytes = 9u,
    .version = 9u,
    .reused_topology = 1u,
    .outputs_produced = 9,
    .graph_reused = 1u,
  };
  compute_callbacks compute_cb{};
  const emel::graph::event::compute compute_request{
    .step_plan = nullptr,
    .output_out = &compute_output,
    .node_count_hint = reserve_output.node_count,
    .tensor_count_hint = reserve_output.tensor_count,
    .bytes_per_tensor = 8u,
    .workspace_capacity_bytes = 64u,
    .step_index = 0,
    .step_size = 1,
    .kv_tokens = 1,
    .expected_outputs = 1,
    .validate = validate_ok,
    .prepare_graph = prepare_graph_reuse,
    .alloc_graph = alloc_graph_ok,
    .bind_inputs = bind_inputs_ok,
    .run_kernel = run_kernel_ok,
    .extract_outputs = extract_outputs_ok,
    .dispatch_done = {&compute_cb, compute_callbacks::on_done},
    .dispatch_error = {&compute_cb, compute_callbacks::on_error},
  };

  CHECK_FALSE(machine.process_event(compute_request));
  CHECK_FALSE(compute_cb.done_called);
  CHECK(compute_cb.error_called);
  CHECK(compute_cb.error_code ==
        static_cast<int32_t>(emel::error::cast(emel::graph::error::invalid_request)));
  CHECK(compute_output.graph_topology == nullptr);
  CHECK(compute_output.node_count == 0u);
  CHECK(compute_output.tensor_count == 0u);
  CHECK(compute_output.required_buffer_bytes == 0u);
  CHECK(compute_output.version == 0u);
  CHECK(compute_output.reused_topology == 0u);
  CHECK(compute_output.outputs_produced == 0);
  CHECK(compute_output.graph_reused == 0u);
}

TEST_CASE("graph_compute_error_guard_classification") {
  emel::graph::action::context ctx{};
  emel::graph::event::compute_output output{};
  compute_callbacks callbacks{};
  emel::graph::event::compute request{
    .output_out = &output,
    .dispatch_done = {&callbacks, compute_callbacks::on_done},
    .dispatch_error = {&callbacks, compute_callbacks::on_error},
  };
  emel::graph::event::compute_ctx phase_ctx{};
  emel::graph::event::compute_graph ev{request, phase_ctx};

  phase_ctx.err = emel::error::cast(emel::graph::error::none);
  CHECK(emel::graph::guard::compute_error_none{}(ev, ctx));

  phase_ctx.err = emel::error::cast(emel::graph::error::invalid_request);
  CHECK(emel::graph::guard::compute_error_invalid_request{}(ev, ctx));

  phase_ctx.err = emel::error::cast(emel::graph::error::assembler_failed);
  CHECK(emel::graph::guard::compute_error_assembler_failed{}(ev, ctx));

  phase_ctx.err = emel::error::cast(emel::graph::error::processor_failed);
  CHECK(emel::graph::guard::compute_error_processor_failed{}(ev, ctx));

  phase_ctx.err = emel::error::cast(emel::graph::error::busy);
  CHECK(emel::graph::guard::compute_error_busy{}(ev, ctx));

  phase_ctx.err = emel::error::cast(emel::graph::error::internal_error);
  CHECK(emel::graph::guard::compute_error_internal_error{}(ev, ctx));

  phase_ctx.err = emel::error::cast(emel::graph::error::untracked);
  CHECK(emel::graph::guard::compute_error_untracked{}(ev, ctx));

  phase_ctx.err = static_cast<emel::error::type>(0x7fff);
  CHECK(emel::graph::guard::compute_error_unknown{}(ev, ctx));
}
