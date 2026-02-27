#include <doctest/doctest.h>

#include <cstdint>

#include "emel/error/error.hpp"
#include "emel/graph/assembler/errors.hpp"
#include "emel/graph/assembler/events.hpp"
#include "emel/graph/assembler/sm.hpp"

namespace {

struct reserve_callbacks {
  bool done_called = false;
  bool error_called = false;
  emel::graph::assembler::event::reserve_output done_output = {};
  emel::graph::assembler::event::reserve_output error_output = {};
  int32_t error_code = 0;

  static bool on_done(void * owner,
                      const emel::graph::assembler::events::reserve_done & ev) noexcept {
    auto * self = static_cast<reserve_callbacks *>(owner);
    self->done_called = true;
    self->done_output = ev.output;
    return true;
  }

  static bool on_error(void * owner,
                       const emel::graph::assembler::events::reserve_error & ev) noexcept {
    auto * self = static_cast<reserve_callbacks *>(owner);
    self->error_called = true;
    self->error_output = ev.output;
    self->error_code = ev.err;
    return true;
  }
};

struct assemble_callbacks {
  bool done_called = false;
  bool error_called = false;
  emel::graph::assembler::event::assemble_output done_output = {};
  emel::graph::assembler::event::assemble_output error_output = {};
  int32_t error_code = 0;

  static bool on_done(void * owner,
                      const emel::graph::assembler::events::assemble_done & ev) noexcept {
    auto * self = static_cast<assemble_callbacks *>(owner);
    self->done_called = true;
    self->done_output = ev.output;
    return true;
  }

  static bool on_error(void * owner,
                       const emel::graph::assembler::events::assemble_error & ev) noexcept {
    auto * self = static_cast<assemble_callbacks *>(owner);
    self->error_called = true;
    self->error_output = ev.output;
    self->error_code = ev.err;
    return true;
  }
};

}  // namespace

TEST_CASE("graph_assembler_reserve_successful_path") {
  emel::graph::assembler::sm machine{};
  emel::graph::assembler::event::reserve_output output{};
  reserve_callbacks callbacks{};

  const emel::graph::assembler::event::reserve request{
    .model_topology = reinterpret_cast<const void *>(0x11),
    .output_out = &output,
    .max_node_count = 6u,
    .max_tensor_count = 8u,
    .bytes_per_tensor = 16u,
    .workspace_capacity_bytes = 128u,
    .dispatch_done = {&callbacks, reserve_callbacks::on_done},
    .dispatch_error = {&callbacks, reserve_callbacks::on_error},
  };

  CHECK(machine.process_event(request));
  CHECK(callbacks.done_called);
  CHECK_FALSE(callbacks.error_called);
  CHECK(output.graph_topology == request.model_topology);
  CHECK(output.node_count == 6u);
  CHECK(output.tensor_count == 8u);
  CHECK(output.required_buffer_bytes == 128u);
  CHECK(output.version == 1u);
}

TEST_CASE("graph_assembler_reserve_rejects_invalid_request") {
  emel::graph::assembler::sm machine{};
  emel::graph::assembler::event::reserve_output output{
    .graph_topology = reinterpret_cast<const void *>(0x55),
    .node_count = 77u,
    .tensor_count = 77u,
    .required_buffer_bytes = 77u,
    .version = 77u,
  };
  reserve_callbacks callbacks{};

  const emel::graph::assembler::event::reserve request{
    .model_topology = nullptr,
    .output_out = &output,
    .max_node_count = 6u,
    .max_tensor_count = 8u,
    .bytes_per_tensor = 16u,
    .workspace_capacity_bytes = 128u,
    .dispatch_done = {&callbacks, reserve_callbacks::on_done},
    .dispatch_error = {&callbacks, reserve_callbacks::on_error},
  };

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(callbacks.done_called);
  CHECK(callbacks.error_called);
  CHECK(callbacks.error_code ==
        static_cast<int32_t>(emel::error::cast(emel::graph::assembler::error::invalid_request)));
  CHECK(output.graph_topology == nullptr);
  CHECK(output.node_count == 0u);
  CHECK(output.tensor_count == 0u);
  CHECK(output.required_buffer_bytes == 0u);
  CHECK(output.version == 0u);
}

TEST_CASE("graph_assembler_assemble_reuses_reserved_topology") {
  emel::graph::assembler::sm machine{};

  emel::graph::assembler::event::reserve_output reserve_output{};
  reserve_callbacks reserve_callbacks_state{};
  const emel::graph::assembler::event::reserve reserve_request{
    .model_topology = reinterpret_cast<const void *>(0x99),
    .output_out = &reserve_output,
    .max_node_count = 4u,
    .max_tensor_count = 5u,
    .bytes_per_tensor = 8u,
    .workspace_capacity_bytes = 64u,
    .dispatch_done = {&reserve_callbacks_state, reserve_callbacks::on_done},
    .dispatch_error = {&reserve_callbacks_state, reserve_callbacks::on_error},
  };
  REQUIRE(machine.process_event(reserve_request));
  REQUIRE(reserve_callbacks_state.done_called);

  emel::graph::assembler::event::assemble_output assemble_output{};
  assemble_callbacks assemble_callbacks_state{};
  const emel::graph::assembler::event::assemble assemble_request{
    .step_plan = reinterpret_cast<const void *>(0x1234),
    .output_out = &assemble_output,
    .node_count_hint = reserve_output.node_count,
    .tensor_count_hint = reserve_output.tensor_count,
    .bytes_per_tensor = 8u,
    .workspace_capacity_bytes = 64u,
    .dispatch_done = {&assemble_callbacks_state, assemble_callbacks::on_done},
    .dispatch_error = {&assemble_callbacks_state, assemble_callbacks::on_error},
  };

  CHECK(machine.process_event(assemble_request));
  CHECK(assemble_callbacks_state.done_called);
  CHECK_FALSE(assemble_callbacks_state.error_called);
  CHECK(assemble_output.reused_topology == 1u);
  CHECK(assemble_output.graph_topology == reserve_output.graph_topology);
  CHECK(assemble_output.node_count == reserve_output.node_count);
  CHECK(assemble_output.tensor_count == reserve_output.tensor_count);
  CHECK(assemble_output.required_buffer_bytes == reserve_output.required_buffer_bytes);
  CHECK(assemble_output.version == reserve_output.version);
}
