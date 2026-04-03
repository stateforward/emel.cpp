#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>

#include "emel/error/error.hpp"
#include "emel/graph/assembler/errors.hpp"
#include "emel/graph/assembler/events.hpp"
#include "emel/graph/assembler/sm.hpp"
#include "emel/model/builder/detail.hpp"
#include "emel/model/detail.hpp"
#include "emel/model/loader/errors.hpp"

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

struct lifecycle_fixture {
  std::array<emel::graph::processor::event::lifecycle_tensor_binding, 1> tensors{{
      {
          .tensor_id = 0,
          .buffer = reinterpret_cast<void *>(0xA0),
          .buffer_bytes = 32u,
          .consumer_refs = 0,
          .is_leaf = true,
      },
  }};
  emel::graph::processor::event::lifecycle_manifest reserve{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .phase = nullptr,
  };
  emel::graph::processor::event::lifecycle_phase phase{};
  emel::graph::processor::event::lifecycle_manifest compute{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .phase = &phase,
  };
};

void copy_architecture(std::array<char, emel::model::data::k_max_architecture_name> & dest,
                       const std::string_view value) {
  dest.fill('\0');
  const size_t count = std::min(dest.size() - 1u, value.size());
  for (size_t i = 0; i < count; ++i) {
    dest[i] = value[i];
  }
}

void append_tensor_name(emel::model::data & model,
                        emel::model::data::tensor_record & tensor,
                        const std::string_view name) {
  tensor.name_offset = model.name_bytes_used;
  tensor.name_length = static_cast<uint32_t>(name.size());
  for (size_t i = 0; i < name.size(); ++i) {
    model.name_storage[model.name_bytes_used + static_cast<uint32_t>(i)] = name[i];
  }
  model.name_bytes_used += static_cast<uint32_t>(name.size());
  tensor.n_dims = 2;
  tensor.dims[0] = 8;
  tensor.dims[1] = 8;
  tensor.data = &tensor;
  tensor.data_size = 64u;
}

void build_canonical_model(emel::model::data & model, const int32_t block_count) {
  std::memset(&model, 0, sizeof(model));
  copy_architecture(model.architecture_name, "llama");
  model.n_layers = block_count;
  model.params.n_embd = 64;
  model.params.n_ctx = 128;
  model.weights_data = model.tensors.data();
  model.weights_size = 4096u;

  uint32_t tensor_index = 0u;
  const auto add = [&](const std::string_view name) {
    append_tensor_name(model, model.tensors[tensor_index], name);
    ++tensor_index;
  };
  const auto add_block = [&](const int32_t block, const std::string_view suffix) {
    add(std::string{"blk."} + std::to_string(block) + "." + std::string{suffix});
  };

  add("token_embd.weight");
  add("output_norm.weight");
  add("output.weight");
  for (int32_t block = 0; block < block_count; ++block) {
    add_block(block, "attn_norm.weight");
    add_block(block, "attn_q.weight");
    add_block(block, "attn_k.weight");
    add_block(block, "attn_v.weight");
    add_block(block, "attn_output.weight");
    add_block(block, "ffn_norm.weight");
    add_block(block, "ffn_gate.weight");
    add_block(block, "ffn_down.weight");
    add_block(block, "ffn_up.weight");
  }
  model.n_tensors = tensor_index;
}

}  // namespace

TEST_CASE("graph_assembler_reserve_successful_path") {
  emel::graph::assembler::sm machine{};
  lifecycle_fixture lifecycle{};
  emel::graph::assembler::event::reserve_output output{};
  reserve_callbacks callbacks{};

  const emel::graph::assembler::event::reserve request{
    .model_topology = reinterpret_cast<const void *>(0x11),
    .output_out = &output,
    .lifecycle = &lifecycle.reserve,
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
  CHECK(output.lifecycle == &lifecycle.reserve);
}

TEST_CASE("graph_assembler_reserve_rejects_invalid_request") {
  emel::graph::assembler::sm machine{};
  lifecycle_fixture lifecycle{};
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
    .lifecycle = &lifecycle.reserve,
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
  lifecycle_fixture lifecycle{};

  emel::graph::assembler::event::reserve_output reserve_output{};
  reserve_callbacks reserve_callbacks_state{};
  const emel::graph::assembler::event::reserve reserve_request{
    .model_topology = reinterpret_cast<const void *>(0x99),
    .output_out = &reserve_output,
    .lifecycle = &lifecycle.reserve,
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
    .lifecycle = &lifecycle.compute,
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
  CHECK(assemble_output.lifecycle == &lifecycle.reserve);
}

TEST_CASE("graph_assembler_accepts_canonical_descriptor_handles") {
  auto model = std::make_unique<emel::model::data>();
  build_canonical_model(*model, 2);
  lifecycle_fixture lifecycle{};

  emel::model::builder::detail::execution_view execution = {};
  REQUIRE(emel::model::builder::detail::build_execution_view(*model, execution) ==
          emel::error::cast(emel::model::loader::error::none));
  emel::model::builder::detail::topology topology = {};
  REQUIRE(emel::model::builder::detail::build_topology(execution, topology) ==
          emel::error::cast(emel::model::loader::error::none));
  emel::model::builder::detail::step_plan prefill = {};
  emel::model::builder::detail::step_plan decode = {};
  REQUIRE(emel::model::builder::detail::build_step_plans(topology, prefill, decode) ==
          emel::error::cast(emel::model::loader::error::none));

  emel::graph::assembler::sm machine{};
  emel::graph::assembler::event::reserve_output reserve_output{};
  reserve_callbacks reserve_state{};
  const emel::graph::assembler::event::reserve reserve_request{
    .model_topology = &topology,
    .output_out = &reserve_output,
    .lifecycle = &lifecycle.reserve,
    .max_node_count = topology.node_count,
    .max_tensor_count = topology.tensor_count,
    .bytes_per_tensor = topology.bytes_per_tensor,
    .workspace_capacity_bytes = topology.workspace_capacity_bytes,
    .dispatch_done = {&reserve_state, reserve_callbacks::on_done},
    .dispatch_error = {&reserve_state, reserve_callbacks::on_error},
  };
  REQUIRE(machine.process_event(reserve_request));
  REQUIRE(reserve_state.done_called);

  emel::graph::assembler::event::assemble_output assemble_output{};
  assemble_callbacks assemble_state{};
  const emel::graph::assembler::event::assemble assemble_request{
    .step_plan = &prefill,
    .output_out = &assemble_output,
    .lifecycle = &lifecycle.compute,
    .node_count_hint = prefill.node_count,
    .tensor_count_hint = prefill.tensor_count,
    .bytes_per_tensor = topology.bytes_per_tensor,
    .workspace_capacity_bytes = topology.workspace_capacity_bytes,
    .dispatch_done = {&assemble_state, assemble_callbacks::on_done},
    .dispatch_error = {&assemble_state, assemble_callbacks::on_error},
  };

  CHECK(machine.process_event(assemble_request));
  CHECK(assemble_state.done_called);
  CHECK_FALSE(assemble_state.error_called);
  CHECK(assemble_output.graph_topology == &topology);
  CHECK(assemble_output.node_count == topology.node_count);
  CHECK(assemble_output.tensor_count == topology.tensor_count);
  CHECK(assemble_output.lifecycle == &lifecycle.reserve);
}
