#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <string_view>

#include "emel/error/error.hpp"
#include "emel/graph/errors.hpp"
#include "emel/graph/events.hpp"
#include "emel/graph/guards.hpp"
#include "emel/graph/sm.hpp"
#include "emel/model/llama/detail.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/tensor/errors.hpp"
#include "emel/tensor/events.hpp"

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

struct lifecycle_fixture {
  int32_t input_tokens[2] = {1, 2};
  std::array<emel::graph::processor::event::lifecycle_tensor_binding, 2> tensors{{
      {
          .tensor_id = 0,
          .buffer = reinterpret_cast<void *>(0xCAFE),
          .buffer_bytes = 64u,
          .consumer_refs = 0,
          .is_leaf = true,
      },
      {
          .tensor_id = 1,
          .buffer = input_tokens,
          .buffer_bytes = sizeof(input_tokens),
          .consumer_refs = 1,
          .is_leaf = false,
      },
  }};
  std::array<int32_t, 2> required_ids = {0, 1};
  std::array<int32_t, 1> publish_ids = {1};
  std::array<int32_t, 1> release_ids = {1};
  emel::graph::processor::event::lifecycle_phase phase{
    .required_filled_ids = required_ids.data(),
    .required_filled_count = static_cast<int32_t>(required_ids.size()),
    .publish_ids = publish_ids.data(),
    .publish_count = static_cast<int32_t>(publish_ids.size()),
    .release_ids = release_ids.data(),
    .release_count = static_cast<int32_t>(release_ids.size()),
  };
  emel::graph::processor::event::lifecycle_manifest reserve{
    .tensors = tensors.data(),
    .tensor_count = static_cast<int32_t>(tensors.size()),
    .phase = nullptr,
  };
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

TEST_CASE("graph_machine_reserve_then_compute_success_path") {
  emel::graph::sm machine{};
  lifecycle_fixture lifecycle{};

  emel::graph::event::reserve_output reserve_output{};
  reserve_callbacks reserve_cb{};
  const emel::graph::event::reserve reserve_request{
    .model_topology = reinterpret_cast<const void *>(0xA5),
    .output_out = &reserve_output,
    .lifecycle = &lifecycle.reserve,
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
  CHECK(reserve_output.lifecycle == &lifecycle.reserve);

  emel::tensor::event::tensor_state tensor_state{};
  emel::error::type tensor_err = emel::error::cast(emel::tensor::error::none);
  REQUIRE(machine.try_capture_tensor(0, tensor_state, tensor_err));
  CHECK(tensor_state.lifecycle_state == emel::tensor::event::lifecycle::leaf_filled);
  REQUIRE(machine.try_capture_tensor(1, tensor_state, tensor_err));
  CHECK(tensor_state.lifecycle_state == emel::tensor::event::lifecycle::empty);

  emel::graph::event::compute_output compute_output{};
  compute_callbacks compute_cb{};
  const emel::graph::event::compute compute_request{
    .step_plan = reinterpret_cast<const void *>(0xB6),
    .output_out = &compute_output,
    .lifecycle = &lifecycle.compute,
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
  CHECK(compute_output.lifecycle == &lifecycle.compute);
}

TEST_CASE("graph_machine_dispatches_invalid_compute_error") {
  emel::graph::sm machine{};
  lifecycle_fixture lifecycle{};

  emel::graph::event::reserve_output reserve_output{};
  reserve_callbacks reserve_cb{};
  const emel::graph::event::reserve reserve_request{
    .model_topology = reinterpret_cast<const void *>(0xA5),
    .output_out = &reserve_output,
    .lifecycle = &lifecycle.reserve,
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
    .lifecycle = &lifecycle.compute,
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

TEST_CASE("graph_machine_accepts_canonical_descriptor_handles") {
  auto model = std::make_unique<emel::model::data>();
  build_canonical_model(*model, 2);
  lifecycle_fixture lifecycle{};

  emel::model::llama::detail::execution_view execution = {};
  REQUIRE(emel::model::llama::detail::build_execution_view(*model, execution) ==
          emel::error::cast(emel::model::loader::error::none));
  emel::model::llama::detail::topology topology = {};
  REQUIRE(emel::model::llama::detail::build_topology(execution, topology) ==
          emel::error::cast(emel::model::loader::error::none));
  emel::model::llama::detail::step_plan prefill = {};
  emel::model::llama::detail::step_plan decode = {};
  REQUIRE(emel::model::llama::detail::build_step_plans(topology, prefill, decode) ==
          emel::error::cast(emel::model::loader::error::none));

  emel::graph::sm machine{};

  emel::graph::event::reserve_output reserve_output{};
  reserve_callbacks reserve_cb{};
  const emel::graph::event::reserve reserve_request{
    .model_topology = &topology,
    .output_out = &reserve_output,
    .lifecycle = &lifecycle.reserve,
    .max_node_count = topology.node_count,
    .max_tensor_count = topology.tensor_count,
    .bytes_per_tensor = topology.bytes_per_tensor,
    .workspace_capacity_bytes = topology.workspace_capacity_bytes,
    .dispatch_done = {&reserve_cb, reserve_callbacks::on_done},
    .dispatch_error = {&reserve_cb, reserve_callbacks::on_error},
  };
  REQUIRE(machine.process_event(reserve_request));
  REQUIRE(reserve_cb.done_called);

  emel::graph::event::compute_output compute_output{};
  compute_callbacks compute_cb{};
  const emel::graph::event::compute compute_request{
    .step_plan = &prefill,
    .output_out = &compute_output,
    .lifecycle = &lifecycle.compute,
    .node_count_hint = prefill.node_count,
    .tensor_count_hint = prefill.tensor_count,
    .bytes_per_tensor = topology.bytes_per_tensor,
    .workspace_capacity_bytes = topology.workspace_capacity_bytes,
    .step_index = 0,
    .step_size = 1,
    .kv_tokens = 1,
    .expected_outputs = prefill.expected_outputs,
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
  CHECK(compute_output.graph_topology == &topology);
  CHECK(compute_output.node_count == topology.node_count);
  CHECK(compute_output.tensor_count == topology.tensor_count);
  CHECK(compute_output.outputs_produced == 1);
}
