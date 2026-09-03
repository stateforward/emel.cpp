#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <thread>

#include <doctest/doctest.h>

#include "emel/error/error.hpp"
#include "emel/graph/events.hpp"
#include "emel/graph/sm.hpp"
#include "emel/text/generator/decode_wavefront/sm.hpp"

namespace {

namespace wavefront = emel::text::generator::decode_wavefront;
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

bool run_kernel_counting(const execute_t & request, int32_t * err_out) {
  auto * calls = static_cast<int32_t *>(request.compute_ctx);
  *calls += 1;
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool run_kernel_rejected(const execute_t & request, int32_t * err_out) {
  auto * calls = static_cast<int32_t *>(request.compute_ctx);
  *calls += 1;
  if (err_out != nullptr) {
    *err_out = 1;
  }
  return false;
}

bool run_kernel_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

struct parallel_lane_context {
  std::atomic<int32_t> entered{0};
  std::atomic<int32_t> active{0};
  std::atomic<bool> release{false};
  std::atomic<bool> overlapped{false};
};

struct parallel_lane_owner {
  parallel_lane_context * shared = nullptr;
};

bool run_kernel_synchronized(const execute_t & request, int32_t * err_out) {
  auto * owner = static_cast<parallel_lane_owner *>(request.compute_ctx);
  auto & shared = *owner->shared;
  shared.active.fetch_add(1, std::memory_order_acq_rel);
  shared.entered.fetch_add(1, std::memory_order_release);
  for (int32_t attempt = 0; attempt < 100000; ++attempt) {
    if (shared.entered.load(std::memory_order_acquire) >= 2) {
      break;
    }
    std::this_thread::yield();
  }
  shared.overlapped.store(
      shared.active.load(std::memory_order_acquire) > 1,
      std::memory_order_release);
  shared.active.fetch_sub(1, std::memory_order_acq_rel);
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool run_kernel_wait_for_release(const execute_t & request, int32_t * err_out) {
  auto * owner = static_cast<parallel_lane_owner *>(request.compute_ctx);
  auto & shared = *owner->shared;
  shared.entered.fetch_add(1, std::memory_order_release);
  while (!shared.release.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
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

  static bool on_done(void * owner, const emel::graph::events::reserve_done &) noexcept {
    auto * self = static_cast<reserve_callbacks *>(owner);
    self->done_called = true;
    return true;
  }

  static bool on_error(void * owner, const emel::graph::events::reserve_error &) noexcept {
    auto * self = static_cast<reserve_callbacks *>(owner);
    self->error_called = true;
    return true;
  }
};

struct compute_callbacks {
  bool done_called = false;
  bool error_called = false;
  int32_t error_code = 0;

  static bool on_done(void * owner, const emel::graph::events::compute_done &) noexcept {
    auto * self = static_cast<compute_callbacks *>(owner);
    self->done_called = true;
    return true;
  }

  static bool on_error(void * owner, const emel::graph::events::compute_error & ev) noexcept {
    auto * self = static_cast<compute_callbacks *>(owner);
    self->error_called = true;
    self->error_code = ev.err;
    return true;
  }
};

struct lifecycle_fixture {
  int32_t leaf_tensor = 11;
  int32_t compute_tensor = 29;
  std::array<emel::graph::processor::event::lifecycle_tensor_binding, 2> tensors{{
      {
          .tensor_id = 0,
          .buffer = &leaf_tensor,
          .buffer_bytes = sizeof(leaf_tensor),
          .consumer_refs = 0,
          .is_leaf = true,
      },
      {
          .tensor_id = 1,
          .buffer = &compute_tensor,
          .buffer_bytes = sizeof(compute_tensor),
          .consumer_refs = 1,
          .is_leaf = false,
      },
  }};
  std::array<int32_t, 1> required_ids = {0};
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

struct graph_lane_fixture {
  emel::graph::sm graph{};
  lifecycle_fixture lifecycle{};
  reserve_callbacks reserve_cb{};
  compute_callbacks compute_cb{};
  emel::graph::event::reserve_output reserve_output{};
  emel::graph::event::compute_output compute_output{};
  emel::graph::event::compute compute_request{};
  int32_t kernel_calls = 0;
  bool lane_accepted = false;
  int32_t mutable_execution_owner = 0;

  void reserve_graph() {
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
    REQUIRE(graph.process_event(reserve_request));
    REQUIRE(reserve_cb.done_called);
    REQUIRE_FALSE(reserve_cb.error_called);
  }

  void bind_compute(emel::graph::event::run_kernel_fn kernel_fn = run_kernel_counting,
                    void * compute_ctx = nullptr) {
    compute_request = emel::graph::event::compute{
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
        .compute_ctx = compute_ctx == nullptr ? static_cast<void *>(&kernel_calls)
                                              : compute_ctx,
        .validate = validate_ok,
        .prepare_graph = prepare_graph_reuse,
        .alloc_graph = alloc_graph_ok,
        .bind_inputs = bind_inputs_ok,
        .run_kernel = kernel_fn,
        .extract_outputs = extract_outputs_ok,
        .dispatch_done = {&compute_cb, compute_callbacks::on_done},
        .dispatch_error = {&compute_cb, compute_callbacks::on_error},
    };
  }
};

wavefront::event::compatibility_key make_key(
    const void * model_identity,
    const void * backend_identity,
    const wavefront::event::kernel_route route =
        wavefront::event::kernel_route::q8_k,
    const wavefront::event::output_contract output =
        wavefront::event::output_contract::preselected_argmax) {
  return wavefront::event::compatibility_key{
      .model_identity = model_identity,
      .backend_identity = backend_identity,
      .kernel_kind = emel::kernel::kernel_kind::x86_64,
      .attention = emel::text::generator::attention_mode::flash,
      .route = route,
      .output = output,
      .dtype_layout_contract = static_cast<uint32_t>(route),
      .quantized_contract = static_cast<uint32_t>(route),
      .step_size = 1,
      .token_count = 1,
  };
}

void prepare_lane(graph_lane_fixture & fixture,
                  emel::graph::event::run_kernel_fn kernel_fn = run_kernel_counting,
                  void * compute_ctx = nullptr) {
  fixture.reserve_graph();
  fixture.bind_compute(kernel_fn, compute_ctx);
}

template <class predicate>
bool eventually(predicate && pred) {
  for (int32_t attempt = 0; attempt < 100000; ++attempt) {
    if (pred()) {
      return true;
    }
    std::this_thread::yield();
  }
  return false;
}

template <class prepare_alias>
void check_shared_mutable_alias_serialized(prepare_alias && prepare) {
  int model_tag = 1;
  int backend_tag = 2;
  std::array<graph_lane_fixture, 2> fixtures{};
  prepare(fixtures);

  const auto key = make_key(&model_tag, &backend_tag);
  std::array<wavefront::event::lane, 2> lanes{{
      {fixtures[0].graph, fixtures[0].compute_request, key,
       fixtures[0].lane_accepted, &fixtures[0].mutable_execution_owner},
      {fixtures[1].graph, fixtures[1].compute_request, key,
       fixtures[1].lane_accepted, &fixtures[1].mutable_execution_owner},
  }};
  wavefront::event::dispatch_summary summary{};
  wavefront::event::run request{std::span<wavefront::event::lane>{lanes},
                                summary};
  wavefront::action::worker_pool pool{};
  wavefront::sm machine{pool};

  CHECK(machine.process_event(request));
  CHECK(machine.is(stateforward::sml::state<wavefront::state_idle>));
  CHECK_FALSE(summary.all_submitted);
  CHECK_FALSE(summary.joined);
  CHECK(summary.dispatched_lanes == 2);
  CHECK(fixtures[0].lane_accepted);
  CHECK(fixtures[1].lane_accepted);
}

}  // namespace

TEST_CASE("decode wavefront dispatches one lane inline without grouping") {
  int model_tag = 1;
  int backend_tag = 2;
  graph_lane_fixture fixture{};
  prepare_lane(fixture);

  const auto key = make_key(&model_tag, &backend_tag);
  wavefront::event::lane lane{fixture.graph, fixture.compute_request, key,
                              fixture.lane_accepted,
                              &fixture.mutable_execution_owner};
  wavefront::event::dispatch_summary summary{};
  wavefront::event::run request{std::span<wavefront::event::lane>{&lane, 1u},
                                summary};
  wavefront::sm machine{};

  CHECK(machine.process_event(request));
  CHECK(machine.is(stateforward::sml::state<wavefront::state_idle>));
  CHECK_FALSE(summary.grouped);
  CHECK(summary.dispatched_lanes == 1);
  CHECK(summary.failed_lane == wavefront::event::k_no_failed_lane);
  CHECK(fixture.lane_accepted);
  CHECK(fixture.compute_cb.done_called);
  CHECK_FALSE(fixture.compute_cb.error_called);
  CHECK(fixture.compute_output.reused_topology == 1u);
  CHECK(fixture.compute_output.node_count == fixture.reserve_output.node_count);
  CHECK(fixture.kernel_calls == 1);
}

TEST_CASE("decode wavefront groups compatible lanes with bounded explicit stages") {
  int model_tag = 1;
  int backend_tag = 2;
  std::array<graph_lane_fixture, 4> fixtures{};
  for (auto & fixture : fixtures) {
    prepare_lane(fixture);
  }

  const auto key = make_key(&model_tag, &backend_tag);
  std::array<wavefront::event::lane, 4> lanes{{
      {fixtures[0].graph, fixtures[0].compute_request, key,
       fixtures[0].lane_accepted, &fixtures[0].mutable_execution_owner},
      {fixtures[1].graph, fixtures[1].compute_request, key,
       fixtures[1].lane_accepted, &fixtures[1].mutable_execution_owner},
      {fixtures[2].graph, fixtures[2].compute_request, key,
       fixtures[2].lane_accepted, &fixtures[2].mutable_execution_owner},
      {fixtures[3].graph, fixtures[3].compute_request, key,
       fixtures[3].lane_accepted, &fixtures[3].mutable_execution_owner},
  }};
  wavefront::event::dispatch_summary summary{};
  wavefront::event::run request{std::span<wavefront::event::lane>{lanes},
                                summary};
  wavefront::sm machine{};

  CHECK(machine.process_event(request));
  CHECK(machine.is(stateforward::sml::state<wavefront::state_idle>));
  CHECK(summary.grouped);
  CHECK(summary.dispatched_lanes == 4);
  CHECK(summary.failed_lane == wavefront::event::k_no_failed_lane);
  for (const auto & fixture : fixtures) {
    CHECK(fixture.lane_accepted);
    CHECK(fixture.compute_cb.done_called);
    CHECK_FALSE(fixture.compute_cb.error_called);
    CHECK(fixture.compute_output.reused_topology == 1u);
    CHECK(fixture.compute_output.node_count == fixture.reserve_output.node_count);
    CHECK(fixture.kernel_calls == 1);
  }
}

TEST_CASE("decode wavefront routes duplicate graph actors through serial path") {
  int model_tag = 1;
  int backend_tag = 2;
  std::array<graph_lane_fixture, 2> fixtures{};
  prepare_lane(fixtures[0]);

  const auto key = make_key(&model_tag, &backend_tag);
  std::array<wavefront::event::lane, 2> lanes{{
      {fixtures[0].graph, fixtures[0].compute_request, key,
       fixtures[0].lane_accepted, &fixtures[0].mutable_execution_owner},
      {fixtures[0].graph, fixtures[0].compute_request, key,
       fixtures[1].lane_accepted, &fixtures[1].mutable_execution_owner},
  }};
  wavefront::event::dispatch_summary summary{};
  wavefront::event::run request{std::span<wavefront::event::lane>{lanes},
                                summary};
  wavefront::action::worker_pool pool{};
  wavefront::sm machine{pool};

  CHECK(machine.process_event(request));
  CHECK(machine.is(stateforward::sml::state<wavefront::state_idle>));
  CHECK(summary.grouped);
  CHECK(summary.dispatched_lanes == 2);
  CHECK(summary.failed_lane == wavefront::event::k_no_failed_lane);
  CHECK(fixtures[0].lane_accepted);
  CHECK(fixtures[1].lane_accepted);
  CHECK(fixtures[0].kernel_calls == 2);
}

TEST_CASE("decode wavefront routes lanes sharing an outcome slot through serial path") {
  int model_tag = 1;
  int backend_tag = 2;
  std::array<graph_lane_fixture, 2> fixtures{};
  prepare_lane(fixtures[0]);
  prepare_lane(fixtures[1], run_kernel_rejected);

  const auto key = make_key(&model_tag, &backend_tag);
  // Distinct graph actors but ONE caller-owned outcome slot: parallel workers
  // would race on the shared bool and misreport the failed lane (or hide it),
  // so the machine must select the serial path, which routes each lane's
  // outcome before the next lane writes.
  bool shared_accepted = false;
  std::array<wavefront::event::lane, 2> lanes{{
      {fixtures[0].graph, fixtures[0].compute_request, key, shared_accepted,
       &fixtures[0].mutable_execution_owner},
      {fixtures[1].graph, fixtures[1].compute_request, key, shared_accepted,
       &fixtures[1].mutable_execution_owner},
  }};
  wavefront::event::dispatch_summary summary{};
  wavefront::event::run request{std::span<wavefront::event::lane>{lanes},
                                summary};
  wavefront::action::worker_pool pool{};
  wavefront::sm machine{pool};

  CHECK(machine.process_event(request));
  CHECK(machine.is(stateforward::sml::state<wavefront::state_idle>));
  CHECK(summary.err == emel::error::cast(wavefront::error::lane_rejected));
  CHECK(summary.grouped);
  CHECK(summary.dispatched_lanes == 2);
  CHECK(summary.failed_lane == 1);
  CHECK(fixtures[0].kernel_calls == 1);
  CHECK(fixtures[1].kernel_calls == 1);
}

TEST_CASE("decode wavefront serializes shared mutable compute payloads") {
  SUBCASE("output slot") {
    parallel_lane_context shared{};
    std::array<parallel_lane_owner, 2> owners{{{&shared}, {&shared}}};
    check_shared_mutable_alias_serialized([&](auto & fixtures) {
      prepare_lane(fixtures[0], run_kernel_synchronized, &owners[0]);
      prepare_lane(fixtures[1], run_kernel_synchronized, &owners[1]);
      fixtures[1].compute_request.output_out = &fixtures[0].compute_output;
    });
    CHECK_FALSE(shared.overlapped.load(std::memory_order_acquire));
  }

  SUBCASE("compute context") {
    int32_t shared_kernel_calls = 0;
    check_shared_mutable_alias_serialized([&](auto & fixtures) {
      prepare_lane(fixtures[0], run_kernel_counting, &shared_kernel_calls);
      prepare_lane(fixtures[1], run_kernel_counting, &shared_kernel_calls);
    });
    CHECK(shared_kernel_calls == 2);
  }

  SUBCASE("distinct compute wrappers sharing a mutable execution owner") {
    parallel_lane_context shared{};
    std::array<parallel_lane_owner, 2> owners{{{&shared}, {&shared}}};
    std::array<graph_lane_fixture, 2> fixtures{};
    prepare_lane(fixtures[0], run_kernel_synchronized, &owners[0]);
    prepare_lane(fixtures[1], run_kernel_synchronized, &owners[1]);

    int model_tag = 1;
    int backend_tag = 2;
    const auto key = make_key(&model_tag, &backend_tag);
    std::array<wavefront::event::lane, 2> lanes{{
        {fixtures[0].graph, fixtures[0].compute_request, key,
         fixtures[0].lane_accepted, &shared},
        {fixtures[1].graph, fixtures[1].compute_request, key,
         fixtures[1].lane_accepted, &shared},
    }};
    wavefront::event::dispatch_summary summary{};
    wavefront::event::run request{std::span<wavefront::event::lane>{lanes},
                                  summary};
    wavefront::action::worker_pool pool{};
    wavefront::sm machine{pool};

    CHECK(machine.process_event(request));
    CHECK_FALSE(summary.all_submitted);
    CHECK_FALSE(summary.joined);
    CHECK(summary.dispatched_lanes == 2);
    CHECK_FALSE(shared.overlapped.load(std::memory_order_acquire));
  }

  SUBCASE("missing mutable execution owner") {
    int model_tag = 1;
    int backend_tag = 2;
    std::array<graph_lane_fixture, 2> fixtures{};
    prepare_lane(fixtures[0]);
    prepare_lane(fixtures[1]);

    const auto key = make_key(&model_tag, &backend_tag);
    std::array<wavefront::event::lane, 2> lanes{{
        {fixtures[0].graph, fixtures[0].compute_request, key,
         fixtures[0].lane_accepted, nullptr},
        {fixtures[1].graph, fixtures[1].compute_request, key,
         fixtures[1].lane_accepted, &fixtures[1].mutable_execution_owner},
    }};
    wavefront::event::dispatch_summary summary{};
    wavefront::event::run request{std::span<wavefront::event::lane>{lanes},
                                  summary};
    wavefront::action::worker_pool pool{};
    wavefront::sm machine{pool};

    CHECK(machine.process_event(request));
    CHECK_FALSE(summary.all_submitted);
    CHECK_FALSE(summary.joined);
    CHECK(summary.dispatched_lanes == 2);
  }

  SUBCASE("callback owner") {
    compute_callbacks callbacks{};
    check_shared_mutable_alias_serialized([&](auto & fixtures) {
      prepare_lane(fixtures[0]);
      prepare_lane(fixtures[1]);
      for (auto & fixture : fixtures) {
        fixture.compute_request.dispatch_done = {
            &callbacks, compute_callbacks::on_done};
        fixture.compute_request.dispatch_error = {
            &callbacks, compute_callbacks::on_error};
      }
    });
    CHECK(callbacks.done_called);
    CHECK_FALSE(callbacks.error_called);
  }

  SUBCASE("memory owner") {
    int memory_owner = 0;
    check_shared_mutable_alias_serialized([&](auto & fixtures) {
      prepare_lane(fixtures[0]);
      prepare_lane(fixtures[1]);
      fixtures[0].compute_request.memory_sm = &memory_owner;
      fixtures[1].compute_request.memory_sm = &memory_owner;
    });
  }

  SUBCASE("lifecycle buffer") {
    check_shared_mutable_alias_serialized([](auto & fixtures) {
      prepare_lane(fixtures[0]);
      prepare_lane(fixtures[1]);
      fixtures[1].lifecycle.tensors[1].buffer =
          fixtures[0].lifecycle.tensors[1].buffer;
    });
  }
}

TEST_CASE("decode wavefront serializes accepted-to-opaque cross aliases") {
  check_shared_mutable_alias_serialized([](auto & fixtures) {
    prepare_lane(fixtures[0], run_kernel_ok);
    prepare_lane(fixtures[1], run_kernel_ok,
                 static_cast<void *>(&fixtures[0].lane_accepted));
  });
}

TEST_CASE("decode wavefront serializes graph-to-opaque cross aliases") {
  check_shared_mutable_alias_serialized([](auto & fixtures) {
    prepare_lane(fixtures[0]);
    auto * graph_storage = reinterpret_cast<uint8_t *>(&fixtures[0].graph);
    prepare_lane(fixtures[1], run_kernel_ok,
                 static_cast<void *>(graph_storage + 1u));
  });
}

TEST_CASE("decode wavefront serializes partially overlapping mutable ranges") {
  std::array<uint8_t, 12> shared{};
  check_shared_mutable_alias_serialized([&](auto & fixtures) {
    prepare_lane(fixtures[0]);
    prepare_lane(fixtures[1]);
    fixtures[0].lifecycle.tensors[1].buffer = shared.data();
    fixtures[0].lifecycle.tensors[1].buffer_bytes = 8u;
    fixtures[1].lifecycle.tensors[1].buffer = shared.data() + 4u;
    fixtures[1].lifecycle.tensors[1].buffer_bytes = 8u;
  });
}

TEST_CASE("decode wavefront serializes manifests beyond mutable range contract") {
  using binding = emel::graph::processor::event::lifecycle_tensor_binding;
  std::array<std::array<int32_t, 5>, 2> buffers{};
  std::array<std::array<binding, 5>, 2> tensors{};

  check_shared_mutable_alias_serialized([&](auto & fixtures) {
    for (size_t lane_index = 0u; lane_index < fixtures.size(); ++lane_index) {
      for (size_t tensor_index = 0u; tensor_index < tensors[lane_index].size();
           ++tensor_index) {
        tensors[lane_index][tensor_index] = binding{
            .tensor_id = static_cast<int32_t>(tensor_index),
            .buffer = &buffers[lane_index][tensor_index],
            .buffer_bytes = sizeof(buffers[lane_index][tensor_index]),
            .consumer_refs = tensor_index == 0u ? 0 : 1,
            .is_leaf = tensor_index == 0u,
        };
      }
      fixtures[lane_index].lifecycle.reserve.tensors = tensors[lane_index].data();
      fixtures[lane_index].lifecycle.reserve.tensor_count =
          static_cast<int32_t>(tensors[lane_index].size());
      fixtures[lane_index].lifecycle.compute.tensors = tensors[lane_index].data();
      fixtures[lane_index].lifecycle.compute.tensor_count =
          static_cast<int32_t>(tensors[lane_index].size());
      prepare_lane(fixtures[lane_index]);
    }
  });
}

TEST_CASE("decode wavefront parallel dispatch permits shared leaf model ranges") {
  int model_tag = 1;
  int backend_tag = 2;
  std::array<graph_lane_fixture, 2> fixtures{};
  parallel_lane_context kernel_ctx{};
  std::array<parallel_lane_owner, 2> owners{{{&kernel_ctx}, {&kernel_ctx}}};
  prepare_lane(fixtures[0], run_kernel_wait_for_release, &owners[0]);
  prepare_lane(fixtures[1], run_kernel_wait_for_release, &owners[1]);
  fixtures[1].lifecycle.tensors[0].buffer =
      fixtures[0].lifecycle.tensors[0].buffer;
  fixtures[1].lifecycle.tensors[0].buffer_bytes =
      fixtures[0].lifecycle.tensors[0].buffer_bytes;

  const auto key = make_key(&model_tag, &backend_tag);
  std::array<wavefront::event::lane, 2> lanes{{
      {fixtures[0].graph, fixtures[0].compute_request, key,
       fixtures[0].lane_accepted, &fixtures[0].mutable_execution_owner},
      {fixtures[1].graph, fixtures[1].compute_request, key,
       fixtures[1].lane_accepted, &fixtures[1].mutable_execution_owner},
  }};
  wavefront::event::dispatch_summary summary{};
  wavefront::event::run request{std::span<wavefront::event::lane>{lanes},
                                summary};
  wavefront::action::worker_pool pool{};
  wavefront::sm machine{pool};
  std::atomic<bool> dispatch_returned{false};
  bool accepted = false;

  std::thread dispatch_thread{[&]() {
    accepted = machine.process_event(request);
    dispatch_returned.store(true, std::memory_order_release);
  }};

  const bool both_lanes_entered = eventually([&]() {
    return kernel_ctx.entered.load(std::memory_order_acquire) == 2;
  });
  CHECK(both_lanes_entered);
  CHECK_FALSE(dispatch_returned.load(std::memory_order_acquire));

  kernel_ctx.release.store(true, std::memory_order_release);
  dispatch_thread.join();

  CHECK(accepted);
  CHECK(machine.is(stateforward::sml::state<wavefront::state_idle>));
  CHECK(summary.grouped);
  CHECK(summary.all_submitted);
  CHECK(summary.joined);
  CHECK(summary.dispatched_lanes == 2);
  CHECK(summary.failed_lane == wavefront::event::k_no_failed_lane);
  for (const auto & fixture : fixtures) {
    CHECK(fixture.lane_accepted);
    CHECK(fixture.compute_cb.done_called);
    CHECK_FALSE(fixture.compute_cb.error_called);
    CHECK(fixture.compute_output.reused_topology == 1u);
  }
}

TEST_CASE("decode wavefront parallel dispatch routes each supported lane count") {
  int model_tag = 1;
  int backend_tag = 2;
  for (size_t lane_count = 2u; lane_count <= wavefront::event::k_max_lanes;
       ++lane_count) {
    CAPTURE(lane_count);
    std::array<std::unique_ptr<graph_lane_fixture>,
               wavefront::event::k_max_lanes>
        fixtures{};
    for (auto & fixture : fixtures) {
      fixture = std::make_unique<graph_lane_fixture>();
      prepare_lane(*fixture);
    }

    const auto key = make_key(&model_tag, &backend_tag);
    std::array<wavefront::event::lane, wavefront::event::k_max_lanes> lanes{{
        {fixtures[0]->graph, fixtures[0]->compute_request, key,
         fixtures[0]->lane_accepted, &fixtures[0]->mutable_execution_owner},
        {fixtures[1]->graph, fixtures[1]->compute_request, key,
         fixtures[1]->lane_accepted, &fixtures[1]->mutable_execution_owner},
        {fixtures[2]->graph, fixtures[2]->compute_request, key,
         fixtures[2]->lane_accepted, &fixtures[2]->mutable_execution_owner},
        {fixtures[3]->graph, fixtures[3]->compute_request, key,
         fixtures[3]->lane_accepted, &fixtures[3]->mutable_execution_owner},
        {fixtures[4]->graph, fixtures[4]->compute_request, key,
         fixtures[4]->lane_accepted, &fixtures[4]->mutable_execution_owner},
        {fixtures[5]->graph, fixtures[5]->compute_request, key,
         fixtures[5]->lane_accepted, &fixtures[5]->mutable_execution_owner},
        {fixtures[6]->graph, fixtures[6]->compute_request, key,
         fixtures[6]->lane_accepted, &fixtures[6]->mutable_execution_owner},
        {fixtures[7]->graph, fixtures[7]->compute_request, key,
         fixtures[7]->lane_accepted, &fixtures[7]->mutable_execution_owner},
    }};
    wavefront::event::dispatch_summary summary{};
    wavefront::event::run request{
        std::span<wavefront::event::lane>{lanes.data(), lane_count}, summary};
    wavefront::action::worker_pool pool{};
    wavefront::sm machine{pool};

    CHECK(machine.process_event(request));
    CHECK(machine.is(stateforward::sml::state<wavefront::state_idle>));
    CHECK(summary.err == emel::error::cast(wavefront::error::none));
    CHECK(summary.grouped);
    CHECK(summary.all_submitted);
    CHECK(summary.joined);
    CHECK(summary.dispatched_lanes == static_cast<int32_t>(lane_count));
    CHECK(summary.failed_lane == wavefront::event::k_no_failed_lane);
    for (size_t lane_index = 0u; lane_index < lane_count; ++lane_index) {
      CHECK(fixtures[lane_index]->lane_accepted);
    }
  }
}

TEST_CASE("decode wavefront rejects incompatible multi-lane groups before dispatch") {
  int model_tag = 1;
  int backend_tag = 2;
  std::array<graph_lane_fixture, 2> fixtures{};
  for (auto & fixture : fixtures) {
    prepare_lane(fixture);
  }

  const auto first_key = make_key(&model_tag, &backend_tag);
  const auto second_key =
      make_key(&model_tag, &backend_tag, wavefront::event::kernel_route::kernel);
  std::array<wavefront::event::lane, 2> lanes{{
      {fixtures[0].graph, fixtures[0].compute_request, first_key,
       fixtures[0].lane_accepted, &fixtures[0].mutable_execution_owner},
      {fixtures[1].graph, fixtures[1].compute_request, second_key,
       fixtures[1].lane_accepted, &fixtures[1].mutable_execution_owner},
  }};
  wavefront::event::dispatch_summary summary{};
  wavefront::event::run request{std::span<wavefront::event::lane>{lanes},
                                summary};
  wavefront::sm machine{};

  CHECK(machine.process_event(request));
  CHECK(machine.is(stateforward::sml::state<wavefront::state_idle>));
  CHECK(summary.err == emel::error::cast(wavefront::error::incompatible_lanes));
  CHECK_FALSE(summary.grouped);
  CHECK(summary.dispatched_lanes == 0);
  CHECK_FALSE(fixtures[0].lane_accepted);
  CHECK_FALSE(fixtures[1].lane_accepted);
  CHECK(fixtures[0].kernel_calls == 0);
  CHECK(fixtures[1].kernel_calls == 0);
}

TEST_CASE("decode wavefront reports the first rejected lane and stops") {
  int model_tag = 1;
  int backend_tag = 2;
  std::array<graph_lane_fixture, 3> fixtures{};
  prepare_lane(fixtures[0]);
  prepare_lane(fixtures[1], run_kernel_rejected);
  prepare_lane(fixtures[2]);

  const auto key = make_key(&model_tag, &backend_tag);
  std::array<wavefront::event::lane, 3> lanes{{
      {fixtures[0].graph, fixtures[0].compute_request, key,
       fixtures[0].lane_accepted, &fixtures[0].mutable_execution_owner},
      {fixtures[1].graph, fixtures[1].compute_request, key,
       fixtures[1].lane_accepted, &fixtures[1].mutable_execution_owner},
      {fixtures[2].graph, fixtures[2].compute_request, key,
       fixtures[2].lane_accepted, &fixtures[2].mutable_execution_owner},
  }};
  wavefront::event::dispatch_summary summary{};
  wavefront::event::run request{std::span<wavefront::event::lane>{lanes},
                                summary};
  wavefront::sm machine{};

  CHECK(machine.process_event(request));
  CHECK(machine.is(stateforward::sml::state<wavefront::state_idle>));
  CHECK(summary.err == emel::error::cast(wavefront::error::lane_rejected));
  CHECK(summary.grouped);
  CHECK(summary.dispatched_lanes == 2);
  CHECK(summary.failed_lane == 1);
  CHECK(fixtures[0].lane_accepted);
  CHECK_FALSE(fixtures[1].lane_accepted);
  CHECK_FALSE(fixtures[2].lane_accepted);
  CHECK(fixtures[0].kernel_calls == 1);
  CHECK(fixtures[1].kernel_calls == 1);
  CHECK(fixtures[2].kernel_calls == 0);
}

TEST_CASE("decode wavefront parallel dispatch reports first rejected lane after join") {
  int model_tag = 1;
  int backend_tag = 2;
  std::array<graph_lane_fixture, 3> fixtures{};
  prepare_lane(fixtures[0]);
  prepare_lane(fixtures[1], run_kernel_rejected);
  prepare_lane(fixtures[2]);

  const auto key = make_key(&model_tag, &backend_tag);
  std::array<wavefront::event::lane, 3> lanes{{
      {fixtures[0].graph, fixtures[0].compute_request, key,
       fixtures[0].lane_accepted, &fixtures[0].mutable_execution_owner},
      {fixtures[1].graph, fixtures[1].compute_request, key,
       fixtures[1].lane_accepted, &fixtures[1].mutable_execution_owner},
      {fixtures[2].graph, fixtures[2].compute_request, key,
       fixtures[2].lane_accepted, &fixtures[2].mutable_execution_owner},
  }};
  wavefront::event::dispatch_summary summary{};
  wavefront::event::run request{std::span<wavefront::event::lane>{lanes},
                                summary};
  wavefront::action::worker_pool pool{};
  wavefront::sm machine{pool};

  CHECK(machine.process_event(request));
  CHECK(machine.is(stateforward::sml::state<wavefront::state_idle>));
  CHECK(summary.err == emel::error::cast(wavefront::error::lane_rejected));
  CHECK(summary.grouped);
  CHECK(summary.all_submitted);
  CHECK(summary.joined);
  CHECK(summary.dispatched_lanes == 3);
  CHECK(summary.failed_lane == 1);
  CHECK(fixtures[0].lane_accepted);
  CHECK_FALSE(fixtures[1].lane_accepted);
  CHECK(fixtures[2].lane_accepted);
  CHECK(fixtures[0].kernel_calls == 1);
  CHECK(fixtures[1].kernel_calls == 1);
  CHECK(fixtures[2].kernel_calls == 1);
}

TEST_CASE("decode wavefront rejects requests beyond the fixed lane bound") {
  int model_tag = 1;
  int backend_tag = 2;
  emel::graph::sm graph{};
  emel::graph::event::compute compute{};
  std::array<bool, wavefront::event::k_max_lanes + 1u> accepted{};
  const auto key = make_key(&model_tag, &backend_tag);
  std::array<wavefront::event::lane, wavefront::event::k_max_lanes + 1u> lanes{{
      {graph, compute, key, accepted[0], &accepted[0]},
      {graph, compute, key, accepted[1], &accepted[1]},
      {graph, compute, key, accepted[2], &accepted[2]},
      {graph, compute, key, accepted[3], &accepted[3]},
      {graph, compute, key, accepted[4], &accepted[4]},
      {graph, compute, key, accepted[5], &accepted[5]},
      {graph, compute, key, accepted[6], &accepted[6]},
      {graph, compute, key, accepted[7], &accepted[7]},
      {graph, compute, key, accepted[8], &accepted[8]},
  }};
  wavefront::event::dispatch_summary summary{};
  wavefront::event::run request{std::span<wavefront::event::lane>{lanes},
                                summary};
  wavefront::sm machine{};

  CHECK(machine.process_event(request));
  CHECK(machine.is(stateforward::sml::state<wavefront::state_idle>));
  CHECK(summary.err == emel::error::cast(wavefront::error::invalid_request));
  CHECK(summary.dispatched_lanes == 0);
  for (const bool lane_accepted : accepted) {
    CHECK_FALSE(lane_accepted);
  }
}

TEST_CASE("decode wavefront async surface completes within the RTC call") {
  int model_tag = 1;
  int backend_tag = 2;
  graph_lane_fixture fixture{};
  prepare_lane(fixture);

  const auto key = make_key(&model_tag, &backend_tag);
  wavefront::event::lane lane{fixture.graph, fixture.compute_request, key,
                              fixture.lane_accepted,
                              &fixture.mutable_execution_owner};
  wavefront::event::dispatch_summary summary{};
  wavefront::event::run request{std::span<wavefront::event::lane>{&lane, 1u},
                                summary};
  wavefront::sm machine{};

  emel::bool_task task = machine.process_event_async(request);
  CHECK(task.result());
  CHECK(machine.is(stateforward::sml::state<wavefront::state_idle>));
  CHECK(summary.dispatched_lanes == 1);
  CHECK(fixture.lane_accepted);
  CHECK(fixture.compute_cb.done_called);
  CHECK(fixture.kernel_calls == 1);
}

TEST_CASE("decode wavefront async surface leaves invalid-request error in event") {
  wavefront::event::dispatch_summary summary{};
  wavefront::event::run request{std::span<wavefront::event::lane>{}, summary};
  wavefront::sm machine{};

  emel::bool_task task = machine.process_event_async(request);
  CHECK(task.await_ready());
  CHECK(task.result());
  CHECK(machine.is(stateforward::sml::state<wavefront::state_idle>));
  CHECK(summary.err == emel::error::cast(wavefront::error::invalid_request));
  CHECK(summary.dispatched_lanes == 0);
}
