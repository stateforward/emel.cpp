#include "bench_cases.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "emel/error/error.hpp"
#include "emel/graph/processor/context.hpp"
#include "emel/graph/processor/errors.hpp"
#include "emel/graph/processor/events.hpp"
#include "emel/graph/processor/sm.hpp"
#include "emel/graph/tensor/errors.hpp"
#include "emel/graph/tensor/events.hpp"
#include "emel/graph/tensor/sm.hpp"
#include "emel/sm.hpp"

namespace {

using execute_t = emel::graph::processor::event::execute;
namespace processor = emel::graph::processor;
namespace processor_action = emel::graph::processor::action;
using processor_context = processor_action::context;
using processor_error = emel::graph::processor::error;

struct lifecycle_fixture {
  int32_t leaf_tensor = 11;
  int32_t compute_tensor = 29;
  emel::graph::tensor::sm tensor_machine{};
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
  emel::graph::processor::event::lifecycle_manifest manifest{
      .tensors = tensors.data(),
      .tensor_count = static_cast<int32_t>(tensors.size()),
      .phase = &phase,
  };
};

struct dispatch_state {
  bool done_called = false;
  bool error_called = false;
  int32_t output_count = 0;
  int32_t error_code = 0;

  void reset() noexcept {
    done_called = false;
    error_called = false;
    output_count = 0;
    error_code = 0;
  }

  static bool on_done(
      void * owner,
      const emel::graph::processor::events::execution_done & ev) noexcept {
    auto * self = static_cast<dispatch_state *>(owner);
    self->done_called = true;
    self->output_count = ev.output.outputs_produced;
    return true;
  }

  static bool on_error(
      void * owner,
      const emel::graph::processor::events::execution_error & ev) noexcept {
    auto * self = static_cast<dispatch_state *>(owner);
    self->error_called = true;
    self->error_code = ev.err;
    return true;
  }
};

struct baseline_processor_sm : public emel::sm<processor::model, processor_context> {
  using base_type = emel::sm<processor::model, processor_context>;
  using base_type::base_type;

  bool process_event(const emel::graph::processor::event::execute & ev) {
    emel::graph::processor::event::execute_ctx ctx{};
    emel::graph::processor::event::execute_step evt{ev, ctx};
    const bool accepted = base_type::process_event(evt);
    return accepted && ctx.err == emel::error::cast(emel::graph::processor::error::none);
  }
};

template <class machine_type>
struct bench_fixture {
  lifecycle_fixture lifecycle{};
  machine_type machine{};
  dispatch_state dispatch{};
  emel::graph::processor::event::execution_output output{};
  execute_t request = {};
  volatile int32_t sink = 0;
};

[[noreturn]] void bench_abort(const char * message) {
  std::fprintf(stderr, "error: graph processor benchmark setup failed: %s\n", message);
  std::abort();
}

void reserve_lifecycle(lifecycle_fixture & lifecycle) {
  int32_t err =
      static_cast<int32_t>(emel::error::cast(emel::graph::tensor::error::none));
  const bool leaf_ok = lifecycle.tensor_machine.process_event(
      emel::graph::tensor::event::reserve_tensor{
          .tensor_id = lifecycle.tensors[0].tensor_id,
          .buffer = lifecycle.tensors[0].buffer,
          .buffer_bytes = lifecycle.tensors[0].buffer_bytes,
          .consumer_refs = lifecycle.tensors[0].consumer_refs,
          .is_leaf = lifecycle.tensors[0].is_leaf,
          .error_out = &err,
      });
  const bool compute_ok = lifecycle.tensor_machine.process_event(
      emel::graph::tensor::event::reserve_tensor{
          .tensor_id = lifecycle.tensors[1].tensor_id,
          .buffer = lifecycle.tensors[1].buffer,
          .buffer_bytes = lifecycle.tensors[1].buffer_bytes,
          .consumer_refs = lifecycle.tensors[1].consumer_refs,
          .is_leaf = lifecycle.tensors[1].is_leaf,
          .error_out = &err,
      });
  if (!leaf_ok || !compute_ok) {
    bench_abort("tensor reservation failed");
  }
}

bool validate_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool prepare_reused(const execute_t &, bool * reused_out, int32_t * err_out) {
  if (reused_out != nullptr) {
    *reused_out = true;
  }
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool prepare_needs_alloc(const execute_t &, bool * reused_out, int32_t * err_out) {
  if (reused_out != nullptr) {
    *reused_out = false;
  }
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool alloc_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool bind_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool kernel_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool extract_ok(const execute_t &, int32_t * outputs_out, int32_t * err_out) {
  if (outputs_out != nullptr) {
    *outputs_out = 2;
  }
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

execute_t make_execute(emel::graph::processor::event::execution_output & output,
                       dispatch_state & state,
                       lifecycle_fixture & lifecycle,
                       emel::graph::processor::event::prepare_graph_fn prepare_fn) {
  return execute_t{
      .step_plan = reinterpret_cast<const void *>(0xCC11),
      .output_out = &output,
      .lifecycle = &lifecycle.manifest,
      .tensor_machine = &lifecycle.tensor_machine,
      .step_index = 0,
      .step_size = 1,
      .kv_tokens = 1,
      .expected_outputs = 1,
      .positions_count = 0,
      .seq_mask_words = 1,
      .seq_masks_count = 0,
      .seq_primary_ids_count = 0,
      .validate = validate_ok,
      .prepare_graph = prepare_fn,
      .alloc_graph = alloc_ok,
      .bind_inputs = bind_ok,
      .run_kernel = kernel_ok,
      .extract_outputs = extract_ok,
      .dispatch_done = {&state, dispatch_state::on_done},
      .dispatch_error = {&state, dispatch_state::on_error},
  };
}

template <class machine_type>
bench_fixture<machine_type> make_happy_fixture(
    emel::graph::processor::event::prepare_graph_fn prepare_fn) {
  bench_fixture<machine_type> fixture{};
  reserve_lifecycle(fixture.lifecycle);
  fixture.request =
      make_execute(fixture.output, fixture.dispatch, fixture.lifecycle, prepare_fn);
  if (!fixture.machine.process_event(fixture.request) ||
      !fixture.dispatch.done_called ||
      fixture.output.outputs_produced != 2) {
    bench_abort("happy path validation failed");
  }
  fixture.dispatch.reset();
  fixture.output = {};
  return fixture;
}

template <class machine_type>
bench_fixture<machine_type> make_invalid_fixture() {
  bench_fixture<machine_type> fixture{};
  reserve_lifecycle(fixture.lifecycle);
  fixture.request =
      make_execute(fixture.output, fixture.dispatch, fixture.lifecycle, prepare_reused);
  fixture.request.step_size = 0;
  if (fixture.machine.process_event(fixture.request) ||
      !fixture.dispatch.error_called ||
      fixture.dispatch.error_code !=
          static_cast<int32_t>(emel::error::cast(processor_error::invalid_request))) {
    bench_abort("invalid path validation failed");
  }
  fixture.dispatch.reset();
  fixture.output = {};
  return fixture;
}

template <class machine_type>
void append_reused_case(std::vector<emel::bench::result> & results,
                        const emel::bench::config & cfg) {
  auto fixture = make_happy_fixture<machine_type>(prepare_reused);
  auto fn = [&fixture]() {
    fixture.dispatch.reset();
    fixture.output = {};
    const bool ok = fixture.machine.process_event(fixture.request);
    fixture.sink += ok ? fixture.output.outputs_produced : -1;
  };
  results.push_back(emel::bench::measure_case("graph/processor_reused", cfg, fn));
}

template <class machine_type>
void append_alloc_case(std::vector<emel::bench::result> & results,
                       const emel::bench::config & cfg) {
  auto fixture = make_happy_fixture<machine_type>(prepare_needs_alloc);
  auto fn = [&fixture]() {
    fixture.dispatch.reset();
    fixture.output = {};
    const bool ok = fixture.machine.process_event(fixture.request);
    fixture.sink += ok ? fixture.output.outputs_produced : -1;
  };
  results.push_back(emel::bench::measure_case("graph/processor_alloc", cfg, fn));
}

template <class machine_type>
void append_invalid_case(std::vector<emel::bench::result> & results,
                         const emel::bench::config & cfg) {
  auto fixture = make_invalid_fixture<machine_type>();
  auto fn = [&fixture]() {
    fixture.dispatch.reset();
    fixture.output = {};
    const bool ok = fixture.machine.process_event(fixture.request);
    fixture.sink += ok ? 1 : fixture.dispatch.error_code;
  };
  results.push_back(emel::bench::measure_case("graph/processor_invalid", cfg, fn));
}

template <class machine_type>
void append_processor_cases(std::vector<emel::bench::result> & results,
                            const emel::bench::config & cfg) {
  append_reused_case<machine_type>(results, cfg);
  append_alloc_case<machine_type>(results, cfg);
  append_invalid_case<machine_type>(results, cfg);
}

}  // namespace

namespace emel::bench {

void append_emel_graph_processor_cases(std::vector<result> & results,
                                       const config & cfg) {
  append_processor_cases<emel::graph::processor::sm>(results, cfg);
}

void append_reference_graph_processor_cases(std::vector<result> & results,
                                            const config & cfg) {
  append_processor_cases<baseline_processor_sm>(results, cfg);
}

}  // namespace emel::bench
