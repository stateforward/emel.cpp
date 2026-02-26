#include <doctest/doctest.h>

#include <cstdint>

#include "emel/error/error.hpp"
#include "emel/graph/processor/actions.hpp"
#include "emel/graph/processor/alloc_step/actions.hpp"
#include "emel/graph/processor/alloc_step/guards.hpp"
#include "emel/graph/processor/bind_step/actions.hpp"
#include "emel/graph/processor/bind_step/guards.hpp"
#include "emel/graph/processor/errors.hpp"
#include "emel/graph/processor/events.hpp"
#include "emel/graph/processor/extract_step/actions.hpp"
#include "emel/graph/processor/extract_step/guards.hpp"
#include "emel/graph/processor/guards.hpp"
#include "emel/graph/processor/kernel_step/actions.hpp"
#include "emel/graph/processor/kernel_step/guards.hpp"
#include "emel/graph/processor/prepare_step/actions.hpp"
#include "emel/graph/processor/prepare_step/guards.hpp"
#include "emel/graph/processor/validate_step/actions.hpp"
#include "emel/graph/processor/validate_step/guards.hpp"

namespace {

using execute_t = emel::graph::processor::event::execute;
using processor_error = emel::graph::processor::error;

struct dispatch_state {
  bool done_called = false;
  bool error_called = false;
  emel::graph::processor::event::execution_output done_output = {};
  emel::graph::processor::event::execution_output error_output = {};
  int32_t error_code = 0;

  static bool on_done(void * owner,
                      const emel::graph::processor::events::execution_done & ev) noexcept {
    auto * self = static_cast<dispatch_state *>(owner);
    self->done_called = true;
    self->done_output = ev.output;
    return true;
  }

  static bool on_error(void * owner,
                       const emel::graph::processor::events::execution_error & ev) noexcept {
    auto * self = static_cast<dispatch_state *>(owner);
    self->error_called = true;
    self->error_output = ev.output;
    self->error_code = ev.err;
    return true;
  }
};

bool validate_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool validate_fail_with_error(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = static_cast<int32_t>(emel::error::cast(processor_error::kernel_failed));
  }
  return false;
}

bool validate_fail_without_error(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return false;
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

bool prepare_fail_with_error(const execute_t &, bool * reused_out, int32_t * err_out) {
  if (reused_out != nullptr) {
    *reused_out = false;
  }
  if (err_out != nullptr) {
    *err_out = static_cast<int32_t>(emel::error::cast(processor_error::kernel_failed));
  }
  return false;
}

bool prepare_fail_without_error(const execute_t &, bool * reused_out, int32_t * err_out) {
  if (reused_out != nullptr) {
    *reused_out = false;
  }
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return false;
}

bool alloc_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool alloc_fail_with_error(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = static_cast<int32_t>(emel::error::cast(processor_error::kernel_failed));
  }
  return false;
}

bool alloc_fail_without_error(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return false;
}

bool bind_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool bind_fail_with_error(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = static_cast<int32_t>(emel::error::cast(processor_error::kernel_failed));
  }
  return false;
}

bool bind_fail_without_error(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return false;
}

bool kernel_ok(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return true;
}

bool kernel_fail_with_error(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = static_cast<int32_t>(emel::error::cast(processor_error::kernel_failed));
  }
  return false;
}

bool kernel_fail_without_error(const execute_t &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return false;
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

bool extract_fail_with_error(const execute_t &, int32_t * outputs_out, int32_t * err_out) {
  if (outputs_out != nullptr) {
    *outputs_out = 0;
  }
  if (err_out != nullptr) {
    *err_out = static_cast<int32_t>(emel::error::cast(processor_error::kernel_failed));
  }
  return false;
}

bool extract_fail_without_error(const execute_t &, int32_t * outputs_out, int32_t * err_out) {
  if (outputs_out != nullptr) {
    *outputs_out = 0;
  }
  if (err_out != nullptr) {
    *err_out = 0;
  }
  return false;
}

execute_t make_valid_execute(emel::graph::processor::event::execution_output * output,
                             dispatch_state * state) {
  return execute_t{
    .step_plan = reinterpret_cast<const void *>(0xCC11),
    .output_out = output,
    .step_index = 0,
    .step_size = 1,
    .kv_tokens = 1,
    .expected_outputs = 1,
    .positions_count = 0,
    .seq_mask_words = 1,
    .seq_masks_count = 0,
    .seq_primary_ids_count = 0,
    .validate = validate_ok,
    .prepare_graph = prepare_reused,
    .alloc_graph = alloc_ok,
    .bind_inputs = bind_ok,
    .run_kernel = kernel_ok,
    .extract_outputs = extract_ok,
    .dispatch_done = {state, dispatch_state::on_done},
    .dispatch_error = {state, dispatch_state::on_error},
  };
}

}  // namespace

TEST_CASE("graph_processor_action_and_guard_branches") {
  namespace action = emel::graph::processor::action;
  namespace event = emel::graph::processor::event;
  namespace guard = emel::graph::processor::guard;

  action::context machine_ctx{};
  dispatch_state state{};
  event::execution_output output{};
  event::execute request = make_valid_execute(&output, &state);
  event::execute_ctx phase_ctx{};
  event::execute_step ev{request, phase_ctx};

  CHECK(guard::valid_execute{}(ev, machine_ctx));
  CHECK_FALSE(guard::invalid_execute{}(ev, machine_ctx));

  request.step_size = 0;
  CHECK(guard::invalid_execute{}(ev, machine_ctx));
  CHECK(guard::invalid_execute_with_dispatchable_output{}(ev, machine_ctx));
  request.dispatch_error = {};
  CHECK(guard::invalid_execute_with_output_only{}(ev, machine_ctx));
  request.output_out = nullptr;
  CHECK(guard::invalid_execute_without_output{}(ev, machine_ctx));

  request = make_valid_execute(&output, &state);
  action::begin_execute(ev, machine_ctx);
  CHECK(machine_ctx.dispatch_generation == 1u);
  CHECK(ev.ctx.err == emel::error::cast(processor_error::none));
  CHECK(ev.ctx.outputs_produced == 0);
  CHECK(ev.ctx.graph_reused == 0u);

  ev.ctx.outputs_produced = 9;
  ev.ctx.graph_reused = 1u;
  action::commit_output(ev, machine_ctx);
  CHECK(output.outputs_produced == 9);
  CHECK(output.graph_reused == 1u);

  state = {};
  action::dispatch_done(ev, machine_ctx);
  CHECK(state.done_called);
  CHECK_FALSE(state.error_called);
  CHECK(state.done_output.outputs_produced == 9);

  state = {};
  ev.ctx.err = emel::error::cast(processor_error::kernel_failed);
  action::dispatch_error(ev, machine_ctx);
  CHECK_FALSE(state.done_called);
  CHECK(state.error_called);
  CHECK(state.error_code ==
        static_cast<int32_t>(emel::error::cast(processor_error::kernel_failed)));

  state = {};
  output = {.outputs_produced = 3, .graph_reused = 1u};
  action::reject_invalid_execute_with_dispatch(ev, machine_ctx);
  CHECK(state.error_called);
  CHECK(output.outputs_produced == 0);
  CHECK(output.graph_reused == 0u);

  output = {.outputs_produced = 3, .graph_reused = 1u};
  request.dispatch_error = {};
  action::reject_invalid_execute_with_output_only(ev, machine_ctx);
  CHECK(output.outputs_produced == 0);
  CHECK(output.graph_reused == 0u);

  request.output_out = nullptr;
  action::reject_invalid_execute_without_output(ev, machine_ctx);
  CHECK(ev.ctx.err == emel::error::cast(processor_error::invalid_request));

  ev.ctx.err = emel::error::cast(processor_error::none);
  CHECK(guard::phase_ok{}(ev, machine_ctx));
  CHECK_FALSE(guard::phase_failed{}(ev, machine_ctx));
  ev.ctx.err = emel::error::cast(processor_error::kernel_failed);
  CHECK_FALSE(guard::phase_ok{}(ev, machine_ctx));
  CHECK(guard::phase_failed{}(ev, machine_ctx));

  ev.ctx.err = emel::error::cast(processor_error::none);
  ev.ctx.validate_outcome = emel::graph::processor::validate_step::events::phase_outcome::done;
  CHECK(guard::validate_done{}(ev, machine_ctx));
  ev.ctx.validate_outcome = emel::graph::processor::validate_step::events::phase_outcome::failed;
  CHECK(guard::validate_failed{}(ev, machine_ctx));

  ev.ctx.prepare_outcome = emel::graph::processor::prepare_step::events::phase_outcome::done;
  ev.ctx.graph_reused = 1u;
  CHECK(guard::prepare_done_reused{}(ev, machine_ctx));
  ev.ctx.graph_reused = 0u;
  CHECK(guard::prepare_done_needs_allocation{}(ev, machine_ctx));
  ev.ctx.prepare_outcome = emel::graph::processor::prepare_step::events::phase_outcome::failed;
  CHECK(guard::prepare_failed{}(ev, machine_ctx));

  ev.ctx.alloc_outcome = emel::graph::processor::alloc_step::events::phase_outcome::done;
  CHECK(guard::alloc_done{}(ev, machine_ctx));
  ev.ctx.alloc_outcome = emel::graph::processor::alloc_step::events::phase_outcome::failed;
  CHECK(guard::alloc_failed{}(ev, machine_ctx));

  ev.ctx.bind_outcome = emel::graph::processor::bind_step::events::phase_outcome::done;
  CHECK(guard::bind_done{}(ev, machine_ctx));
  ev.ctx.bind_outcome = emel::graph::processor::bind_step::events::phase_outcome::failed;
  CHECK(guard::bind_failed{}(ev, machine_ctx));

  ev.ctx.kernel_outcome = emel::graph::processor::kernel_step::events::phase_outcome::done;
  CHECK(guard::kernel_done{}(ev, machine_ctx));
  ev.ctx.kernel_outcome = emel::graph::processor::kernel_step::events::phase_outcome::failed;
  CHECK(guard::kernel_failed{}(ev, machine_ctx));

  ev.ctx.extract_outcome = emel::graph::processor::extract_step::events::phase_outcome::done;
  CHECK(guard::extract_done{}(ev, machine_ctx));
  ev.ctx.extract_outcome = emel::graph::processor::extract_step::events::phase_outcome::failed;
  CHECK(guard::extract_failed{}(ev, machine_ctx));

  ev.ctx.err = emel::error::cast(processor_error::none);
  action::on_unexpected(ev, machine_ctx);
  CHECK(ev.ctx.err == emel::error::cast(processor_error::internal_error));
}

TEST_CASE("graph_processor_step_action_and_guard_branches") {
  namespace event = emel::graph::processor::event;

  event::execution_output output{};
  dispatch_state dispatch{};
  event::execute request = make_valid_execute(&output, &dispatch);
  event::execute_ctx ctx{};
  event::execute_step ev{request, ctx};

  emel::graph::processor::validate_step::action::context validate_ctx{};
  emel::graph::processor::prepare_step::action::context prepare_ctx{};
  emel::graph::processor::alloc_step::action::context alloc_ctx{};
  emel::graph::processor::bind_step::action::context bind_ctx{};
  emel::graph::processor::kernel_step::action::context kernel_ctx{};
  emel::graph::processor::extract_step::action::context extract_ctx{};

  // validate_step
  ev.ctx.err = emel::error::cast(processor_error::none);
  CHECK(emel::graph::processor::validate_step::guard::phase_request_callback{}(ev, validate_ctx));
  emel::graph::processor::validate_step::action::run_callback(ev, validate_ctx);
  CHECK(ev.ctx.phase_callback_ok);
  CHECK(ev.ctx.phase_callback_err == 0);
  CHECK(emel::graph::processor::validate_step::guard::callback_ok{}(ev, validate_ctx));

  request.validate = validate_fail_with_error;
  emel::graph::processor::validate_step::action::run_callback(ev, validate_ctx);
  CHECK(emel::graph::processor::validate_step::guard::callback_error{}(ev, validate_ctx));

  request.validate = validate_fail_without_error;
  emel::graph::processor::validate_step::action::run_callback(ev, validate_ctx);
  CHECK(emel::graph::processor::validate_step::guard::callback_failed_without_error{}(ev,
                                                                                       validate_ctx));

  request.validate = nullptr;
  CHECK(emel::graph::processor::validate_step::guard::phase_missing_callback{}(ev, validate_ctx));
  ev.ctx.err = emel::error::cast(processor_error::internal_error);
  CHECK(emel::graph::processor::validate_step::guard::phase_prefailed{}(ev, validate_ctx));

  ev.ctx.err = emel::error::cast(processor_error::none);
  emel::graph::processor::validate_step::action::mark_done(ev, validate_ctx);
  emel::graph::processor::validate_step::action::mark_failed_existing_error(ev, validate_ctx);
  ev.ctx.phase_callback_err = static_cast<int32_t>(emel::error::cast(processor_error::kernel_failed));
  emel::graph::processor::validate_step::action::mark_failed_callback_error(ev, validate_ctx);
  ev.ctx.phase_callback_err = 0;
  emel::graph::processor::validate_step::action::mark_failed_callback_without_error(ev, validate_ctx);
  emel::graph::processor::validate_step::action::mark_failed_invalid_request(ev, validate_ctx);
  emel::graph::processor::validate_step::action::on_unexpected(ev, validate_ctx);

  // prepare_step
  request = make_valid_execute(&output, &dispatch);
  ev.ctx.err = emel::error::cast(processor_error::none);
  CHECK(emel::graph::processor::prepare_step::guard::phase_request_callback{}(ev, prepare_ctx));
  emel::graph::processor::prepare_step::action::run_callback(ev, prepare_ctx);
  CHECK(ev.ctx.graph_reused == 1u);
  CHECK(emel::graph::processor::prepare_step::guard::callback_ok{}(ev, prepare_ctx));

  request.prepare_graph = prepare_fail_with_error;
  emel::graph::processor::prepare_step::action::run_callback(ev, prepare_ctx);
  CHECK(emel::graph::processor::prepare_step::guard::callback_error{}(ev, prepare_ctx));

  request.prepare_graph = prepare_fail_without_error;
  emel::graph::processor::prepare_step::action::run_callback(ev, prepare_ctx);
  CHECK(emel::graph::processor::prepare_step::guard::callback_failed_without_error{}(ev,
                                                                                      prepare_ctx));

  request.prepare_graph = nullptr;
  CHECK(emel::graph::processor::prepare_step::guard::phase_missing_callback{}(ev, prepare_ctx));
  ev.ctx.err = emel::error::cast(processor_error::internal_error);
  CHECK(emel::graph::processor::prepare_step::guard::phase_prefailed{}(ev, prepare_ctx));

  ev.ctx.err = emel::error::cast(processor_error::none);
  emel::graph::processor::prepare_step::action::mark_done(ev, prepare_ctx);
  emel::graph::processor::prepare_step::action::mark_failed_existing_error(ev, prepare_ctx);
  ev.ctx.phase_callback_err = static_cast<int32_t>(emel::error::cast(processor_error::kernel_failed));
  emel::graph::processor::prepare_step::action::mark_failed_callback_error(ev, prepare_ctx);
  ev.ctx.phase_callback_err = 0;
  emel::graph::processor::prepare_step::action::mark_failed_callback_without_error(ev, prepare_ctx);
  emel::graph::processor::prepare_step::action::mark_failed_invalid_request(ev, prepare_ctx);
  emel::graph::processor::prepare_step::action::on_unexpected(ev, prepare_ctx);

  // alloc_step
  request = make_valid_execute(&output, &dispatch);
  ev.ctx.err = emel::error::cast(processor_error::none);
  CHECK(emel::graph::processor::alloc_step::guard::phase_request_callback{}(ev, alloc_ctx));
  emel::graph::processor::alloc_step::action::run_callback(ev, alloc_ctx);
  CHECK(emel::graph::processor::alloc_step::guard::callback_ok{}(ev, alloc_ctx));

  request.alloc_graph = alloc_fail_with_error;
  emel::graph::processor::alloc_step::action::run_callback(ev, alloc_ctx);
  CHECK(emel::graph::processor::alloc_step::guard::callback_error{}(ev, alloc_ctx));

  request.alloc_graph = alloc_fail_without_error;
  emel::graph::processor::alloc_step::action::run_callback(ev, alloc_ctx);
  CHECK(emel::graph::processor::alloc_step::guard::callback_failed_without_error{}(ev, alloc_ctx));

  request.alloc_graph = nullptr;
  CHECK(emel::graph::processor::alloc_step::guard::phase_missing_callback{}(ev, alloc_ctx));
  ev.ctx.err = emel::error::cast(processor_error::internal_error);
  CHECK(emel::graph::processor::alloc_step::guard::phase_prefailed{}(ev, alloc_ctx));

  ev.ctx.err = emel::error::cast(processor_error::none);
  emel::graph::processor::alloc_step::action::mark_done(ev, alloc_ctx);
  emel::graph::processor::alloc_step::action::mark_failed_existing_error(ev, alloc_ctx);
  ev.ctx.phase_callback_err = static_cast<int32_t>(emel::error::cast(processor_error::kernel_failed));
  emel::graph::processor::alloc_step::action::mark_failed_callback_error(ev, alloc_ctx);
  ev.ctx.phase_callback_err = 0;
  emel::graph::processor::alloc_step::action::mark_failed_callback_without_error(ev, alloc_ctx);
  emel::graph::processor::alloc_step::action::mark_failed_invalid_request(ev, alloc_ctx);
  emel::graph::processor::alloc_step::action::on_unexpected(ev, alloc_ctx);

  // bind_step
  request = make_valid_execute(&output, &dispatch);
  ev.ctx.err = emel::error::cast(processor_error::none);
  CHECK(emel::graph::processor::bind_step::guard::phase_request_callback{}(ev, bind_ctx));
  emel::graph::processor::bind_step::action::run_callback(ev, bind_ctx);
  CHECK(emel::graph::processor::bind_step::guard::callback_ok{}(ev, bind_ctx));

  request.bind_inputs = bind_fail_with_error;
  emel::graph::processor::bind_step::action::run_callback(ev, bind_ctx);
  CHECK(emel::graph::processor::bind_step::guard::callback_error{}(ev, bind_ctx));

  request.bind_inputs = bind_fail_without_error;
  emel::graph::processor::bind_step::action::run_callback(ev, bind_ctx);
  CHECK(emel::graph::processor::bind_step::guard::callback_failed_without_error{}(ev, bind_ctx));

  request.bind_inputs = nullptr;
  CHECK(emel::graph::processor::bind_step::guard::phase_missing_callback{}(ev, bind_ctx));
  ev.ctx.err = emel::error::cast(processor_error::internal_error);
  CHECK(emel::graph::processor::bind_step::guard::phase_prefailed{}(ev, bind_ctx));

  ev.ctx.err = emel::error::cast(processor_error::none);
  emel::graph::processor::bind_step::action::mark_done(ev, bind_ctx);
  emel::graph::processor::bind_step::action::mark_failed_existing_error(ev, bind_ctx);
  ev.ctx.phase_callback_err = static_cast<int32_t>(emel::error::cast(processor_error::kernel_failed));
  emel::graph::processor::bind_step::action::mark_failed_callback_error(ev, bind_ctx);
  ev.ctx.phase_callback_err = 0;
  emel::graph::processor::bind_step::action::mark_failed_callback_without_error(ev, bind_ctx);
  emel::graph::processor::bind_step::action::mark_failed_invalid_request(ev, bind_ctx);
  emel::graph::processor::bind_step::action::on_unexpected(ev, bind_ctx);

  // kernel_step
  request = make_valid_execute(&output, &dispatch);
  ev.ctx.err = emel::error::cast(processor_error::none);
  CHECK(emel::graph::processor::kernel_step::guard::phase_request_callback{}(ev, kernel_ctx));
  emel::graph::processor::kernel_step::action::run_callback(ev, kernel_ctx);
  CHECK(emel::graph::processor::kernel_step::guard::callback_ok{}(ev, kernel_ctx));

  request.run_kernel = kernel_fail_with_error;
  emel::graph::processor::kernel_step::action::run_callback(ev, kernel_ctx);
  CHECK(emel::graph::processor::kernel_step::guard::callback_error{}(ev, kernel_ctx));

  request.run_kernel = kernel_fail_without_error;
  emel::graph::processor::kernel_step::action::run_callback(ev, kernel_ctx);
  CHECK(emel::graph::processor::kernel_step::guard::callback_failed_without_error{}(ev, kernel_ctx));

  request.run_kernel = nullptr;
  CHECK(emel::graph::processor::kernel_step::guard::phase_missing_callback{}(ev, kernel_ctx));
  ev.ctx.err = emel::error::cast(processor_error::internal_error);
  CHECK(emel::graph::processor::kernel_step::guard::phase_prefailed{}(ev, kernel_ctx));

  ev.ctx.err = emel::error::cast(processor_error::none);
  emel::graph::processor::kernel_step::action::mark_done(ev, kernel_ctx);
  emel::graph::processor::kernel_step::action::mark_failed_existing_error(ev, kernel_ctx);
  ev.ctx.phase_callback_err = static_cast<int32_t>(emel::error::cast(processor_error::kernel_failed));
  emel::graph::processor::kernel_step::action::mark_failed_callback_error(ev, kernel_ctx);
  ev.ctx.phase_callback_err = 0;
  emel::graph::processor::kernel_step::action::mark_failed_callback_without_error(ev, kernel_ctx);
  emel::graph::processor::kernel_step::action::mark_failed_invalid_request(ev, kernel_ctx);
  emel::graph::processor::kernel_step::action::on_unexpected(ev, kernel_ctx);

  // extract_step
  request = make_valid_execute(&output, &dispatch);
  ev.ctx.err = emel::error::cast(processor_error::none);
  CHECK(emel::graph::processor::extract_step::guard::phase_request_callback{}(ev, extract_ctx));
  emel::graph::processor::extract_step::action::run_callback(ev, extract_ctx);
  CHECK(ev.ctx.outputs_produced == 2);
  CHECK(emel::graph::processor::extract_step::guard::callback_ok{}(ev, extract_ctx));

  request.extract_outputs = extract_fail_with_error;
  emel::graph::processor::extract_step::action::run_callback(ev, extract_ctx);
  CHECK(emel::graph::processor::extract_step::guard::callback_error{}(ev, extract_ctx));

  request.extract_outputs = extract_fail_without_error;
  emel::graph::processor::extract_step::action::run_callback(ev, extract_ctx);
  CHECK(
      emel::graph::processor::extract_step::guard::callback_failed_without_error{}(ev, extract_ctx));

  request.extract_outputs = nullptr;
  CHECK(emel::graph::processor::extract_step::guard::phase_missing_callback{}(ev, extract_ctx));
  ev.ctx.err = emel::error::cast(processor_error::internal_error);
  CHECK(emel::graph::processor::extract_step::guard::phase_prefailed{}(ev, extract_ctx));

  ev.ctx.err = emel::error::cast(processor_error::none);
  emel::graph::processor::extract_step::action::mark_done(ev, extract_ctx);
  emel::graph::processor::extract_step::action::mark_failed_existing_error(ev, extract_ctx);
  ev.ctx.phase_callback_err = static_cast<int32_t>(emel::error::cast(processor_error::kernel_failed));
  emel::graph::processor::extract_step::action::mark_failed_callback_error(ev, extract_ctx);
  ev.ctx.phase_callback_err = 0;
  emel::graph::processor::extract_step::action::mark_failed_callback_without_error(ev, extract_ctx);
  emel::graph::processor::extract_step::action::mark_failed_invalid_request(ev, extract_ctx);
  emel::graph::processor::extract_step::action::on_unexpected(ev, extract_ctx);

  // Use callback variants not exercised above.
  request.prepare_graph = prepare_needs_alloc;
  emel::graph::processor::prepare_step::action::run_callback(ev, prepare_ctx);
}
