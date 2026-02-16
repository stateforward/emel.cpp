#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/decoder/sm.hpp"
#include "emel/emel.h"

namespace {

struct noop_queue {
  using container_type = void;

  template <class Event>
  void push(const Event &) noexcept {}
};

TEST_CASE("decoder_sm_intent_event_coverage_path") {
  emel::decoder::action::context ctx{};
  noop_queue queue{};
  emel::decoder::Process process{queue};
  boost::sml::sm<emel::decoder::model, boost::sml::testing, emel::decoder::Process>
    machine{ctx, process};
  int32_t tokens[1] = {1};
  int32_t err = EMEL_OK;
  bool retryable = false;
  bool rollback_needed = false;

  emel::decoder::event::decode request{
    .token_ids = tokens,
    .n_tokens = 1,
    .n_ubatch = 1,
    .error_out = &err,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::decoder::event::validate{.error_out = &err}));
  CHECK(machine.process_event(emel::decoder::events::validate_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::event::initialize_batch{.error_out = &err}));
  CHECK(machine.process_event(emel::decoder::events::initialize_batch_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::event::update_memory{.error_out = &err}));
  CHECK(machine.process_event(emel::decoder::events::update_memory_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::event::prepare_memory_batch{
    .error_out = &err,
    .retryable_out = &retryable,
  }));
  CHECK(machine.process_event(emel::decoder::events::prepare_memory_batch_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::event::reserve_output{.error_out = &err}));
  CHECK(machine.process_event(emel::decoder::events::reserve_output_done{.request = &request}));

  ctx.ubatches_total = 1;
  ctx.ubatches_processed = 1;
  CHECK(machine.process_event(emel::decoder::event::process_ubatch{
    .error_out = &err,
    .rollback_needed_out = &rollback_needed,
  }));
  CHECK(machine.process_event(emel::decoder::events::ubatch_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::event::finalize_outputs{.error_out = &err}));
  CHECK(machine.process_event(emel::decoder::events::finalize_outputs_done{.request = &request}));

  (void)machine.process_event(emel::decoder::events::decoding_done{
    .outputs = 1,
    .error_out = &err,
    .request = &request,
  });
}

TEST_CASE("decoder_sm_optimize_and_rollback_intents") {
  emel::decoder::action::context ctx{};
  noop_queue queue{};
  emel::decoder::Process process{queue};
  boost::sml::sm<emel::decoder::model, boost::sml::testing, emel::decoder::Process>
    machine{ctx, process};
  int32_t tokens[1] = {1};
  int32_t err = EMEL_OK;

  emel::decoder::event::decode request{
    .token_ids = tokens,
    .n_tokens = 1,
    .n_ubatch = 1,
    .error_out = &err,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::decoder::events::validate_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::events::initialize_batch_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::events::update_memory_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::events::prepare_memory_batch_retryable_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::event::optimize_memory{.error_out = &err}));
  CHECK(machine.process_event(emel::decoder::events::optimize_memory_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::events::prepare_memory_batch_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::events::reserve_output_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::events::ubatch_error{
    .err = EMEL_ERR_BACKEND,
    .rollback_needed = true,
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::event::rollback_ubatch{
    .error_out = &err,
    .rollback_needed = true,
  }));
  CHECK(machine.process_event(emel::decoder::events::rollback_done{
    .err = EMEL_OK,
    .request = &request,
  }));

  (void)machine.process_event(emel::decoder::events::decoding_error{
    .err = EMEL_ERR_BACKEND,
    .error_out = &err,
    .request = &request,
  });
}

}  // namespace
