#include <boost/sml.hpp>
#include <cstdint>
#include <type_traits>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/decoder/actions.hpp"
#include "emel/decoder/sm.hpp"

namespace {

struct noop_queue {
  using container_type = void;

  template <class Event>
  void push(const Event &) noexcept {}
};

struct decoder_queue {
  using container_type = void;

  bool error_validate = false;
  bool error_initialize = false;
  bool error_update = false;
  bool error_prepare = false;
  bool retryable_prepare = false;
  bool error_optimize = false;
  bool error_reserve = false;
  bool error_process = false;
  bool rollback_needed = false;
  bool error_rollback = false;
  bool error_finalize = false;

  template <class Event>
  void push(const Event & ev) noexcept {
    using namespace emel::decoder;
    if constexpr (std::is_same_v<Event, event::validate>) {
      if (error_validate && ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    } else if constexpr (std::is_same_v<Event, event::initialize_batch>) {
      if (error_initialize && ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    } else if constexpr (std::is_same_v<Event, event::update_memory>) {
      if (error_update && ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    } else if constexpr (std::is_same_v<Event, event::prepare_memory_batch>) {
      if (error_prepare && ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
      if (ev.retryable_out != nullptr) {
        *ev.retryable_out = retryable_prepare;
      }
    } else if constexpr (std::is_same_v<Event, event::optimize_memory>) {
      if (error_optimize && ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    } else if constexpr (std::is_same_v<Event, event::reserve_output>) {
      if (error_reserve && ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    } else if constexpr (std::is_same_v<Event, event::process_ubatch>) {
      if (error_process && ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
      if (ev.rollback_needed_out != nullptr) {
        *ev.rollback_needed_out = rollback_needed;
      }
    } else if constexpr (std::is_same_v<Event, event::rollback_ubatch>) {
      if (error_rollback && ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    } else if constexpr (std::is_same_v<Event, event::finalize_outputs>) {
      if (error_finalize && ev.error_out != nullptr) {
        *ev.error_out = EMEL_ERR_BACKEND;
      }
    }
  }
};

template <class Machine>
void drive_to_processing_ubatch(Machine & machine, emel::decoder::event::decode & decode) {
  CHECK(machine.process_event(emel::decoder::events::validate_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::initialize_batch_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::update_memory_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::prepare_memory_batch_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::reserve_output_done{.request = &decode}));
}

TEST_CASE("decoder_sm_rejects_invalid_decode_request") {
  emel::decoder::sm machine{};
  int32_t err = EMEL_OK;

  machine.process_event(emel::decoder::event::decode{
    .error_out = &err,
  });
  CHECK(err != EMEL_OK);
}

TEST_CASE("decoder_sm_testing_policy_drives_transitions") {
  emel::decoder::action::context ctx{};
  ctx.ubatches_total = 1;
  ctx.ubatches_processed = 1;
  ctx.outputs_total = 1;
  ctx.outputs_processed = 1;

  noop_queue queue{};
  emel::decoder::Process process{queue};
  boost::sml::sm<
    emel::decoder::model,
    boost::sml::testing,
    emel::decoder::Process> machine{ctx, process};

  std::array<int32_t, 1> tokens = {{1}};
  int32_t err = EMEL_OK;
  emel::decoder::event::decode decode{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 1,
    .error_out = &err,
  };

  CHECK(machine.process_event(decode));
  CHECK(machine.is(boost::sml::state<emel::decoder::validating_request>));
  CHECK(machine.process_event(emel::decoder::events::validate_done{.request = &decode}));
  CHECK(machine.is(boost::sml::state<emel::decoder::initializing_batch>));
  CHECK(machine.process_event(emel::decoder::events::initialize_batch_done{.request = &decode}));
  CHECK(machine.is(boost::sml::state<emel::decoder::updating_memory_pre>));
  CHECK(machine.process_event(emel::decoder::events::update_memory_done{.request = &decode}));
  CHECK(machine.is(boost::sml::state<emel::decoder::preparing_memory_batch_initial>));
  CHECK(machine.process_event(emel::decoder::events::prepare_memory_batch_done{.request = &decode}));
  CHECK(machine.is(boost::sml::state<emel::decoder::reserving_output>));
  CHECK(machine.process_event(emel::decoder::events::reserve_output_done{.request = &decode}));
  CHECK(machine.is(boost::sml::state<emel::decoder::processing_ubatch>));
  CHECK(machine.process_event(emel::decoder::events::ubatch_done{.request = &decode}));
  CHECK(machine.is(boost::sml::state<emel::decoder::finalizing_outputs>));
  CHECK(machine.process_event(emel::decoder::events::finalize_outputs_done{.request = &decode}));
  CHECK(machine.is(boost::sml::state<emel::decoder::done>));
  CHECK(machine.process_event(emel::decoder::events::decoding_done{.request = &decode}));
  CHECK(machine.is(boost::sml::state<emel::decoder::initialized>));
}

TEST_CASE("decoder_sm_testing_policy_handles_error_path") {
  emel::decoder::action::context ctx{};
  noop_queue queue{};
  emel::decoder::Process process{queue};
  boost::sml::sm<
    emel::decoder::model,
    boost::sml::testing,
    emel::decoder::Process> machine{ctx, process};

  std::array<int32_t, 1> tokens = {{1}};
  int32_t err = EMEL_OK;
  emel::decoder::event::decode decode{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 1,
    .error_out = &err,
  };

  CHECK(machine.process_event(decode));
  CHECK(machine.is(boost::sml::state<emel::decoder::validating_request>));
  CHECK(machine.process_event(emel::decoder::events::validate_error{
    .err = EMEL_ERR_INVALID_ARGUMENT,
    .request = &decode,
  }));
  CHECK(machine.is(boost::sml::state<emel::decoder::errored>));
  CHECK(machine.process_event(emel::decoder::events::decoding_error{
    .err = EMEL_ERR_INVALID_ARGUMENT,
    .request = &decode,
  }));
  CHECK(machine.is(boost::sml::state<emel::decoder::initialized>));
}

TEST_CASE("decoder_sm_prepare_memory_batch_invalid") {
  emel::decoder::sm machine{};
  int32_t err = EMEL_OK;

  machine.process_event(emel::decoder::event::prepare_memory_batch{
    .error_out = &err,
  });
  CHECK(err == EMEL_OK);
}

TEST_CASE("decoder_sm_process_ubatch_invalid") {
  emel::decoder::sm machine{};
  int32_t err = EMEL_OK;
  bool rollback_needed = false;

  machine.process_event(emel::decoder::event::process_ubatch{
    .error_out = &err,
    .rollback_needed_out = &rollback_needed,
  });
  CHECK(err == EMEL_OK);
}

TEST_CASE("decoder_sm_finalize_outputs_invalid") {
  emel::decoder::sm machine{};
  int32_t err = EMEL_OK;

  machine.process_event(emel::decoder::event::finalize_outputs{
    .error_out = &err,
  });
  CHECK(err == EMEL_OK);
}

TEST_CASE("decoder_sm_testing_policy_validate_error_from_queue") {
  emel::decoder::action::context ctx{};
  decoder_queue queue{.error_validate = true};
  emel::decoder::Process process{queue};
  boost::sml::sm<emel::decoder::model, boost::sml::testing, emel::decoder::Process>
    machine{ctx, process};

  std::array<int32_t, 1> tokens = {{1}};
  int32_t err = EMEL_OK;
  emel::decoder::event::decode decode{
    .token_ids = tokens.data(),
    .n_tokens = 1,
    .n_ubatch = 1,
    .error_out = &err,
  };

  CHECK(machine.process_event(decode));
  CHECK(err == EMEL_ERR_BACKEND);
  CHECK(machine.process_event(emel::decoder::events::validate_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
  CHECK(machine.process_event(emel::decoder::events::decoding_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
}

TEST_CASE("decoder_sm_testing_policy_initialize_batch_error_from_queue") {
  emel::decoder::action::context ctx{};
  decoder_queue queue{.error_initialize = true};
  emel::decoder::Process process{queue};
  boost::sml::sm<emel::decoder::model, boost::sml::testing, emel::decoder::Process>
    machine{ctx, process};

  std::array<int32_t, 1> tokens = {{1}};
  emel::decoder::event::decode decode{
    .token_ids = tokens.data(),
    .n_tokens = 1,
    .n_ubatch = 1,
  };

  CHECK(machine.process_event(decode));
  CHECK(machine.process_event(emel::decoder::events::validate_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::initialize_batch_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
  CHECK(machine.process_event(emel::decoder::events::decoding_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
}

TEST_CASE("decoder_sm_testing_policy_update_memory_error_from_queue") {
  emel::decoder::action::context ctx{};
  decoder_queue queue{.error_update = true};
  emel::decoder::Process process{queue};
  boost::sml::sm<emel::decoder::model, boost::sml::testing, emel::decoder::Process>
    machine{ctx, process};

  std::array<int32_t, 1> tokens = {{1}};
  emel::decoder::event::decode decode{
    .token_ids = tokens.data(),
    .n_tokens = 1,
    .n_ubatch = 1,
  };

  CHECK(machine.process_event(decode));
  CHECK(machine.process_event(emel::decoder::events::validate_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::initialize_batch_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::update_memory_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
  CHECK(machine.process_event(emel::decoder::events::decoding_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
}

TEST_CASE("decoder_sm_testing_policy_prepare_retryable_optimize_error") {
  emel::decoder::action::context ctx{};
  decoder_queue queue{
    .error_prepare = true,
    .retryable_prepare = true,
    .error_optimize = true,
  };
  emel::decoder::Process process{queue};
  boost::sml::sm<emel::decoder::model, boost::sml::testing, emel::decoder::Process>
    machine{ctx, process};

  std::array<int32_t, 1> tokens = {{1}};
  emel::decoder::event::decode decode{
    .token_ids = tokens.data(),
    .n_tokens = 1,
    .n_ubatch = 1,
  };

  CHECK(machine.process_event(decode));
  CHECK(machine.process_event(emel::decoder::events::validate_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::initialize_batch_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::update_memory_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::prepare_memory_batch_retryable_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
  CHECK(machine.process_event(emel::decoder::events::optimize_memory_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
  CHECK(machine.process_event(emel::decoder::events::decoding_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
}

TEST_CASE("decoder_sm_testing_policy_prepare_permanent_error") {
  emel::decoder::action::context ctx{};
  decoder_queue queue{.error_prepare = true, .retryable_prepare = false};
  emel::decoder::Process process{queue};
  boost::sml::sm<emel::decoder::model, boost::sml::testing, emel::decoder::Process>
    machine{ctx, process};

  std::array<int32_t, 1> tokens = {{1}};
  emel::decoder::event::decode decode{
    .token_ids = tokens.data(),
    .n_tokens = 1,
    .n_ubatch = 1,
  };

  CHECK(machine.process_event(decode));
  CHECK(machine.process_event(emel::decoder::events::validate_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::initialize_batch_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::update_memory_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::prepare_memory_batch_permanent_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
  CHECK(machine.process_event(emel::decoder::events::decoding_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
}

TEST_CASE("decoder_sm_testing_policy_reserve_output_error") {
  emel::decoder::action::context ctx{};
  decoder_queue queue{.error_reserve = true};
  emel::decoder::Process process{queue};
  boost::sml::sm<emel::decoder::model, boost::sml::testing, emel::decoder::Process>
    machine{ctx, process};

  std::array<int32_t, 1> tokens = {{1}};
  emel::decoder::event::decode decode{
    .token_ids = tokens.data(),
    .n_tokens = 1,
    .n_ubatch = 1,
  };

  CHECK(machine.process_event(decode));
  CHECK(machine.process_event(emel::decoder::events::validate_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::initialize_batch_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::update_memory_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::prepare_memory_batch_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::reserve_output_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
  CHECK(machine.process_event(emel::decoder::events::decoding_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
}

TEST_CASE("decoder_sm_testing_policy_ubatch_error_with_rollback_error") {
  emel::decoder::action::context ctx{};
  decoder_queue queue{
    .error_process = true,
    .rollback_needed = true,
    .error_rollback = true,
  };
  emel::decoder::Process process{queue};
  boost::sml::sm<emel::decoder::model, boost::sml::testing, emel::decoder::Process>
    machine{ctx, process};

  std::array<int32_t, 1> tokens = {{1}};
  emel::decoder::event::decode decode{
    .token_ids = tokens.data(),
    .n_tokens = 1,
    .n_ubatch = 1,
  };

  CHECK(machine.process_event(decode));
  drive_to_processing_ubatch(machine, decode);
  CHECK(machine.process_event(emel::decoder::events::ubatch_error{
    .err = EMEL_ERR_BACKEND,
    .rollback_needed = true,
    .request = &decode,
  }));
  CHECK(machine.process_event(emel::decoder::events::rollback_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
  CHECK(machine.process_event(emel::decoder::events::decoding_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
}

TEST_CASE("decoder_sm_testing_policy_ubatch_error_with_rollback_done") {
  emel::decoder::action::context ctx{};
  decoder_queue queue{
    .error_process = true,
    .rollback_needed = true,
    .error_rollback = false,
  };
  emel::decoder::Process process{queue};
  boost::sml::sm<emel::decoder::model, boost::sml::testing, emel::decoder::Process>
    machine{ctx, process};

  std::array<int32_t, 1> tokens = {{1}};
  emel::decoder::event::decode decode{
    .token_ids = tokens.data(),
    .n_tokens = 1,
    .n_ubatch = 1,
  };

  CHECK(machine.process_event(decode));
  drive_to_processing_ubatch(machine, decode);
  CHECK(machine.process_event(emel::decoder::events::ubatch_error{
    .err = EMEL_ERR_BACKEND,
    .rollback_needed = true,
    .request = &decode,
  }));
  CHECK(machine.process_event(emel::decoder::events::rollback_done{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
  CHECK(machine.process_event(emel::decoder::events::decoding_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
}

TEST_CASE("decoder_sm_testing_policy_finalize_outputs_error") {
  emel::decoder::action::context ctx{};
  decoder_queue queue{.error_finalize = true};
  emel::decoder::Process process{queue};
  boost::sml::sm<emel::decoder::model, boost::sml::testing, emel::decoder::Process>
    machine{ctx, process};

  std::array<int32_t, 1> tokens = {{1}};
  emel::decoder::event::decode decode{
    .token_ids = tokens.data(),
    .n_tokens = 1,
    .n_ubatch = 1,
  };

  CHECK(machine.process_event(decode));
  drive_to_processing_ubatch(machine, decode);
  ctx.ubatches_total = 1;
  ctx.ubatches_processed = 1;
  CHECK(machine.process_event(emel::decoder::events::ubatch_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::finalize_outputs_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
  CHECK(machine.process_event(emel::decoder::events::decoding_error{
    .err = EMEL_ERR_BACKEND,
    .request = &decode,
  }));
}

TEST_CASE("decoder_sm_testing_policy_ubatch_done_guard_paths") {
  emel::decoder::action::context ctx{};
  noop_queue queue{};
  emel::decoder::Process process{queue};
  boost::sml::sm<emel::decoder::model, boost::sml::testing, emel::decoder::Process>
    machine{ctx, process};

  std::array<int32_t, 1> tokens = {{1}};
  emel::decoder::event::decode decode{
    .token_ids = tokens.data(),
    .n_tokens = 1,
    .n_ubatch = 1,
  };

  CHECK(machine.process_event(decode));
  drive_to_processing_ubatch(machine, decode);

  ctx.ubatches_total = 2;
  ctx.ubatches_processed = 0;
  CHECK(machine.process_event(emel::decoder::events::ubatch_done{.request = &decode}));

  ctx.ubatches_processed = 2;
  CHECK(machine.process_event(emel::decoder::events::ubatch_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::finalize_outputs_done{.request = &decode}));
  CHECK(machine.process_event(emel::decoder::events::decoding_done{.request = &decode}));
}

TEST_CASE("decoder_sm_testing_policy_errored_without_request") {
  emel::decoder::action::context ctx{};
  noop_queue queue{};
  emel::decoder::Process process{queue};
  boost::sml::sm<emel::decoder::model, boost::sml::testing, emel::decoder::Process>
    machine{ctx, process};

  std::array<int32_t, 1> tokens = {{1}};
  emel::decoder::event::decode decode{
    .token_ids = tokens.data(),
    .n_tokens = 1,
    .n_ubatch = 1,
  };

  CHECK(machine.process_event(decode));
  CHECK(machine.process_event(emel::decoder::events::validate_error{
    .err = EMEL_ERR_BACKEND,
    .request = nullptr,
  }));
  CHECK(machine.process_event(emel::decoder::events::decoding_error{
    .err = EMEL_ERR_BACKEND,
    .request = nullptr,
  }));
}

TEST_CASE("decoder_sm_retryable_prepare_success_path") {
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
  CHECK(machine.process_event(emel::decoder::events::optimize_memory_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::events::prepare_memory_batch_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::events::reserve_output_done{.request = &request}));

  ctx.ubatches_total = 1;
  ctx.ubatches_processed = 1;
  CHECK(machine.process_event(emel::decoder::events::ubatch_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::events::finalize_outputs_done{.request = &request}));
}

TEST_CASE("decoder_sm_ubatch_done_loops_then_finalizes") {
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
  CHECK(machine.process_event(emel::decoder::events::prepare_memory_batch_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::events::reserve_output_done{.request = &request}));

  ctx.ubatches_total = 2;
  ctx.ubatches_processed = 0;
  CHECK(machine.process_event(emel::decoder::events::ubatch_done{.request = &request}));

  ctx.ubatches_processed = 2;
  CHECK(machine.process_event(emel::decoder::events::ubatch_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::events::finalize_outputs_done{.request = &request}));
}

TEST_CASE("decoder_sm_rollback_done_path") {
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
  CHECK(machine.process_event(emel::decoder::events::prepare_memory_batch_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::events::reserve_output_done{.request = &request}));
  CHECK(machine.process_event(emel::decoder::events::ubatch_error{
    .err = EMEL_ERR_BACKEND,
    .rollback_needed = true,
    .request = &request,
  }));
  CHECK(machine.process_event(emel::decoder::events::rollback_done{
    .err = EMEL_OK,
    .request = &request,
  }));
}

}  // namespace
