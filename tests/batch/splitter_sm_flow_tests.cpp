#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/batch/splitter/sm.hpp"
#include "emel/emel.h"

namespace {

struct noop_queue {
  using container_type = void;

  template <class Event>
  void push(const Event &) noexcept {}
};

TEST_CASE("batch_splitter_sm_success_path") {
  emel::batch::splitter::sm machine{};
  int32_t tokens[3] = {1, 2, 3};
  int32_t ubatch_sizes[3] = {0, 0, 0};
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;
  int32_t err = EMEL_OK;

  emel::batch::splitter::event::split request{
    .token_ids = tokens,
    .n_tokens = 3,
    .n_ubatch = 2,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .ubatch_sizes_out = ubatch_sizes,
    .ubatch_sizes_capacity = 3,
    .ubatch_count_out = &ubatch_count,
    .total_outputs_out = &total_outputs,
    .error_out = &err,
  };

  CHECK(machine.process_event(request));
  CHECK(err == EMEL_OK);
  CHECK(ubatch_count > 0);
  CHECK(total_outputs > 0);
}

TEST_CASE("batch_splitter_sm_rejects_invalid_ubatch_buffer") {
  emel::batch::splitter::sm machine{};
  int32_t tokens[1] = {1};
  int32_t err = EMEL_OK;

  emel::batch::splitter::event::split request{
    .token_ids = tokens,
    .n_tokens = 1,
    .n_ubatch = 1,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .ubatch_sizes_out = nullptr,
    .ubatch_sizes_capacity = 1,
    .ubatch_count_out = nullptr,
    .total_outputs_out = nullptr,
    .error_out = &err,
  };

  CHECK_FALSE(machine.process_event(request));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("batch_splitter_sm_reports_validate_error") {
  emel::batch::splitter::sm machine{};
  int32_t ubatch_sizes[1] = {0};
  int32_t err = EMEL_OK;

  emel::batch::splitter::event::split request{
    .token_ids = nullptr,
    .n_tokens = 1,
    .n_ubatch = 1,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .ubatch_sizes_out = ubatch_sizes,
    .ubatch_sizes_capacity = 1,
    .ubatch_count_out = nullptr,
    .total_outputs_out = nullptr,
    .error_out = &err,
  };

  CHECK_FALSE(machine.process_event(request));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("batch_splitter_sm_reports_output_capacity_error") {
  emel::batch::splitter::sm machine{};
  int32_t tokens[2] = {1, 2};
  int32_t ubatch_sizes[1] = {0};
  int32_t ubatch_count = 0;
  int32_t total_outputs = 0;
  int32_t err = EMEL_OK;

  emel::batch::splitter::event::split request{
    .token_ids = tokens,
    .n_tokens = 2,
    .n_ubatch = 1,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .ubatch_sizes_out = ubatch_sizes,
    .ubatch_sizes_capacity = 0,
    .ubatch_count_out = &ubatch_count,
    .total_outputs_out = &total_outputs,
    .error_out = &err,
  };

  CHECK_FALSE(machine.process_event(request));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("batch_splitter_sm_manual_normalize_error_path") {
  emel::batch::splitter::action::context ctx{};
  noop_queue queue{};
  emel::batch::splitter::Process process{queue};
  boost::sml::sm<
    emel::batch::splitter::model,
    boost::sml::testing,
    emel::batch::splitter::Process>
    machine{ctx, process};
  int32_t tokens[1] = {1};
  int32_t err = EMEL_OK;

  emel::batch::splitter::event::split request{
    .token_ids = tokens,
    .n_tokens = 1,
    .n_ubatch = 1,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .ubatch_sizes_out = nullptr,
    .ubatch_sizes_capacity = 0,
    .ubatch_count_out = nullptr,
    .total_outputs_out = nullptr,
    .error_out = &err,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::batch::splitter::events::validate_done{.request = &request}));
  CHECK(machine.process_event(emel::batch::splitter::events::normalize_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  CHECK(machine.process_event(emel::batch::splitter::events::splitting_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
}

TEST_CASE("batch_splitter_sm_manual_split_error_path") {
  emel::batch::splitter::action::context ctx{};
  noop_queue queue{};
  emel::batch::splitter::Process process{queue};
  boost::sml::sm<
    emel::batch::splitter::model,
    boost::sml::testing,
    emel::batch::splitter::Process>
    machine{ctx, process};
  int32_t tokens[1] = {1};
  int32_t err = EMEL_OK;

  emel::batch::splitter::event::split request{
    .token_ids = tokens,
    .n_tokens = 1,
    .n_ubatch = 1,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .ubatch_sizes_out = nullptr,
    .ubatch_sizes_capacity = 0,
    .ubatch_count_out = nullptr,
    .total_outputs_out = nullptr,
    .error_out = &err,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::batch::splitter::events::validate_done{.request = &request}));
  CHECK(machine.process_event(emel::batch::splitter::events::normalize_done{.request = &request}));
  CHECK(machine.process_event(emel::batch::splitter::events::split_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  CHECK(machine.process_event(emel::batch::splitter::events::splitting_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
}

TEST_CASE("batch_splitter_sm_manual_publish_error_path") {
  emel::batch::splitter::action::context ctx{};
  noop_queue queue{};
  emel::batch::splitter::Process process{queue};
  boost::sml::sm<
    emel::batch::splitter::model,
    boost::sml::testing,
    emel::batch::splitter::Process>
    machine{ctx, process};
  int32_t tokens[1] = {1};
  int32_t err = EMEL_OK;

  emel::batch::splitter::event::split request{
    .token_ids = tokens,
    .n_tokens = 1,
    .n_ubatch = 1,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .ubatch_sizes_out = nullptr,
    .ubatch_sizes_capacity = 0,
    .ubatch_count_out = nullptr,
    .total_outputs_out = nullptr,
    .error_out = &err,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::batch::splitter::events::validate_done{.request = &request}));
  CHECK(machine.process_event(emel::batch::splitter::events::normalize_done{.request = &request}));
  CHECK(machine.process_event(emel::batch::splitter::events::split_done{.request = &request}));
  CHECK(machine.process_event(emel::batch::splitter::events::publish_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
  CHECK(machine.process_event(emel::batch::splitter::events::splitting_error{
    .err = EMEL_ERR_BACKEND,
    .request = &request,
  }));
}

TEST_CASE("batch_splitter_sm_manual_publish_done_null_request") {
  emel::batch::splitter::action::context ctx{};
  noop_queue queue{};
  emel::batch::splitter::Process process{queue};
  boost::sml::sm<
    emel::batch::splitter::model,
    boost::sml::testing,
    emel::batch::splitter::Process>
    machine{ctx, process};
  int32_t tokens[1] = {1};
  int32_t err = EMEL_OK;

  emel::batch::splitter::event::split request{
    .token_ids = tokens,
    .n_tokens = 1,
    .n_ubatch = 1,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .ubatch_sizes_out = nullptr,
    .ubatch_sizes_capacity = 0,
    .ubatch_count_out = nullptr,
    .total_outputs_out = nullptr,
    .error_out = &err,
  };

  CHECK(machine.process_event(request));
  CHECK(machine.process_event(emel::batch::splitter::events::validate_done{.request = &request}));
  CHECK(machine.process_event(emel::batch::splitter::events::normalize_done{.request = &request}));
  CHECK(machine.process_event(emel::batch::splitter::events::split_done{.request = &request}));
  CHECK(machine.process_event(emel::batch::splitter::events::publish_done{.request = nullptr}));
  CHECK(machine.process_event(emel::batch::splitter::events::splitting_done{
    .request = nullptr,
    .ubatch_count = 0,
    .total_outputs = 0,
  }));
}

}  // namespace
