#include <array>
#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/batch/splitter/sm.hpp"
#include "emel/callback.hpp"
#include "emel/emel.h"

namespace {

struct split_capture {
  int32_t err = EMEL_OK;
  bool done_called = false;
  bool error_called = false;

  void on_done(const emel::batch::splitter::events::splitting_done &) noexcept {
    done_called = true;
    err = EMEL_OK;
  }

  void on_error(const emel::batch::splitter::events::splitting_error & ev) noexcept {
    error_called = true;
    err = ev.err;
  }
};

inline emel::callback<void(const emel::batch::splitter::events::splitting_done &)> make_done(
    split_capture * capture) {
  return emel::callback<void(const emel::batch::splitter::events::splitting_done &)>::from<
    split_capture,
    &split_capture::on_done>(capture);
}

inline emel::callback<void(const emel::batch::splitter::events::splitting_error &)> make_error(
    split_capture * capture) {
  return emel::callback<void(const emel::batch::splitter::events::splitting_error &)>::from<
    split_capture,
    &split_capture::on_error>(capture);
}

}  // namespace

TEST_CASE("batch_splitter_sm_successful_split") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  split_capture capture{};

  machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 1,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .on_done = make_done(&capture),
    .on_error = make_error(&capture),
  });

  CHECK(capture.done_called);
  CHECK(machine.is(boost::sml::state<emel::batch::splitter::done>));
}

TEST_CASE("batch_splitter_sm_reports_callback_contract_error") {
  emel::batch::splitter::sm machine{};
  std::array<int32_t, 2> tokens = {{1, 2}};
  split_capture capture{};

  machine.process_event(emel::batch::splitter::event::split{
    .token_ids = tokens.data(),
    .n_tokens = static_cast<int32_t>(tokens.size()),
    .n_ubatch = 1,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .on_done = {},
    .on_error = make_error(&capture),
  });

  CHECK(capture.error_called);
  CHECK(capture.err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(machine.is(boost::sml::state<emel::batch::splitter::invalid_request>));
}

TEST_CASE("batch_splitter_sm_validation_error_path") {
  emel::batch::splitter::sm machine{};
  split_capture capture{};

  machine.process_event(emel::batch::splitter::event::split{
    .token_ids = nullptr,
    .n_tokens = 0,
    .n_ubatch = 1,
    .mode = emel::batch::splitter::event::split_mode::simple,
    .on_done = make_done(&capture),
    .on_error = make_error(&capture),
  });

  CHECK(capture.error_called);
  CHECK(capture.err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(machine.is(boost::sml::state<emel::batch::splitter::invalid_request>));
}
