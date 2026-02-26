#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/generator/sm.hpp"

namespace {

bool dispatch_done_test(void * owner, const emel::generator::events::generation_done &) {
  *static_cast<bool *>(owner) = true;
  return true;
}

bool dispatch_error_test(void * owner, const emel::generator::events::generation_error &) {
  *static_cast<bool *>(owner) = true;
  return true;
}

}  // namespace

TEST_CASE("generator_starts_initialized") {
  emel::generator::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::generator::initialized>));
}

TEST_CASE("generator_valid_generate_dispatches_done") {
  emel::generator::sm machine{};
  int32_t error = EMEL_ERR_BACKEND;
  bool done_called = false;
  bool error_called = false;

  emel::generator::event::generate ev{
    .prompt = "hello",
    .max_tokens = 3,
    .error_out = &error,
    .dispatch_done = emel::callback<bool(const emel::generator::events::generation_done &)>(
        &done_called, dispatch_done_test),
    .dispatch_error = emel::callback<bool(const emel::generator::events::generation_error &)>(
        &error_called, dispatch_error_test),
  };

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::generator::initialized>));
  CHECK(done_called);
  CHECK_FALSE(error_called);
  CHECK(error == EMEL_OK);
  CHECK(machine.context_ref().tokens_generated == 3);
  CHECK(machine.context_ref().last_error == EMEL_OK);
}

TEST_CASE("generator_invalid_generate_dispatches_error") {
  emel::generator::sm machine{};
  int32_t error = EMEL_OK;
  bool done_called = false;
  bool error_called = false;

  emel::generator::event::generate ev{
    .prompt = "hello",
    .max_tokens = 0,
    .error_out = &error,
    .dispatch_done = emel::callback<bool(const emel::generator::events::generation_done &)>(
        &done_called, dispatch_done_test),
    .dispatch_error = emel::callback<bool(const emel::generator::events::generation_error &)>(
        &error_called, dispatch_error_test),
  };

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::generator::initialized>));
  CHECK_FALSE(done_called);
  CHECK(error_called);
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(machine.context_ref().last_error == EMEL_ERR_INVALID_ARGUMENT);
}
