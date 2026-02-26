#include <boost/sml.hpp>
#include <cstdint>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/generator/errors.hpp"
#include "emel/generator/sm.hpp"

namespace {

struct callback_tracker {
  bool done_called = false;
  bool error_called = false;
  int32_t tokens_generated = -1;
  emel::error::type err = emel::error::cast(emel::generator::error::none);
};

void dispatch_done_test(void * owner, const emel::generator::events::generation_done & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->done_called = true;
  tracker->tokens_generated = ev.tokens_generated;
}

void dispatch_error_test(void * owner, const emel::generator::events::generation_error & ev) {
  auto * tracker = static_cast<callback_tracker *>(owner);
  tracker->error_called = true;
  tracker->tokens_generated = ev.tokens_generated;
  tracker->err = static_cast<emel::error::type>(ev.err);
}

}  // namespace

TEST_CASE("generator_starts_ready") {
  emel::generator::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::generator::ready>));
}

TEST_CASE("generator_valid_generate_dispatches_done") {
  emel::generator::sm machine{};
  emel::error::type error = emel::error::cast(emel::generator::error::backend);
  callback_tracker tracker{};
  const auto on_done = emel::callback<void(const emel::generator::events::generation_done &)>(
      &tracker, dispatch_done_test);
  const auto on_error = emel::callback<void(const emel::generator::events::generation_error &)>(
      &tracker, dispatch_error_test);

  emel::generator::event::generate ev{
    .prompt = "hello",
    .max_tokens = 3,
    .error_out = &error,
    .on_done = on_done,
    .on_error = on_error,
  };

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::generator::ready>));
  CHECK(tracker.done_called);
  CHECK_FALSE(tracker.error_called);
  CHECK(error == emel::error::cast(emel::generator::error::none));
  CHECK(tracker.tokens_generated == 3);
}

TEST_CASE("generator_invalid_generate_dispatches_error") {
  emel::generator::sm machine{};
  emel::error::type error = emel::error::cast(emel::generator::error::none);
  callback_tracker tracker{};
  const auto on_done = emel::callback<void(const emel::generator::events::generation_done &)>(
      &tracker, dispatch_done_test);
  const auto on_error = emel::callback<void(const emel::generator::events::generation_error &)>(
      &tracker, dispatch_error_test);

  emel::generator::event::generate ev{
    .prompt = "hello",
    .max_tokens = 0,
    .error_out = &error,
    .on_done = on_done,
    .on_error = on_error,
  };

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::generator::ready>));
  CHECK_FALSE(tracker.done_called);
  CHECK(tracker.error_called);
  CHECK(error == emel::error::cast(emel::generator::error::invalid_request));
  CHECK(tracker.err == emel::error::cast(emel::generator::error::invalid_request));
  CHECK(tracker.tokens_generated == 0);
}
