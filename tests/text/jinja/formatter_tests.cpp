#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "doctest/doctest.h"

#include "emel/text/jinja/formatter/sm.hpp"

namespace {

using emel::text::jinja::event::render;
using emel::text::jinja::formatter::action::context;
using emel::text::jinja::formatter::done;
using emel::text::jinja::formatter::errored;
using emel::text::jinja::formatter::error;
using emel::text::jinja::formatter::initialized;
using emel::text::jinja::formatter::sm;
using emel::text::jinja::formatter::unexpected;
using done_cb = emel::callback<bool(const emel::text::jinja::events::rendering_done &)>;
using error_cb = emel::callback<bool(const emel::text::jinja::events::rendering_error &)>;

bool ignore_done_callback(const emel::text::jinja::events::rendering_done &) {
  return true;
}

bool ignore_error_callback(const emel::text::jinja::events::rendering_error &) {
  return true;
}

constexpr done_cb k_ignore_done_callback = done_cb::from<&ignore_done_callback>();
constexpr error_cb k_ignore_error_callback = error_cb::from<&ignore_error_callback>();

struct callback_tracker {
  bool done_called = false;
  bool error_called = false;
  size_t done_length = 0;
  bool done_truncated = false;
  int32_t error_code = static_cast<int32_t>(error::none);
  size_t error_pos = 0;

  bool on_done(const emel::text::jinja::events::rendering_done & ev) {
    done_called = true;
    done_length = ev.output_length;
    done_truncated = ev.output_truncated;
    return ev.request.output_capacity > 0;
  }

  bool on_error(const emel::text::jinja::events::rendering_error & ev) {
    error_called = true;
    error_code = ev.err;
    error_pos = ev.error_pos;
    return ev.error_pos == 0;
  }
};

}  // namespace

TEST_CASE("jinja_formatter_starts_initialized") {
  context ctx{};
  sm machine{ctx};
  CHECK(machine.is(boost::sml::state<initialized>));
}

TEST_CASE("jinja_formatter_copies_source_text") {
  emel::text::jinja::program program{};
  std::array<char, 64> buffer = {};
  size_t output_length = 0;
  bool output_truncated = true;
  int32_t err = static_cast<int32_t>(error::invalid_request);
  size_t error_pos = 99;

  context ctx{};
  sm machine{ctx};
  render ev{
      program,
      "hello world",
      buffer[0],
      buffer.size(),
      k_ignore_done_callback,
      k_ignore_error_callback,
      nullptr,
      &output_length,
      &output_truncated,
      &err,
      &error_pos,
  };

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<done>));
  CHECK(err == static_cast<int32_t>(error::none));
  CHECK(error_pos == 0);
  CHECK(output_truncated == false);
  CHECK(output_length == ev.source_text.size());
  CHECK(std::string_view(buffer.data(), output_length) == ev.source_text);
}

TEST_CASE("jinja_formatter_handles_empty_source_text") {
  emel::text::jinja::program program{};
  std::array<char, 8> buffer = {};
  size_t output_length = 7;
  bool output_truncated = true;
  int32_t err = static_cast<int32_t>(error::invalid_request);

  context ctx{};
  sm machine{ctx};
  render ev{
      program,
      "",
      buffer[0],
      buffer.size(),
      k_ignore_done_callback,
      k_ignore_error_callback,
      nullptr,
      &output_length,
      &output_truncated,
      &err,
  };

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<done>));
  CHECK(err == static_cast<int32_t>(error::none));
  CHECK(output_length == 0);
  CHECK(output_truncated == false);
}

TEST_CASE("jinja_formatter_reports_capacity_error") {
  emel::text::jinja::program program{};
  std::array<char, 4> buffer = {};
  size_t output_length = 123;
  bool output_truncated = false;
  int32_t err = static_cast<int32_t>(error::none);
  size_t error_pos = 88;

  context ctx{};
  sm machine{ctx};
  render ev{
      program,
      "overflow",
      buffer[0],
      buffer.size(),
      k_ignore_done_callback,
      k_ignore_error_callback,
      nullptr,
      &output_length,
      &output_truncated,
      &err,
      &error_pos,
  };

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<errored>));
  CHECK(err == static_cast<int32_t>(error::invalid_request));
  CHECK(output_length == 0);
  CHECK(output_truncated == true);
  CHECK(error_pos == 0);
}

TEST_CASE("jinja_formatter_rejects_invalid_request") {
  emel::text::jinja::program program{};
  std::array<char, 16> buffer = {};
  size_t output_length = 5;
  bool output_truncated = true;
  int32_t err = static_cast<int32_t>(error::none);
  size_t error_pos = 7;

  context ctx{};
  sm machine{ctx};
  render ev{
      program,
      "ignored",
      buffer[0],
      0,
      k_ignore_done_callback,
      k_ignore_error_callback,
      nullptr,
      &output_length,
      &output_truncated,
      &err,
      &error_pos,
  };

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<errored>));
  CHECK(err == static_cast<int32_t>(error::invalid_request));
  CHECK(output_length == 0);
  CHECK(output_truncated == false);
  CHECK(error_pos == 0);
}

TEST_CASE("jinja_formatter_rejects_missing_callbacks") {
  emel::text::jinja::program program{};
  std::array<char, 16> buffer = {};
  size_t output_length = 3;
  bool output_truncated = true;
  int32_t err = static_cast<int32_t>(error::none);
  size_t error_pos = 9;

  context ctx{};
  sm machine{ctx};
  render ev{
      program,
      "ignored",
      buffer[0],
      buffer.size(),
      done_cb{},
      error_cb{},
      nullptr,
      &output_length,
      &output_truncated,
      &err,
      &error_pos,
  };

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<errored>));
  CHECK(err == static_cast<int32_t>(error::invalid_request));
  CHECK(output_length == 0);
  CHECK(output_truncated == false);
  CHECK(error_pos == 0);
}

TEST_CASE("jinja_formatter_dispatches_done_callback") {
  emel::text::jinja::program program{};
  std::array<char, 16> buffer = {};
  size_t output_length = 0;
  int32_t err = static_cast<int32_t>(error::none);
  callback_tracker tracker{};

  context ctx{};
  sm machine{ctx};
  render ev{
      program,
      "ok",
      buffer[0],
      buffer.size(),
      done_cb::from<callback_tracker, &callback_tracker::on_done>(&tracker),
      error_cb::from<callback_tracker, &callback_tracker::on_error>(&tracker),
      nullptr,
      &output_length,
      nullptr,
      &err,
      nullptr,
  };

  CHECK(machine.process_event(ev));
  CHECK(tracker.done_called);
  CHECK_FALSE(tracker.error_called);
  CHECK(tracker.done_length == output_length);
  CHECK(tracker.done_truncated == false);
}

TEST_CASE("jinja_formatter_dispatches_error_callback") {
  emel::text::jinja::program program{};
  std::array<char, 2> buffer = {};
  size_t output_length = 0;
  int32_t err = static_cast<int32_t>(error::none);
  callback_tracker tracker{};

  context ctx{};
  sm machine{ctx};
  render ev{
      program,
      "bad",
      buffer[0],
      0,
      k_ignore_done_callback,
      error_cb::from<callback_tracker, &callback_tracker::on_error>(&tracker),
      nullptr,
      &output_length,
      nullptr,
      &err,
      nullptr,
  };

  CHECK_FALSE(machine.process_event(ev));
  CHECK_FALSE(tracker.done_called);
  CHECK(tracker.error_called);
  CHECK(tracker.error_code == static_cast<int32_t>(error::invalid_request));
  CHECK(tracker.error_pos == 0);
}

TEST_CASE("jinja_formatter_unexpected_event_transitions_state") {
  struct unknown_event {
    int value = 0;
  };

  context ctx{};
  sm machine{ctx};
  machine.process_event(unknown_event{});

  CHECK(machine.is(boost::sml::state<unexpected>));
}
