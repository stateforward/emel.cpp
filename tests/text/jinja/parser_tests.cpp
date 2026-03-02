#include <boost/sml.hpp>
#include <doctest/doctest.h>
#include <array>
#include <string_view>

#include "emel/text/jinja/parser/detail.hpp"
#include "emel/text/jinja/parser/errors.hpp"
#include "emel/text/jinja/parser/events.hpp"
#include "emel/text/jinja/parser/sm.hpp"

namespace {

using emel::text::jinja::event::parse;
using emel::text::jinja::events::parsing_done;
using emel::text::jinja::events::parsing_error;
using done_cb = parse::done_callback;
using error_cb = parse::error_callback;

bool ignore_done_callback(const parsing_done &) {
  return true;
}

bool ignore_error_callback(const parsing_error &) {
  return true;
}

constexpr done_cb k_ignore_done_callback = done_cb::from<&ignore_done_callback>();
constexpr error_cb k_ignore_error_callback = error_cb::from<&ignore_error_callback>();

struct callback_tracker {
  bool done_called = false;
  bool error_called = false;
  int32_t err = static_cast<int32_t>(emel::text::jinja::parser::error::none);
  size_t error_pos = 0;

  bool on_done(const parsing_done &) {
    done_called = true;
    return true;
  }

  bool on_error(const parsing_error & ev) {
    error_called = true;
    err = ev.err;
    error_pos = ev.error_pos;
    return true;
  }
};

}  // namespace

TEST_CASE("jinja_parser_starts_initialized") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::initialized>));
}

TEST_CASE("jinja_parser_valid_parse_reaches_done") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  emel::text::jinja::program program{};
  int32_t err = -1;
  size_t error_pos = 999;
  callback_tracker tracker{};

  parse ev{
      "{{ foo }}",
      program,
      done_cb::from<callback_tracker, &callback_tracker::on_done>(&tracker),
      error_cb::from<callback_tracker, &callback_tracker::on_error>(&tracker),
      err,
      error_pos,
  };

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::done>));
  CHECK(tracker.done_called);
  CHECK_FALSE(tracker.error_called);
  CHECK(err == static_cast<int32_t>(emel::text::jinja::parser::error::none));
  CHECK(error_pos == 0);
  CHECK(program.body.size() == 1);
  CHECK(dynamic_cast<emel::text::jinja::identifier *>(program.body[0].get()) != nullptr);
}

TEST_CASE("jinja_parser_invalid_request_with_callbacks_dispatches_error") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  emel::text::jinja::program program{};
  int32_t err = static_cast<int32_t>(emel::text::jinja::parser::error::none);
  size_t error_pos = 999;
  callback_tracker tracker{};

  parse ev{
      "",
      program,
      done_cb::from<callback_tracker, &callback_tracker::on_done>(&tracker),
      error_cb::from<callback_tracker, &callback_tracker::on_error>(&tracker),
      err,
      error_pos,
  };

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::errored>));
  CHECK_FALSE(tracker.done_called);
  CHECK(tracker.error_called);
  CHECK(err == static_cast<int32_t>(emel::text::jinja::parser::error::invalid_request));
  CHECK(error_pos == 0);
  CHECK(program.body.empty());
}

TEST_CASE("jinja_parser_parse_failure_reports_error") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  emel::text::jinja::program program{};
  int32_t err = static_cast<int32_t>(emel::text::jinja::parser::error::none);
  size_t error_pos = 0;
  callback_tracker tracker{};

  parse ev{
      "{{ }}",
      program,
      done_cb::from<callback_tracker, &callback_tracker::on_done>(&tracker),
      error_cb::from<callback_tracker, &callback_tracker::on_error>(&tracker),
      err,
      error_pos,
  };

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::errored>));
  CHECK_FALSE(tracker.done_called);
  CHECK(tracker.error_called);
  CHECK(err == static_cast<int32_t>(emel::text::jinja::parser::error::parse_failed));
  CHECK(error_pos > 0);
  CHECK(program.body.empty());
}

TEST_CASE("jinja_parser_lex_failure_reports_error") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  emel::text::jinja::program program{};
  int32_t err = static_cast<int32_t>(emel::text::jinja::parser::error::none);
  size_t error_pos = 0;
  callback_tracker tracker{};

  parse ev{
      "{{ \"\\x\" }}",
      program,
      done_cb::from<callback_tracker, &callback_tracker::on_done>(&tracker),
      error_cb::from<callback_tracker, &callback_tracker::on_error>(&tracker),
      err,
      error_pos,
  };

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::errored>));
  CHECK_FALSE(tracker.done_called);
  CHECK(tracker.error_called);
  CHECK(err == static_cast<int32_t>(emel::text::jinja::parser::error::parse_failed));
  CHECK(error_pos > 0);
  CHECK(program.body.empty());
}

TEST_CASE("jinja_parser_parses_control_statements") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  emel::text::jinja::program program{};
  int32_t err = static_cast<int32_t>(emel::text::jinja::parser::error::none);
  size_t error_pos = 0;

  const std::string_view tmpl =
      "{% if cond %}{{ value }}{% elif other %}x{% else %}y{% endif %}"
      "{% for item in items %}{{ item }}{% else %}empty{% endfor %}"
      "{% set name %}{{ value }}{% endset %}"
      "{% macro greet(name) %}hi{% endmacro %}"
      "{% call(user) greet(name) %}body{% endcall %}"
      "{% call greet() %}body{% endcall %}"
      "{% filter upper %}x{% endfilter %}"
      "{% filter upper(value) %}x{% endfilter %}"
      "{% generation %}{% endgeneration %}";

  parse ev{
      tmpl,
      program,
      k_ignore_done_callback,
      k_ignore_error_callback,
      err,
      error_pos,
  };

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::done>));
  CHECK(err == static_cast<int32_t>(emel::text::jinja::parser::error::none));
  CHECK(program.body.size() >= 7);
}

TEST_CASE("jinja_parser_parses_expressions") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  emel::text::jinja::program program{};
  int32_t err = static_cast<int32_t>(emel::text::jinja::parser::error::none);
  size_t error_pos = 0;

  const std::string_view tmpl =
      "{{ foo.bar[1:2:3](a=1, *args)|filter }}"
      "{{ value not in items }}"
      "{{ value is not none }}"
      "{{ a if b else c }}"
      "{{ d if e }}"
      "{{ [1,2,3] }}"
      "{{ {'a':1, 'b':2} }}"
      "{{ 'a' 'b' }}"
      "{{ -1 + 2 * 3 }}";

  parse ev{
      tmpl,
      program,
      k_ignore_done_callback,
      k_ignore_error_callback,
      err,
      error_pos,
  };

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::done>));
  CHECK(err == static_cast<int32_t>(emel::text::jinja::parser::error::none));
  CHECK(program.body.size() >= 9);
}

TEST_CASE("jinja_parser_parses_additional_expressions") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  emel::text::jinja::program program{};
  int32_t err = static_cast<int32_t>(emel::text::jinja::parser::error::none);
  size_t error_pos = 0;

  const std::string_view tmpl =
      "{# note #}"
      "{% set a, b = pair %}"
      "{{ not foo or bar and baz }}"
      "{{ value in items }}"
      "{{ left >= right }}"
      "{{ value is even(2) }}"
      "{{ foo | bar(1) }}"
      "{{ foo()() }}"
      "{{ 1.5 }}";

  parse ev{
      tmpl,
      program,
      k_ignore_done_callback,
      k_ignore_error_callback,
      err,
      error_pos,
  };

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::done>));
  CHECK(err == static_cast<int32_t>(emel::text::jinja::parser::error::none));
  CHECK(program.body.size() >= 8);
}

TEST_CASE("jinja_parser_parses_slices_and_loops") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  emel::text::jinja::program program{};
  int32_t err = static_cast<int32_t>(emel::text::jinja::parser::error::none);
  size_t error_pos = 0;

  const std::string_view tmpl =
      "{% set name = value %}"
      "{% for item in items %}{% break %}{% continue %}{% endfor %}"
      "{{ arr[:] }}{{ arr[1:] }}{{ arr[:2] }}{{ arr[1:2] }}";

  parse ev{
      tmpl,
      program,
      k_ignore_done_callback,
      k_ignore_error_callback,
      err,
      error_pos,
  };

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::done>));
  CHECK(err == static_cast<int32_t>(emel::text::jinja::parser::error::none));
  CHECK(program.body.size() >= 5);
}

TEST_CASE("jinja_parser_rejects_unknown_statement") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  emel::text::jinja::program program{};
  int32_t err = static_cast<int32_t>(emel::text::jinja::parser::error::none);
  size_t error_pos = 0;

  parse ev{
      "{% unknown %}",
      program,
      k_ignore_done_callback,
      k_ignore_error_callback,
      err,
      error_pos,
  };

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::errored>));
  CHECK(err == static_cast<int32_t>(emel::text::jinja::parser::error::parse_failed));
  CHECK(program.body.empty());
}

TEST_CASE("jinja_parser_unexpected_event_transitions_state") {
  struct unknown_event {
    int value = 0;
  };

  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  machine.process_event(unknown_event{});

  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::unexpected>));
}
