#include <boost/sml.hpp>
#include <doctest/doctest.h>
#include <string_view>

#include "emel/emel.h"
#include "emel/jinja/ast.hpp"
#include "emel/jinja/parser/actions.hpp"
#include "emel/jinja/parser/events.hpp"
#include "emel/jinja/parser/sm.hpp"
#include "emel/jinja/types.hpp"

namespace {

bool dispatch_done_test(void * owner,
                        const emel::jinja::events::parsing_done &) {
  *static_cast<bool *>(owner) = true;
  return true;
}

bool dispatch_error_test(void * owner,
                         const emel::jinja::events::parsing_error &) {
  *static_cast<bool *>(owner) = true;
  return true;
}

} // namespace

TEST_CASE("jinja_parser_starts_initialized") {
  emel::jinja::parser::action::context ctx{};
  emel::jinja::parser::sm machine{ctx};
  CHECK(machine.is(boost::sml::state<emel::jinja::parser::Initialized>));
}

TEST_CASE("jinja_parser_valid_parse_reaches_done") {
  emel::jinja::parser::action::context ctx{};
  emel::jinja::parser::sm machine{ctx};
  int32_t error = -1;
  emel::jinja::program program{};
  bool done_called = false;
  bool error_called = false;

  ::emel::jinja::event::parse ev{
      .template_text = "{{ foo }}",
      .program_out = &program,
      .error_out = &error,
      .owner_sm = &error_called,
      .dispatch_done =
          ::emel::callback<bool(const ::emel::jinja::events::parsing_done &)>(
              &done_called, dispatch_done_test),
      .dispatch_error =
          ::emel::callback<bool(const ::emel::jinja::events::parsing_error &)>(
              &error_called, dispatch_error_test)};

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::jinja::parser::Done>));
  CHECK(done_called);
  CHECK_FALSE(error_called);
  CHECK(error == EMEL_OK);
  CHECK(program.body.size() == 1);
  CHECK(dynamic_cast<emel::jinja::identifier *>(program.body[0].get()) != nullptr);
}

TEST_CASE("jinja_parser_invalid_parse_reaches_errored") {
  emel::jinja::parser::action::context ctx{};
  emel::jinja::parser::sm machine{ctx};
  int32_t error = EMEL_OK;
  emel::jinja::program program{};
  bool error_called = false;

  ::emel::jinja::event::parse ev{
      .template_text = "",
      .program_out = &program,
      .error_out = &error,
      .owner_sm = &error_called,
      .dispatch_error =
          ::emel::callback<bool(const ::emel::jinja::events::parsing_error &)>(
              &error_called, dispatch_error_test)};

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::jinja::parser::Errored>));
  CHECK(error_called);
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(program.body.empty());
}

TEST_CASE("jinja_parser_parse_failure_reports_parse_error") {
  emel::jinja::parser::action::context ctx{};
  emel::jinja::parser::sm machine{ctx};
  int32_t error = EMEL_OK;
  emel::jinja::program program{};
  bool error_called = false;

  ::emel::jinja::event::parse ev{
      .template_text = "{{ }}",
      .program_out = &program,
      .error_out = &error,
      .owner_sm = &error_called,
      .dispatch_error =
          ::emel::callback<bool(const ::emel::jinja::events::parsing_error &)>(
              &error_called, dispatch_error_test)};

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::jinja::parser::Errored>));
  CHECK(error_called);
  CHECK(error == EMEL_ERR_PARSE_FAILED);
  CHECK(program.body.empty());
}

TEST_CASE("jinja_parser_lex_failure_reports_parse_error") {
  emel::jinja::parser::action::context ctx{};
  emel::jinja::parser::sm machine{ctx};
  int32_t error = EMEL_OK;
  emel::jinja::program program{};
  bool error_called = false;

  ::emel::jinja::event::parse ev{
      .template_text = "{{ \"\\x\" }}",
      .program_out = &program,
      .error_out = &error,
      .owner_sm = &error_called,
      .dispatch_error =
          ::emel::callback<bool(const ::emel::jinja::events::parsing_error &)>(
              &error_called, dispatch_error_test)};

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::jinja::parser::Errored>));
  CHECK(error_called);
  CHECK(error == EMEL_ERR_PARSE_FAILED);
  CHECK(program.body.empty());
}

TEST_CASE("jinja_parser_parses_control_statements") {
  emel::jinja::parser::action::context ctx{};
  emel::jinja::parser::sm machine{ctx};
  int32_t error = EMEL_OK;
  emel::jinja::program program{};
  bool done_called = false;

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

  ::emel::jinja::event::parse ev{
      .template_text = tmpl,
      .program_out = &program,
      .error_out = &error,
      .owner_sm = &done_called,
      .dispatch_done =
          ::emel::callback<bool(const ::emel::jinja::events::parsing_done &)>(
              &done_called, dispatch_done_test)};

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::jinja::parser::Done>));
  CHECK(done_called);
  CHECK(error == EMEL_OK);
  CHECK(program.body.size() >= 7);
}

TEST_CASE("jinja_parser_parses_expressions") {
  emel::jinja::parser::action::context ctx{};
  emel::jinja::parser::sm machine{ctx};
  int32_t error = EMEL_OK;
  emel::jinja::program program{};

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

  ::emel::jinja::event::parse ev{
      .template_text = tmpl,
      .program_out = &program,
      .error_out = &error,
  };

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::jinja::parser::Done>));
  CHECK(error == EMEL_OK);
  CHECK(program.body.size() >= 9);
}

TEST_CASE("jinja_parser_parses_additional_expressions") {
  emel::jinja::parser::action::context ctx{};
  emel::jinja::parser::sm machine{ctx};
  int32_t error = EMEL_OK;
  emel::jinja::program program{};

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

  ::emel::jinja::event::parse ev{
      .template_text = tmpl,
      .program_out = &program,
      .error_out = &error,
  };

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::jinja::parser::Done>));
  CHECK(error == EMEL_OK);
  CHECK(program.body.size() >= 8);
}

TEST_CASE("jinja_parser_parses_slices_and_loops") {
  emel::jinja::parser::action::context ctx{};
  emel::jinja::parser::sm machine{ctx};
  int32_t error = EMEL_OK;
  emel::jinja::program program{};

  const std::string_view tmpl =
      "{% set name = value %}"
      "{% for item in items %}{% break %}{% continue %}{% endfor %}"
      "{{ arr[:] }}{{ arr[1:] }}{{ arr[:2] }}{{ arr[1:2] }}";

  ::emel::jinja::event::parse ev{
      .template_text = tmpl,
      .program_out = &program,
      .error_out = &error,
  };

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::jinja::parser::Done>));
  CHECK(error == EMEL_OK);
  CHECK(program.body.size() >= 5);
}

TEST_CASE("jinja_parser_rejects_unknown_statement") {
  emel::jinja::parser::action::context ctx{};
  emel::jinja::parser::sm machine{ctx};
  int32_t error = EMEL_OK;
  emel::jinja::program program{};

  ::emel::jinja::event::parse ev{
      .template_text = "{% unknown %}",
      .program_out = &program,
      .error_out = &error,
  };

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::jinja::parser::Errored>));
  CHECK(error == EMEL_ERR_PARSE_FAILED);
  CHECK(program.body.empty());
}

TEST_CASE("jinja_parser_action_rejects_missing_program") {
  emel::jinja::parser::action::context ctx{};
  int32_t error = EMEL_OK;
  bool error_called = false;

  ::emel::jinja::event::parse ev{
      .template_text = "{{ foo }}",
      .program_out = nullptr,
      .error_out = &error,
      .owner_sm = &error_called,
      .dispatch_error =
          ::emel::callback<bool(const ::emel::jinja::events::parsing_error &)>(
              &error_called, dispatch_error_test)};

  emel::jinja::parser::action::run_parse(ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(error_called);
}

TEST_CASE("jinja_parser_on_unexpected_sets_backend_error") {
  emel::jinja::parser::action::context ctx{};
  emel::jinja::program program{};
  int32_t error = EMEL_OK;

  ::emel::jinja::event::parse ev{
      .template_text = "{{ foo }}",
      .program_out = &program,
      .error_out = &error,
  };

  emel::jinja::parser::action::on_unexpected(ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
}
