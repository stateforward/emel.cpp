#include <boost/sml.hpp>
#include <doctest/doctest.h>
#include <string_view>

#include "emel/emel.h"
#include "emel/text/jinja/ast.hpp"
#include "emel/text/jinja/parser/actions.hpp"
#include "emel/text/jinja/parser/events.hpp"
#include "emel/text/jinja/parser/sm.hpp"
#include "emel/text/jinja/types.hpp"

namespace {

bool dispatch_done_test(void * owner,
                        const emel::text::jinja::events::parsing_done &) {
  *static_cast<bool *>(owner) = true;
  return true;
}

bool dispatch_error_test(void * owner,
                         const emel::text::jinja::events::parsing_error &) {
  *static_cast<bool *>(owner) = true;
  return true;
}

} // namespace

TEST_CASE("jinja_parser_starts_initialized") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::initialized>));
}

TEST_CASE("jinja_parser_valid_parse_reaches_done") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  int32_t error = -1;
  emel::text::jinja::program program{};
  bool done_called = false;
  bool error_called = false;

  ::emel::text::jinja::event::parse ev{
      .template_text = "{{ foo }}",
      .program_out = &program,
      .error_out = &error,
      .owner_sm = &error_called,
      .dispatch_done =
          ::emel::callback<bool(const ::emel::text::jinja::events::parsing_done &)>(
              &done_called, dispatch_done_test),
      .dispatch_error =
          ::emel::callback<bool(const ::emel::text::jinja::events::parsing_error &)>(
              &error_called, dispatch_error_test)};

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::done>));
  CHECK(done_called);
  CHECK_FALSE(error_called);
  CHECK(error == EMEL_OK);
  CHECK(program.body.size() == 1);
  CHECK(dynamic_cast<emel::text::jinja::identifier *>(program.body[0].get()) != nullptr);
}

TEST_CASE("jinja_parser_invalid_parse_reaches_errored") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  int32_t error = EMEL_OK;
  emel::text::jinja::program program{};
  bool error_called = false;

  ::emel::text::jinja::event::parse ev{
      .template_text = "",
      .program_out = &program,
      .error_out = &error,
      .owner_sm = &error_called,
      .dispatch_error =
          ::emel::callback<bool(const ::emel::text::jinja::events::parsing_error &)>(
              &error_called, dispatch_error_test)};

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::errored>));
  CHECK(error_called);
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(program.body.empty());
}

TEST_CASE("jinja_parser_parse_failure_reports_parse_error") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  int32_t error = EMEL_OK;
  emel::text::jinja::program program{};
  bool error_called = false;

  ::emel::text::jinja::event::parse ev{
      .template_text = "{{ }}",
      .program_out = &program,
      .error_out = &error,
      .owner_sm = &error_called,
      .dispatch_error =
          ::emel::callback<bool(const ::emel::text::jinja::events::parsing_error &)>(
              &error_called, dispatch_error_test)};

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::errored>));
  CHECK(error_called);
  CHECK(error == EMEL_ERR_PARSE_FAILED);
  CHECK(program.body.empty());
}

TEST_CASE("jinja_parser_lex_failure_reports_parse_error") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  int32_t error = EMEL_OK;
  emel::text::jinja::program program{};
  bool error_called = false;

  ::emel::text::jinja::event::parse ev{
      .template_text = "{{ \"\\x\" }}",
      .program_out = &program,
      .error_out = &error,
      .owner_sm = &error_called,
      .dispatch_error =
          ::emel::callback<bool(const ::emel::text::jinja::events::parsing_error &)>(
              &error_called, dispatch_error_test)};

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::errored>));
  CHECK(error_called);
  CHECK(error == EMEL_ERR_PARSE_FAILED);
  CHECK(program.body.empty());
}

TEST_CASE("jinja_parser_parses_control_statements") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  int32_t error = EMEL_OK;
  emel::text::jinja::program program{};
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

  ::emel::text::jinja::event::parse ev{
      .template_text = tmpl,
      .program_out = &program,
      .error_out = &error,
      .owner_sm = &done_called,
      .dispatch_done =
          ::emel::callback<bool(const ::emel::text::jinja::events::parsing_done &)>(
              &done_called, dispatch_done_test)};

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::done>));
  CHECK(done_called);
  CHECK(error == EMEL_OK);
  CHECK(program.body.size() >= 7);
}

TEST_CASE("jinja_parser_parses_expressions") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  int32_t error = EMEL_OK;
  emel::text::jinja::program program{};

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

  ::emel::text::jinja::event::parse ev{
      .template_text = tmpl,
      .program_out = &program,
      .error_out = &error,
  };

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::done>));
  CHECK(error == EMEL_OK);
  CHECK(program.body.size() >= 9);
}

TEST_CASE("jinja_parser_parses_additional_expressions") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  int32_t error = EMEL_OK;
  emel::text::jinja::program program{};

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

  ::emel::text::jinja::event::parse ev{
      .template_text = tmpl,
      .program_out = &program,
      .error_out = &error,
  };

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::done>));
  CHECK(error == EMEL_OK);
  CHECK(program.body.size() >= 8);
}

TEST_CASE("jinja_parser_parses_slices_and_loops") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  int32_t error = EMEL_OK;
  emel::text::jinja::program program{};

  const std::string_view tmpl =
      "{% set name = value %}"
      "{% for item in items %}{% break %}{% continue %}{% endfor %}"
      "{{ arr[:] }}{{ arr[1:] }}{{ arr[:2] }}{{ arr[1:2] }}";

  ::emel::text::jinja::event::parse ev{
      .template_text = tmpl,
      .program_out = &program,
      .error_out = &error,
  };

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::done>));
  CHECK(error == EMEL_OK);
  CHECK(program.body.size() >= 5);
}

TEST_CASE("jinja_parser_rejects_unknown_statement") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};
  int32_t error = EMEL_OK;
  emel::text::jinja::program program{};

  ::emel::text::jinja::event::parse ev{
      .template_text = "{% unknown %}",
      .program_out = &program,
      .error_out = &error,
  };

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::text::jinja::parser::errored>));
  CHECK(error == EMEL_ERR_PARSE_FAILED);
  CHECK(program.body.empty());
}

TEST_CASE("jinja_parser_action_rejects_missing_program") {
  emel::text::jinja::parser::action::context ctx{};
  int32_t error = EMEL_OK;
  bool error_called = false;

  ::emel::text::jinja::event::parse ev{
      .template_text = "{{ foo }}",
      .program_out = nullptr,
      .error_out = &error,
      .owner_sm = &error_called,
      .dispatch_error =
          ::emel::callback<bool(const ::emel::text::jinja::events::parsing_error &)>(
              &error_called, dispatch_error_test)};

  emel::text::jinja::parser::action::run_parse(ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(error_called);
}

TEST_CASE("jinja_parser_on_unexpected_sets_backend_error") {
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::program program{};
  int32_t error = EMEL_OK;

  ::emel::text::jinja::event::parse ev{
      .template_text = "{{ foo }}",
      .program_out = &program,
      .error_out = &error,
  };

  emel::text::jinja::parser::action::on_unexpected(ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
}
