#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/emel.h"
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

TEST_CASE("jinja_parser_valid_parse_reports_format_unsupported") {
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

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::jinja::parser::Errored>));
  CHECK_FALSE(done_called);
  CHECK(error_called);
  CHECK(error == EMEL_ERR_FORMAT_UNSUPPORTED);
  CHECK(program.node_count == 0);
  CHECK(program.error_count == 0);
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
  CHECK(program.node_count == 0);
  CHECK(program.error_count == 0);
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
