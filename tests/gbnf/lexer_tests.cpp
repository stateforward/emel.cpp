#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/gbnf/lexer/errors.hpp"
#include "emel/gbnf/lexer/sm.hpp"

namespace {

struct probe {
  bool done_called = false;
  bool error_called = false;
  bool has_token = false;
  emel::gbnf::lexer::event::token token = {};
  emel::gbnf::lexer::cursor next_cursor = {};
  int32_t err = 0;
};

bool on_done(void *owner, const emel::gbnf::lexer::events::next_done &ev) {
  auto *p = static_cast<probe *>(owner);
  p->done_called = true;
  p->has_token = ev.has_token;
  p->token = ev.token;
  p->next_cursor = ev.next_cursor;
  return true;
}

bool on_error(void *owner, const emel::gbnf::lexer::events::next_error &ev) {
  auto *p = static_cast<probe *>(owner);
  p->error_called = true;
  p->err = ev.err;
  return true;
}

int32_t error_code(const emel::gbnf::lexer::error err) {
  return static_cast<int32_t>(emel::error::cast(err));
}

}  // namespace

TEST_CASE("gbnf_lexer_starts_initialized") {
  emel::gbnf::lexer::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::gbnf::lexer::initialized>));
}

TEST_CASE("gbnf_lexer_next_emits_token_stream_then_eof") {
  emel::gbnf::lexer::sm machine{};
  emel::gbnf::lexer::cursor cursor{
      .input = "root ::= \"a\"",
      .offset = 0,
      .token_count = 0,
  };
  probe out{};
  emel::callback<bool(const emel::gbnf::lexer::events::next_done &)> done_cb{&out, on_done};
  emel::callback<bool(const emel::gbnf::lexer::events::next_error &)> error_cb{&out, on_error};

  emel::gbnf::lexer::event::next next_ev{cursor, done_cb, error_cb};

  CHECK(machine.process_event(next_ev));
  CHECK(machine.is(boost::sml::state<emel::gbnf::lexer::scanning>));
  CHECK(out.done_called);
  CHECK_FALSE(out.error_called);
  CHECK(out.has_token);
  CHECK(out.token.kind == emel::gbnf::lexer::event::token_kind::identifier);
  CHECK(out.token.text == "root");
  CHECK(out.token.start == 0);
  CHECK(out.token.end == 4);
  CHECK(cursor.offset == 0);
  CHECK(cursor.token_count == 0);
  CHECK(out.next_cursor.offset == 4);
  CHECK(out.next_cursor.token_count == 1);

  cursor = out.next_cursor;
  emel::gbnf::lexer::event::next next_def{cursor, done_cb, error_cb};

  out = {};
  CHECK(machine.process_event(next_def));
  CHECK(machine.is(boost::sml::state<emel::gbnf::lexer::scanning>));
  CHECK(out.done_called);
  CHECK_FALSE(out.error_called);
  CHECK(out.has_token);
  CHECK(out.token.kind == emel::gbnf::lexer::event::token_kind::definition_operator);
  CHECK(out.token.text == "::=");
  CHECK(out.next_cursor.offset == 8);
  CHECK(out.next_cursor.token_count == 2);

  cursor = out.next_cursor;
  emel::gbnf::lexer::event::next next_lit{cursor, done_cb, error_cb};
  out = {};
  CHECK(machine.process_event(next_lit));
  CHECK(out.done_called);
  CHECK_FALSE(out.error_called);
  CHECK(out.has_token);
  CHECK(out.token.kind == emel::gbnf::lexer::event::token_kind::string_literal);
  CHECK(out.token.text == "\"a\"");
  CHECK(out.next_cursor.offset == cursor.input.size());
  CHECK(out.next_cursor.token_count == 3);

  cursor = out.next_cursor;
  emel::gbnf::lexer::event::next next_eof{cursor, done_cb, error_cb};
  out = {};
  CHECK(machine.process_event(next_eof));
  CHECK(out.done_called);
  CHECK_FALSE(out.error_called);
  CHECK_FALSE(out.has_token);
  CHECK(out.next_cursor.offset == cursor.offset);
  CHECK(out.next_cursor.token_count == cursor.token_count);
}

TEST_CASE("gbnf_lexer_empty_input_returns_eof") {
  emel::gbnf::lexer::sm machine{};
  emel::gbnf::lexer::cursor cursor{
      .input = "",
      .offset = 0,
      .token_count = 0,
  };
  probe out{};
  emel::callback<bool(const emel::gbnf::lexer::events::next_done &)> done_cb{&out, on_done};
  emel::callback<bool(const emel::gbnf::lexer::events::next_error &)> error_cb{&out, on_error};

  emel::gbnf::lexer::event::next next_ev{cursor, done_cb, error_cb};
  CHECK(machine.process_event(next_ev));
  CHECK(machine.is(boost::sml::state<emel::gbnf::lexer::scanning>));
  CHECK(out.done_called);
  CHECK_FALSE(out.error_called);
  CHECK_FALSE(out.has_token);
  CHECK(out.next_cursor.offset == 0);
  CHECK(out.next_cursor.token_count == 0);
}

TEST_CASE("gbnf_lexer_rejects_invalid_cursor_offset") {
  emel::gbnf::lexer::sm machine{};
  emel::gbnf::lexer::cursor cursor{
      .input = "a",
      .offset = 2,
      .token_count = 0,
  };
  probe out{};
  emel::callback<bool(const emel::gbnf::lexer::events::next_done &)> done_cb{&out, on_done};
  emel::callback<bool(const emel::gbnf::lexer::events::next_error &)> error_cb{&out, on_error};

  emel::gbnf::lexer::event::next next_ev{cursor, done_cb, error_cb};
  CHECK(machine.process_event(next_ev));
  CHECK(machine.is(boost::sml::state<emel::gbnf::lexer::initialized>));
  CHECK_FALSE(out.done_called);
  CHECK(out.error_called);
  CHECK(out.err == error_code(emel::gbnf::lexer::error::invalid_request));
}

TEST_CASE("gbnf_lexer_requires_callbacks") {
  emel::gbnf::lexer::sm machine{};
  emel::gbnf::lexer::cursor cursor{
      .input = "a",
      .offset = 0,
      .token_count = 0,
  };
  probe out{};
  emel::callback<bool(const emel::gbnf::lexer::events::next_done &)> missing_done{};
  emel::callback<bool(const emel::gbnf::lexer::events::next_error &)> error_cb{&out, on_error};

  emel::gbnf::lexer::event::next next_ev{cursor, missing_done, error_cb};
  CHECK(machine.process_event(next_ev));
  CHECK(machine.is(boost::sml::state<emel::gbnf::lexer::initialized>));
  CHECK(out.error_called);
  CHECK(out.err == error_code(emel::gbnf::lexer::error::invalid_request));
}
