#include <doctest/doctest.h>

#include <cstddef>
#include <string_view>

#include "emel/text/jinja/parser/detail.hpp"
#include "emel/text/jinja/parser/errors.hpp"
#include "emel/text/jinja/parser/lexer/actions.hpp"
#include "emel/text/jinja/parser/lexer/sm.hpp"

namespace {

using emel::text::jinja::lexer_result;
using emel::text::jinja::lexer::cursor;
using emel::text::jinja::lexer::event::next;
using emel::text::jinja::lexer::events::next_done;
using emel::text::jinja::lexer::events::next_error;
using emel::text::jinja::parser::error;

constexpr int32_t k_ok = emel::text::jinja::parser::to_error_code(error::none);
constexpr int32_t k_parse_failed =
    emel::text::jinja::parser::to_error_code(error::parse_failed);

struct token_step_result {
  bool done_called = false;
  bool error_called = false;
  emel::text::jinja::token token = {};
  bool has_token = false;
  cursor next_cursor = {};
  int32_t err = k_ok;
  size_t error_pos = 0;

  bool on_done(const next_done &ev) {
    done_called = true;
    token = ev.token;
    has_token = ev.has_token;
    next_cursor = ev.next_cursor;
    return true;
  }

  bool on_error(const next_error &ev) {
    error_called = true;
    err = ev.err;
    error_pos = ev.error_pos;
    return true;
  }
};

lexer_result tokenize_with_machine(std::string_view source) {
  lexer_result result{};
  result.source = std::string(source);
  emel::text::jinja::parser::lexer::detail::normalize_source(result.source);

  emel::text::jinja::parser::lexer::action::context ctx{};
  emel::text::jinja::parser::lexer::sm machine{ctx};
  cursor cur{
      result.source,
      0,
      0,
      0,
      emel::text::jinja::token_type::close_statement,
      false,
      false,
  };

  for (;;) {
    token_step_result step{};
    const next ev{
        cur,
        next::done_callback::from<token_step_result,
                                  &token_step_result::on_done>(&step),
        next::error_callback::from<token_step_result,
                                   &token_step_result::on_error>(&step),
    };
    const auto scan = emel::text::jinja::lexer::detail::scan_next_token_safe(cur);
    const emel::text::jinja::parser::lexer::event::next_runtime runtime_ev{
        ev,
        scan,
    };
    const bool accepted = machine.process_event(runtime_ev);
    if (!accepted) {
      result.error = step.error_called ? step.err : k_parse_failed;
      result.error_pos = step.error_called ? step.error_pos : cur.offset;
      break;
    }
    if (step.error_called) {
      result.error = step.err;
      result.error_pos = step.error_pos;
      break;
    }
    if (!step.done_called || !step.has_token) {
      break;
    }
    result.tokens.push_back(step.token);
    cur = step.next_cursor;
  }

  return result;
}

} // namespace

TEST_CASE("jinja_lexer_tokenizes_expression") {
  lexer_result result = tokenize_with_machine("hello {{ name }}");

  CHECK(result.error == k_ok);
  REQUIRE(result.tokens.size() >= 4);
  CHECK(result.tokens[0].type == emel::text::jinja::token_type::text);
  CHECK(result.tokens[0].value == "hello ");
  CHECK(result.tokens[1].type ==
        emel::text::jinja::token_type::open_expression);
  CHECK(result.tokens[2].type == emel::text::jinja::token_type::identifier);
  CHECK(result.tokens[2].value == "name");
  CHECK(result.tokens[3].type ==
        emel::text::jinja::token_type::close_expression);
}

TEST_CASE("jinja_lexer_handles_empty_input") {
  lexer_result result = tokenize_with_machine("");

  CHECK(result.error == k_ok);
  CHECK(result.tokens.empty());
}

TEST_CASE("jinja_lexer_tokenizes_comment") {
  lexer_result result = tokenize_with_machine("text{# note #}{{ x }}");

  CHECK(result.error == k_ok);
  REQUIRE(result.tokens.size() >= 4);
  CHECK(result.tokens[0].type == emel::text::jinja::token_type::text);
  CHECK(result.tokens[1].type == emel::text::jinja::token_type::comment);
  CHECK(result.tokens[2].type ==
        emel::text::jinja::token_type::open_expression);
  CHECK(result.tokens[3].type == emel::text::jinja::token_type::identifier);
}

TEST_CASE("jinja_lexer_rejects_invalid_escape") {
  lexer_result result = tokenize_with_machine("{{ \"\\x\" }}");

  CHECK(result.error == k_parse_failed);
  CHECK(result.error_pos > 0);
}

TEST_CASE("jinja_lexer_rejects_unterminated_escape") {
  lexer_result result = tokenize_with_machine("{{ \"foo\\");

  CHECK(result.error == k_parse_failed);
}

TEST_CASE("jinja_lexer_rejects_unterminated_comment") {
  lexer_result result = tokenize_with_machine("{# comment");

  CHECK(result.error == k_parse_failed);
}

TEST_CASE("jinja_lexer_rejects_unexpected_character") {
  lexer_result result = tokenize_with_machine("{{ @ }}");

  CHECK(result.error == k_parse_failed);
}

TEST_CASE("jinja_lexer_handles_lonely_brace") {
  lexer_result result = tokenize_with_machine("{");

  CHECK(result.error == k_ok);
  REQUIRE(result.tokens.size() == 1);
  CHECK(result.tokens[0].type == emel::text::jinja::token_type::text);
}

TEST_CASE("jinja_lexer_handles_double_dash_expression") {
  lexer_result result = tokenize_with_machine("{{--}}");

  CHECK(result.error == k_ok);
  REQUIRE(result.tokens.size() >= 2);
  CHECK(result.tokens[0].type ==
        emel::text::jinja::token_type::open_expression);
  CHECK(result.tokens.back().type ==
        emel::text::jinja::token_type::close_expression);
}

TEST_CASE("jinja_lexer_rejects_unterminated_string") {
  lexer_result result = tokenize_with_machine("{{ \"foo }}");

  CHECK(result.error == k_parse_failed);
}

TEST_CASE("jinja_lexer_handles_unary_numeric") {
  lexer_result result = tokenize_with_machine("{{ -1 + 2 }}");

  CHECK(result.error == k_ok);
  REQUIRE(result.tokens.size() >= 4);
  CHECK(result.tokens[0].type ==
        emel::text::jinja::token_type::open_expression);
  CHECK(result.tokens[1].type ==
        emel::text::jinja::token_type::numeric_literal);
  CHECK(result.tokens[1].value == "-1");
}

TEST_CASE("jinja_lexer_handles_nested_objects") {
  lexer_result result = tokenize_with_machine("{{ {'a': {'b': 1}} }}");

  CHECK(result.error == k_ok);
  REQUIRE(result.tokens.size() >= 6);
  CHECK(result.tokens[0].type ==
        emel::text::jinja::token_type::open_expression);
  CHECK(result.tokens[1].type ==
        emel::text::jinja::token_type::open_curly_bracket);
}

TEST_CASE("jinja_lexer_handles_decimal_numbers") {
  lexer_result result = tokenize_with_machine("{{ 1.25 }}");

  CHECK(result.error == k_ok);
  REQUIRE(result.tokens.size() >= 3);
  CHECK(result.tokens[1].type ==
        emel::text::jinja::token_type::numeric_literal);
  CHECK(result.tokens[1].value == "1.25");
}

TEST_CASE("jinja_lexer_handles_escaped_strings") {
  lexer_result result =
      tokenize_with_machine("{{ \"a\\n\\t\\r\\b\\f\\v\\\\\\\"\" }}");

  CHECK(result.error == k_ok);
  REQUIRE(result.tokens.size() >= 3);
  CHECK(result.tokens[1].type == emel::text::jinja::token_type::string_literal);
}

TEST_CASE("jinja_lexer_normalizes_crlf") {
  lexer_result result = tokenize_with_machine("a\r\nb\r{{ x }}\n");

  CHECK(result.error == k_ok);
  CHECK(result.source.find('\r') == std::string::npos);
}

TEST_CASE("jinja_lexer_trims_blocks") {
  lexer_result result = tokenize_with_machine(" \n  {%- if x -%}\ntext");

  CHECK(result.error == k_ok);
  REQUIRE(!result.tokens.empty());
  CHECK(result.tokens[0].type == emel::text::jinja::token_type::open_statement);
}

TEST_CASE("jinja_lexer_trims_whitespace_before_block") {
  lexer_result result = tokenize_with_machine("   {%- if x %}");

  CHECK(result.error == k_ok);
  REQUIRE(!result.tokens.empty());
  CHECK(result.tokens[0].type == emel::text::jinja::token_type::open_statement);
}

TEST_CASE("jinja_lexer_trims_whitespace_after_block") {
  lexer_result result = tokenize_with_machine("{%- if x -%}   ");

  CHECK(result.error == k_ok);
  REQUIRE(!result.tokens.empty());
  CHECK(result.tokens[0].type == emel::text::jinja::token_type::open_statement);
}

TEST_CASE("jinja_lexer_trims_newline_after_block") {
  lexer_result result = tokenize_with_machine("{% if x %}\ntext");

  CHECK(result.error == k_ok);
  REQUIRE(result.tokens.size() >= 5);
  CHECK(result.tokens.back().type == emel::text::jinja::token_type::text);
  CHECK(result.tokens.back().value == "text");
}

TEST_CASE("jinja_lexer_handles_single_quote_escape") {
  lexer_result result = tokenize_with_machine("{{ 'it\\'s' }}");

  CHECK(result.error == k_ok);
  REQUIRE(result.tokens.size() >= 3);
  CHECK(result.tokens[1].type == emel::text::jinja::token_type::string_literal);
  CHECK(result.tokens[1].value == "it's");
}

TEST_CASE("jinja_lexer_lstrip_block_trims_trailing_whitespace") {
  lexer_result result = tokenize_with_machine("hello  {%- if x %}");

  CHECK(result.error == k_ok);
  REQUIRE(result.tokens.size() >= 2);
  CHECK(result.tokens[0].type == emel::text::jinja::token_type::text);
  CHECK(result.tokens[0].value == "hello");
  CHECK(result.tokens[1].type == emel::text::jinja::token_type::open_statement);
}

TEST_CASE("jinja_lexer_handles_open_expression_trailing_dash") {
  lexer_result result = tokenize_with_machine("{{-");

  CHECK(result.error == k_ok);
  REQUIRE(result.tokens.size() == 1);
  CHECK(result.tokens[0].type ==
        emel::text::jinja::token_type::open_expression);
}
