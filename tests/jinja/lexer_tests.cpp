#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/jinja/lexer.hpp"

TEST_CASE("jinja_lexer_tokenizes_expression") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("Hello {{ name }}");

  CHECK(result.error == EMEL_OK);
  REQUIRE(result.tokens.size() >= 4);
  CHECK(result.tokens[0].type == emel::jinja::token_type::Text);
  CHECK(result.tokens[0].value == "Hello ");
  CHECK(result.tokens[1].type == emel::jinja::token_type::OpenExpression);
  CHECK(result.tokens[2].type == emel::jinja::token_type::Identifier);
  CHECK(result.tokens[2].value == "name");
  CHECK(result.tokens[3].type == emel::jinja::token_type::CloseExpression);
}

TEST_CASE("jinja_lexer_handles_empty_input") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("");

  CHECK(result.error == EMEL_OK);
  CHECK(result.tokens.empty());
}

TEST_CASE("jinja_lexer_tokenizes_comment") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("text{# note #}{{ x }}");

  CHECK(result.error == EMEL_OK);
  REQUIRE(result.tokens.size() >= 4);
  CHECK(result.tokens[0].type == emel::jinja::token_type::Text);
  CHECK(result.tokens[1].type == emel::jinja::token_type::Comment);
  CHECK(result.tokens[2].type == emel::jinja::token_type::OpenExpression);
  CHECK(result.tokens[3].type == emel::jinja::token_type::Identifier);
}

TEST_CASE("jinja_lexer_rejects_invalid_escape") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("{{ \"\\x\" }}");

  CHECK(result.error == EMEL_ERR_PARSE_FAILED);
  CHECK(result.error_pos > 0);
}

TEST_CASE("jinja_lexer_rejects_unterminated_escape") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("{{ \"foo\\");

  CHECK(result.error == EMEL_ERR_PARSE_FAILED);
}

TEST_CASE("jinja_lexer_rejects_unterminated_comment") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("{# comment");

  CHECK(result.error == EMEL_ERR_PARSE_FAILED);
}

TEST_CASE("jinja_lexer_rejects_unexpected_character") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("{{ @ }}");

  CHECK(result.error == EMEL_ERR_PARSE_FAILED);
}

TEST_CASE("jinja_lexer_handles_lonely_brace") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("{");

  CHECK(result.error == EMEL_OK);
  REQUIRE(result.tokens.size() == 1);
  CHECK(result.tokens[0].type == emel::jinja::token_type::Text);
}

TEST_CASE("jinja_lexer_handles_double_dash_expression") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("{{--}}");

  CHECK(result.error == EMEL_OK);
  REQUIRE(result.tokens.size() >= 2);
  CHECK(result.tokens[0].type == emel::jinja::token_type::OpenExpression);
  CHECK(result.tokens.back().type == emel::jinja::token_type::CloseExpression);
}

TEST_CASE("jinja_lexer_rejects_unterminated_string") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("{{ \"foo }}");

  CHECK(result.error == EMEL_ERR_PARSE_FAILED);
}

TEST_CASE("jinja_lexer_handles_unary_numeric") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("{{ -1 + 2 }}");

  CHECK(result.error == EMEL_OK);
  REQUIRE(result.tokens.size() >= 4);
  CHECK(result.tokens[0].type == emel::jinja::token_type::OpenExpression);
  CHECK(result.tokens[1].type == emel::jinja::token_type::NumericLiteral);
  CHECK(result.tokens[1].value == "-1");
}

TEST_CASE("jinja_lexer_handles_nested_objects") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("{{ {'a': {'b': 1}} }}");

  CHECK(result.error == EMEL_OK);
  REQUIRE(result.tokens.size() >= 6);
  CHECK(result.tokens[0].type == emel::jinja::token_type::OpenExpression);
  CHECK(result.tokens[1].type == emel::jinja::token_type::OpenCurlyBracket);
}

TEST_CASE("jinja_lexer_handles_decimal_numbers") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("{{ 1.25 }}");

  CHECK(result.error == EMEL_OK);
  REQUIRE(result.tokens.size() >= 3);
  CHECK(result.tokens[1].type == emel::jinja::token_type::NumericLiteral);
  CHECK(result.tokens[1].value == "1.25");
}

TEST_CASE("jinja_lexer_handles_escaped_strings") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result =
      lex.tokenize("{{ \"a\\n\\t\\r\\b\\f\\v\\\\\\\"\" }}");

  CHECK(result.error == EMEL_OK);
  REQUIRE(result.tokens.size() >= 3);
  CHECK(result.tokens[1].type == emel::jinja::token_type::StringLiteral);
}

TEST_CASE("jinja_lexer_normalizes_crlf") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("a\r\nb\r{{ x }}\n");

  CHECK(result.error == EMEL_OK);
  CHECK(result.source.find('\r') == std::string::npos);
}

TEST_CASE("jinja_lexer_trims_blocks") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result =
      lex.tokenize(" \n  {%- if x -%}\ntext");

  CHECK(result.error == EMEL_OK);
  REQUIRE(!result.tokens.empty());
  CHECK(result.tokens[0].type == emel::jinja::token_type::OpenStatement);
}

TEST_CASE("jinja_lexer_trims_whitespace_before_block") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("   {%- if x %}");

  CHECK(result.error == EMEL_OK);
  REQUIRE(!result.tokens.empty());
  CHECK(result.tokens[0].type == emel::jinja::token_type::OpenStatement);
}

TEST_CASE("jinja_lexer_trims_whitespace_after_block") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("{%- if x -%}   ");

  CHECK(result.error == EMEL_OK);
  REQUIRE(!result.tokens.empty());
  CHECK(result.tokens[0].type == emel::jinja::token_type::OpenStatement);
}

TEST_CASE("jinja_lexer_trims_newline_after_block") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("{% if x %}\ntext");

  CHECK(result.error == EMEL_OK);
  REQUIRE(result.tokens.size() >= 5);
  CHECK(result.tokens.back().type == emel::jinja::token_type::Text);
  CHECK(result.tokens.back().value == "text");
}

TEST_CASE("jinja_lexer_handles_single_quote_escape") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("{{ 'it\\'s' }}");

  CHECK(result.error == EMEL_OK);
  REQUIRE(result.tokens.size() >= 3);
  CHECK(result.tokens[1].type == emel::jinja::token_type::StringLiteral);
  CHECK(result.tokens[1].value == "it's");
}

TEST_CASE("jinja_lexer_lstrip_block_trims_trailing_whitespace") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("hello  {%- if x %}");

  CHECK(result.error == EMEL_OK);
  REQUIRE(result.tokens.size() >= 2);
  CHECK(result.tokens[0].type == emel::jinja::token_type::Text);
  CHECK(result.tokens[0].value == "hello");
  CHECK(result.tokens[1].type == emel::jinja::token_type::OpenStatement);
}

TEST_CASE("jinja_lexer_handles_open_expression_trailing_dash") {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result result = lex.tokenize("{{-");

  CHECK(result.error == EMEL_OK);
  REQUIRE(result.tokens.size() == 1);
  CHECK(result.tokens[0].type == emel::jinja::token_type::OpenExpression);
}
