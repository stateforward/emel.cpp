#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/text/jinja/lexer/detail.hpp"
#include "emel/text/jinja/parser/detail.hpp"
#include "emel/text/jinja/parser/errors.hpp"

namespace emel::text::jinja::events {

struct parsing_done;
struct parsing_error;

} // namespace emel::text::jinja::events

namespace emel::text::jinja::event {

enum class parse_phase : uint8_t {
  none = 0,
  request_validation = 1,
  tokenization = 2,
  statement_classification = 3,
  parsing = 4,
};

enum class statement_kind : uint8_t {
  unknown = 0,
  text = 1,
  comment = 2,
  expression = 3,
  statement = 4,
};

enum class expression_kind : uint8_t {
  unknown = 0,
  literal = 1,
  identifier = 2,
  unary = 3,
  compound = 4,
};

struct parse {
  using done_callback =
      ::emel::callback<bool(const ::emel::text::jinja::events::parsing_done &)>;
  using error_callback = ::emel::callback<bool(
      const ::emel::text::jinja::events::parsing_error &)>;

  parse(std::string_view template_text_ref,
        emel::text::jinja::program &program_ref,
        const done_callback dispatch_done_ref,
        const error_callback dispatch_error_ref, int32_t &error_out_ref,
        size_t &error_pos_out_ref) noexcept
      : template_text(template_text_ref), program(program_ref),
        dispatch_done(dispatch_done_ref), dispatch_error(dispatch_error_ref),
        error_out(error_out_ref), error_pos_out(error_pos_out_ref) {}

  const std::string_view template_text;
  emel::text::jinja::program &program;
  const done_callback dispatch_done;
  const error_callback dispatch_error;
  int32_t &error_out;
  size_t &error_pos_out;
};

struct parse_ctx {
  parse_ctx(std::string_view template_text_ref, int32_t &error_out_ref,
            size_t &error_pos_out_ref) noexcept
      : error_out(error_out_ref), error_pos_out(error_pos_out_ref) {
    lex_result.source = std::string(template_text_ref);
    ::emel::text::jinja::lexer::detail::normalize_source(lex_result.source);
  }

  parser::error err = parser::error::none;
  size_t error_pos = 0;

  parse_phase phase = parse_phase::none;
  statement_kind statement = statement_kind::unknown;
  expression_kind expression = expression_kind::unknown;
  size_t token_index = 0;
  size_t statement_start = 0;
  size_t expression_start = 0;
  size_t expression_value_index = 0;

  emel::text::jinja::lexer::cursor lex_cursor = {};
  emel::text::jinja::token lex_token = {};
  bool lex_has_token = false;
  emel::text::jinja::lexer_result lex_result = {};

  int32_t &error_out;
  size_t &error_pos_out;
};

struct parse_runtime {
  const parse &request;
  parse_ctx &ctx;
};

} // namespace emel::text::jinja::event

namespace emel::text::jinja::events {

struct parsing_done {
  const event::parse &request;
};

struct parsing_error {
  const event::parse &request;
  int32_t err;
  size_t error_pos;
};

} // namespace emel::text::jinja::events
