#include "bench_cases.hpp"

#include <cstdio>
#include <string>

#include "emel/jinja/lexer.hpp"
#include "emel/jinja/parser/detail.hpp"

#include "jinja/lexer.h"
#include "jinja/parser.h"

namespace {

std::string make_long_template() {
  std::string out;
  out.reserve(2048);
  for (int i = 0; i < 12; ++i) {
    out += "{% if cond %}hello {{ name }}{% else %}bye{% endif %}\n";
    out += "{% for item in items %}{{ item }}{% endfor %}\n";
  }
  return out;
}

void ensure_emel_parses(const std::string & templ) {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result lex_res = lex.tokenize(templ);
  if (lex_res.error != EMEL_OK) {
    std::fprintf(stderr, "error: emel jinja lexer failed at %zu\n", lex_res.error_pos);
    std::abort();
  }
  emel::jinja::program program{};
  emel::jinja::parser::detail::recursive_descent_parser parser{program};
  if (!parser.parse(lex_res)) {
    std::fprintf(stderr, "error: emel jinja parser failed at %zu\n", parser.error_pos());
    std::abort();
  }
}

void ensure_reference_parses(const std::string & templ) {
  try {
    ::jinja::lexer lex;
    ::jinja::lexer_result lex_res = lex.tokenize(templ);
    (void)::jinja::parse_from_tokens(lex_res);
  } catch (const std::exception & ex) {
    std::fprintf(stderr, "error: reference jinja parse failed: %s\n", ex.what());
    std::abort();
  }
}

}  // namespace

namespace emel::bench {

void append_emel_jinja_parser_cases(std::vector<result> & results, const config & cfg) {
  const std::string short_template = "hello {{ name }}";
  const std::string long_template = make_long_template();

  ensure_emel_parses(short_template);
  ensure_emel_parses(long_template);

  emel::jinja::lexer lex;
  auto short_fn = [&]() {
    emel::jinja::lexer_result lex_res = lex.tokenize(short_template);
    emel::jinja::program program{};
    emel::jinja::parser::detail::recursive_descent_parser parser{program};
    (void)parser.parse(lex_res);
  };
  results.push_back(measure_case("jinja/parser_short", cfg, short_fn));

  auto long_fn = [&]() {
    emel::jinja::lexer_result lex_res = lex.tokenize(long_template);
    emel::jinja::program program{};
    emel::jinja::parser::detail::recursive_descent_parser parser{program};
    (void)parser.parse(lex_res);
  };
  results.push_back(measure_case("jinja/parser_long", cfg, long_fn));
}

void append_reference_jinja_parser_cases(std::vector<result> & results, const config & cfg) {
  const std::string short_template = "hello {{ name }}";
  const std::string long_template = make_long_template();

  ensure_reference_parses(short_template);
  ensure_reference_parses(long_template);

  ::jinja::lexer lex;
  auto short_fn = [&]() {
    ::jinja::lexer_result lex_res = lex.tokenize(short_template);
    (void)::jinja::parse_from_tokens(lex_res);
  };
  results.push_back(measure_case("jinja/parser_short", cfg, short_fn));

  auto long_fn = [&]() {
    ::jinja::lexer_result lex_res = lex.tokenize(long_template);
    (void)::jinja::parse_from_tokens(lex_res);
  };
  results.push_back(measure_case("jinja/parser_long", cfg, long_fn));
}

}  // namespace emel::bench
