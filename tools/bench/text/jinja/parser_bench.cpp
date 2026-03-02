#include "bench_cases.hpp"

#include <cstdio>
#include <string>

#include "emel/text/jinja/parser/errors.hpp"
#include "emel/text/jinja/parser/sm.hpp"

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

bool parser_done_sink(const emel::text::jinja::events::parsing_done &) {
  return true;
}

bool parser_error_sink(const emel::text::jinja::events::parsing_error &) {
  return true;
}

void ensure_emel_parses(const std::string & templ) {
  emel::text::jinja::program program{};
  int32_t err = static_cast<int32_t>(emel::text::jinja::parser::error::none);
  size_t error_pos = 0;
  emel::text::jinja::parser::action::context ctx{};
  emel::text::jinja::parser::sm machine{ctx};

  const emel::text::jinja::event::parse ev{
      templ,
      program,
      emel::text::jinja::event::parse::done_callback::from<&parser_done_sink>(),
      emel::text::jinja::event::parse::error_callback::from<&parser_error_sink>(),
      err,
      error_pos,
  };
  const bool ok = machine.process_event(ev);
  if (!ok || err != static_cast<int32_t>(emel::text::jinja::parser::error::none)) {
    std::fprintf(stderr, "error: emel jinja parser failed at %zu\n", error_pos);
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

  emel::text::jinja::parser::action::context parser_ctx{};
  emel::text::jinja::parser::sm machine{parser_ctx};
  const emel::text::jinja::event::parse::done_callback done_cb =
      emel::text::jinja::event::parse::done_callback::from<&parser_done_sink>();
  const emel::text::jinja::event::parse::error_callback error_cb =
      emel::text::jinja::event::parse::error_callback::from<&parser_error_sink>();
  static volatile uint64_t sink = 0;

  auto short_fn = [&]() {
    emel::text::jinja::program program{};
    int32_t err = static_cast<int32_t>(emel::text::jinja::parser::error::none);
    size_t error_pos = 0;
    const emel::text::jinja::event::parse ev{
        short_template,
        program,
        done_cb,
        error_cb,
        err,
        error_pos,
    };
    const bool ok = machine.process_event(ev);
    if (!ok || err != static_cast<int32_t>(emel::text::jinja::parser::error::none)) {
      std::abort();
    }
    sink += program.body.size();
    if (!program.body.empty()) {
      sink += static_cast<uint64_t>(program.body.back()->pos);
    }
  };
  results.push_back(measure_case("text/jinja/parser_short", cfg, short_fn));

  auto long_fn = [&]() {
    emel::text::jinja::program program{};
    int32_t err = static_cast<int32_t>(emel::text::jinja::parser::error::none);
    size_t error_pos = 0;
    const emel::text::jinja::event::parse ev{
        long_template,
        program,
        done_cb,
        error_cb,
        err,
        error_pos,
    };
    const bool ok = machine.process_event(ev);
    if (!ok || err != static_cast<int32_t>(emel::text::jinja::parser::error::none)) {
      std::abort();
    }
    sink += program.body.size();
    if (!program.body.empty()) {
      sink += static_cast<uint64_t>(program.body.back()->pos);
    }
  };
  results.push_back(measure_case("text/jinja/parser_long", cfg, long_fn));
  (void)sink;
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
  results.push_back(measure_case("text/jinja/parser_short", cfg, short_fn));

  auto long_fn = [&]() {
    ::jinja::lexer_result lex_res = lex.tokenize(long_template);
    (void)::jinja::parse_from_tokens(lex_res);
  };
  results.push_back(measure_case("text/jinja/parser_long", cfg, long_fn));
}

}  // namespace emel::bench
