#include "bench_cases.hpp"

#include <cstdio>
#include <string>

#include "emel/jinja/lexer.hpp"
#include "emel/jinja/parser/detail.hpp"
#include "emel/jinja/renderer/sm.hpp"

#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"

namespace {

std::string make_long_template() {
  std::string out;
  out.reserve(2048);
  for (int i = 0; i < 10; ++i) {
    out += "{% if cond %}Hello {{ name|upper }}{% else %}Bye{% endif %}\n";
    out += "{% for item in items %}{{ item }}{% endfor %}\n";
  }
  return out;
}

emel::jinja::program parse_emel(const std::string & templ) {
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
  return program;
}

::jinja::program parse_reference(const std::string & templ) {
  try {
    ::jinja::lexer lex;
    ::jinja::lexer_result lex_res = lex.tokenize(templ);
    return ::jinja::parse_from_tokens(lex_res);
  } catch (const std::exception & ex) {
    std::fprintf(stderr, "error: reference jinja parse failed: %s\n", ex.what());
    std::abort();
  }
}

emel::jinja::value make_string(std::string_view v) {
  emel::jinja::value out;
  out.type = emel::jinja::value_type::string;
  out.string_v.view = v;
  return out;
}

emel::jinja::value make_bool(bool v) {
  emel::jinja::value out;
  out.type = emel::jinja::value_type::boolean;
  out.bool_v = v;
  return out;
}

emel::jinja::value make_int(int64_t v) {
  emel::jinja::value out;
  out.type = emel::jinja::value_type::integer;
  out.int_v = v;
  out.float_v = static_cast<double>(v);
  return out;
}

emel::jinja::value make_array(emel::jinja::value * items, size_t count) {
  emel::jinja::value out;
  out.type = emel::jinja::value_type::array;
  out.array_v.items = items;
  out.array_v.count = count;
  out.array_v.capacity = count;
  return out;
}

}  // namespace

namespace emel::bench {

void append_emel_jinja_renderer_cases(std::vector<result> & results, const config & cfg) {
  const std::string short_template = "Hello {{ name|upper }}";
  const std::string long_template = make_long_template();

  const emel::jinja::program short_program = parse_emel(short_template);
  const emel::jinja::program long_program = parse_emel(long_template);

  std::array<emel::jinja::value, 3> items = {make_int(1), make_int(2), make_int(3)};
  emel::jinja::value items_val = make_array(items.data(), items.size());

  std::array<emel::jinja::object_entry, 3> entries = {};
  entries[0].key = make_string("name");
  entries[0].val = make_string("World");
  entries[1].key = make_string("cond");
  entries[1].val = make_bool(true);
  entries[2].key = make_string("items");
  entries[2].val = items_val;
  emel::jinja::object_value globals{entries.data(), entries.size(), entries.size(), false};

  emel::jinja::renderer::action::context ctx{};
  emel::jinja::renderer::sm machine{ctx};

  auto short_fn = [&]() {
    std::array<char, 256> buffer = {};
    size_t out_len = 0;
    int32_t err = EMEL_OK;
    emel::jinja::event::render ev{
      .program = &short_program,
      .globals = &globals,
      .output = buffer.data(),
      .output_capacity = buffer.size(),
      .output_length = &out_len,
      .error_out = &err,
    };
    (void)machine.process_event(ev);
  };
  results.push_back(measure_case("jinja/renderer_short", cfg, short_fn));

  auto long_fn = [&]() {
    std::array<char, 1024> buffer = {};
    size_t out_len = 0;
    int32_t err = EMEL_OK;
    emel::jinja::event::render ev{
      .program = &long_program,
      .globals = &globals,
      .output = buffer.data(),
      .output_capacity = buffer.size(),
      .output_length = &out_len,
      .error_out = &err,
    };
    (void)machine.process_event(ev);
  };
  results.push_back(measure_case("jinja/renderer_long", cfg, long_fn));
}

void append_reference_jinja_renderer_cases(std::vector<result> & results, const config & cfg) {
  const std::string short_template = "Hello {{ name|upper }}";
  const std::string long_template = make_long_template();

  const ::jinja::program short_program = parse_reference(short_template);
  const ::jinja::program long_program = parse_reference(long_template);

  auto short_fn = [&]() {
    ::jinja::context ctx;
    ctx.set_val("name", ::jinja::mk_val<::jinja::value_string>("World"));
    ::jinja::runtime runtime{ctx};
    auto result = runtime.execute(short_program);
    auto parts = ::jinja::runtime::gather_string_parts(result);
    (void)::jinja::render_string_parts(parts);
  };
  results.push_back(measure_case("jinja/renderer_short", cfg, short_fn));

  auto long_fn = [&]() {
    ::jinja::context ctx;
    ctx.set_val("name", ::jinja::mk_val<::jinja::value_string>("World"));
    ctx.set_val("cond", ::jinja::mk_val<::jinja::value_bool>(true));
    auto items = ::jinja::mk_val<::jinja::value_array>();
    items->push_back(::jinja::mk_val<::jinja::value_int>(1));
    items->push_back(::jinja::mk_val<::jinja::value_int>(2));
    items->push_back(::jinja::mk_val<::jinja::value_int>(3));
    ctx.set_val("items", items);
    ::jinja::runtime runtime{ctx};
    auto result = runtime.execute(long_program);
    auto parts = ::jinja::runtime::gather_string_parts(result);
    (void)::jinja::render_string_parts(parts);
  };
  results.push_back(measure_case("jinja/renderer_long", cfg, long_fn));
}

}  // namespace emel::bench
