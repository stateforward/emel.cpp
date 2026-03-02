#include "bench_cases.hpp"

#include <cstdio>
#include <string>

#include "emel/text/jinja/formatter/sm.hpp"
#include "emel/text/jinja/parser/errors.hpp"
#include "emel/text/jinja/parser/sm.hpp"

#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"

namespace {

bool parser_done_sink(const emel::text::jinja::events::parsing_done &) {
  return true;
}

bool parser_error_sink(const emel::text::jinja::events::parsing_error &) {
  return true;
}

std::string make_long_template() {
  std::string out;
  out.reserve(2048);
  for (int i = 0; i < 10; ++i) {
    out += "{% if cond %}hello {{ name|upper }}{% else %}bye{% endif %}\n";
    out += "{% for item in items %}{{ item }}{% endfor %}\n";
  }
  return out;
}

emel::text::jinja::program parse_emel(const std::string & templ) {
  emel::text::jinja::program program{};
  int32_t parse_err = static_cast<int32_t>(emel::text::jinja::parser::error::none);
  size_t parse_error_pos = 0;
  emel::text::jinja::parser::action::context parse_ctx{};
  emel::text::jinja::parser::sm parser{parse_ctx};
  const emel::text::jinja::event::parse parse_ev{
      templ,
      program,
      emel::text::jinja::event::parse::done_callback::from<&parser_done_sink>(),
      emel::text::jinja::event::parse::error_callback::from<&parser_error_sink>(),
      parse_err,
      parse_error_pos,
  };
  if (!parser.process_event(parse_ev) ||
      parse_err != static_cast<int32_t>(emel::text::jinja::parser::error::none)) {
    std::fprintf(stderr, "error: emel jinja parser failed at %zu\n", parse_error_pos);
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

emel::text::jinja::value make_string(std::string_view v) {
  emel::text::jinja::value out;
  out.type = emel::text::jinja::value_type::string;
  out.string_v.view = v;
  return out;
}

emel::text::jinja::value make_bool(bool v) {
  emel::text::jinja::value out;
  out.type = emel::text::jinja::value_type::boolean;
  out.bool_v = v;
  return out;
}

emel::text::jinja::value make_int(int64_t v) {
  emel::text::jinja::value out;
  out.type = emel::text::jinja::value_type::integer;
  out.int_v = v;
  out.float_v = static_cast<double>(v);
  return out;
}

emel::text::jinja::value make_array(emel::text::jinja::value * items, size_t count) {
  emel::text::jinja::value out;
  out.type = emel::text::jinja::value_type::array;
  out.array_v.items = items;
  out.array_v.count = count;
  out.array_v.capacity = count;
  return out;
}

bool formatter_done_sink(const emel::text::jinja::events::rendering_done &) {
  return true;
}

bool formatter_error_sink(const emel::text::jinja::events::rendering_error &) {
  return true;
}

}  // namespace

namespace emel::bench {

void append_emel_jinja_formatter_cases(std::vector<result> & results, const config & cfg) {
  const std::string short_template = "hello {{ name|upper }}";
  const std::string long_template = make_long_template();

  const emel::text::jinja::program short_program = parse_emel(short_template);
  const emel::text::jinja::program long_program = parse_emel(long_template);

  std::array<emel::text::jinja::value, 3> items = {make_int(1), make_int(2), make_int(3)};
  emel::text::jinja::value items_val = make_array(items.data(), items.size());

  std::array<emel::text::jinja::object_entry, 3> entries = {};
  entries[0].key = make_string("name");
  entries[0].val = make_string("world");
  entries[1].key = make_string("cond");
  entries[1].val = make_bool(true);
  entries[2].key = make_string("items");
  entries[2].val = items_val;
  emel::text::jinja::object_value globals{entries.data(), entries.size(), entries.size(), false};

  emel::text::jinja::formatter::action::context ctx{};
  emel::text::jinja::formatter::sm machine{ctx};
  const emel::text::jinja::event::render::done_callback done_cb =
      emel::text::jinja::event::render::done_callback::from<&formatter_done_sink>();
  const emel::text::jinja::event::render::error_callback error_cb =
      emel::text::jinja::event::render::error_callback::from<&formatter_error_sink>();
  static volatile uint64_t sink = 0;

  auto short_fn = [&]() {
    std::array<char, 256> buffer = {};
    size_t out_len = 0;
    int32_t err = static_cast<int32_t>(emel::text::jinja::formatter::error::none);
    emel::text::jinja::event::render ev{
        short_program,
        short_template,
        buffer[0],
        buffer.size(),
        done_cb,
        error_cb,
        &globals,
        &out_len,
        nullptr,
        &err,
    };
    const bool ok = machine.process_event(ev);
    if (!ok || err != static_cast<int32_t>(emel::text::jinja::formatter::error::none)) {
      std::abort();
    }
    sink += out_len;
    if (out_len > 0) {
      sink += static_cast<uint64_t>(buffer[out_len - 1]);
    }
  };
  results.push_back(measure_case("text/jinja/formatter_short", cfg, short_fn));

  auto long_fn = [&]() {
    std::array<char, 4096> buffer = {};
    size_t out_len = 0;
    int32_t err = static_cast<int32_t>(emel::text::jinja::formatter::error::none);
    emel::text::jinja::event::render ev{
        long_program,
        long_template,
        buffer[0],
        buffer.size(),
        done_cb,
        error_cb,
        &globals,
        &out_len,
        nullptr,
        &err,
    };
    const bool ok = machine.process_event(ev);
    if (!ok || err != static_cast<int32_t>(emel::text::jinja::formatter::error::none)) {
      std::abort();
    }
    sink += out_len;
    if (out_len > 0) {
      sink += static_cast<uint64_t>(buffer[out_len - 1]);
    }
  };
  results.push_back(measure_case("text/jinja/formatter_long", cfg, long_fn));
  (void)sink;
}

void append_reference_jinja_formatter_cases(std::vector<result> & results, const config & cfg) {
  const std::string short_template = "hello {{ name|upper }}";
  const std::string long_template = make_long_template();

  const ::jinja::program short_program = parse_reference(short_template);
  const ::jinja::program long_program = parse_reference(long_template);
  static volatile uint64_t sink = 0;

  auto short_fn = [&]() {
    ::jinja::context ctx;
    ctx.set_val("name", ::jinja::mk_val<::jinja::value_string>("world"));
    ::jinja::runtime runtime{ctx};
    auto result = runtime.execute(short_program);
    auto parts = ::jinja::runtime::gather_string_parts(result);
    const std::string rendered = ::jinja::render_string_parts(parts);
    sink += rendered.size();
    if (!rendered.empty()) {
      sink += static_cast<uint64_t>(rendered.back());
    }
  };
  results.push_back(measure_case("text/jinja/formatter_short", cfg, short_fn));

  auto long_fn = [&]() {
    ::jinja::context ctx;
    ctx.set_val("name", ::jinja::mk_val<::jinja::value_string>("world"));
    ctx.set_val("cond", ::jinja::mk_val<::jinja::value_bool>(true));
    auto items = ::jinja::mk_val<::jinja::value_array>();
    items->push_back(::jinja::mk_val<::jinja::value_int>(1));
    items->push_back(::jinja::mk_val<::jinja::value_int>(2));
    items->push_back(::jinja::mk_val<::jinja::value_int>(3));
    ctx.set_val("items", items);
    ::jinja::runtime runtime{ctx};
    auto result = runtime.execute(long_program);
    auto parts = ::jinja::runtime::gather_string_parts(result);
    const std::string rendered = ::jinja::render_string_parts(parts);
    sink += rendered.size();
    if (!rendered.empty()) {
      sink += static_cast<uint64_t>(rendered.back());
    }
  };
  results.push_back(measure_case("text/jinja/formatter_long", cfg, long_fn));
  (void)sink;
}

}  // namespace emel::bench
