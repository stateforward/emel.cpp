#include <array>
#include <string>
#include <string_view>

#include "doctest/doctest.h"

#include "emel/jinja/lexer.hpp"
#include "emel/jinja/parser/detail.hpp"
#include "emel/jinja/renderer/sm.hpp"

namespace {

auto parse_template(const std::string & text, emel::jinja::program & program) {
  emel::jinja::lexer lex;
  emel::jinja::lexer_result lex_res = lex.tokenize(text);
  CHECK(lex_res.error == EMEL_OK);
  emel::jinja::parser::detail::recursive_descent_parser parser{program};
  CHECK(parser.parse(lex_res));
}

struct render_result {
  int32_t err = EMEL_OK;
  size_t error_pos = 0;
  std::string output;
  bool done = false;
};

render_result render_template(const std::string & templ,
                              const emel::jinja::object_value * globals = nullptr) {
  emel::jinja::program program{};
  parse_template(templ, program);

  std::array<char, 512> buffer = {};
  size_t out_len = 0;
  size_t error_pos = 0;
  int32_t err = EMEL_OK;

  emel::jinja::renderer::action::context ctx{};
  emel::jinja::renderer::sm machine{ctx};
  emel::jinja::event::render ev{
    .program = &program,
    .globals = globals,
    .output = buffer.data(),
    .output_capacity = buffer.size(),
    .output_length = &out_len,
    .error_out = &err,
    .error_pos_out = &error_pos,
  };

  machine.process_event(ev);

  render_result result{};
  result.err = err;
  result.error_pos = error_pos;
  result.output.assign(buffer.data(), out_len);
  result.done = machine.is(boost::sml::state<emel::jinja::renderer::Done>);
  return result;
}

emel::jinja::value make_string(std::string_view v) {
  emel::jinja::value out;
  out.type = emel::jinja::value_type::string;
  out.string_v.view = v;
  return out;
}

emel::jinja::value make_int(int64_t v) {
  emel::jinja::value out;
  out.type = emel::jinja::value_type::integer;
  out.int_v = v;
  out.float_v = static_cast<double>(v);
  return out;
}

emel::jinja::value make_bool(bool v) {
  emel::jinja::value out;
  out.type = emel::jinja::value_type::boolean;
  out.bool_v = v;
  return out;
}

emel::jinja::value make_none() {
  emel::jinja::value out;
  out.type = emel::jinja::value_type::none;
  return out;
}

emel::jinja::value make_float(double v) {
  emel::jinja::value out;
  out.type = emel::jinja::value_type::floating;
  out.float_v = v;
  out.int_v = static_cast<int64_t>(v);
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

emel::jinja::value make_object(emel::jinja::object_entry * entries, size_t count) {
  emel::jinja::value out;
  out.type = emel::jinja::value_type::object;
  out.object_v.entries = entries;
  out.object_v.count = count;
  out.object_v.capacity = count;
  out.object_v.has_builtins = false;
  return out;
}

}  // namespace

TEST_CASE("jinja_renderer_starts_initialized") {
  emel::jinja::renderer::action::context ctx{};
  emel::jinja::renderer::sm machine{ctx};
  CHECK(machine.is(boost::sml::state<emel::jinja::renderer::Initialized>));
}

TEST_CASE("jinja_renderer_renders_simple_template") {
  emel::jinja::program program{};
  parse_template("Hello {{ name }}!", program);

  std::array<emel::jinja::object_entry, 1> entries = {};
  entries[0].key = make_string("name");
  entries[0].val = make_string("World");
  emel::jinja::object_value globals{entries.data(), entries.size(), entries.size(), false};

  std::array<char, 64> buffer = {};
  size_t out_len = 0;
  int32_t err = EMEL_OK;

  emel::jinja::renderer::action::context ctx{};
  emel::jinja::renderer::sm machine{ctx};
  emel::jinja::event::render ev{
    .program = &program,
    .globals = &globals,
    .output = buffer.data(),
    .output_capacity = buffer.size(),
    .output_length = &out_len,
    .error_out = &err,
  };

  machine.process_event(ev);

  CHECK(machine.is(boost::sml::state<emel::jinja::renderer::Done>));
  CHECK(err == EMEL_OK);
  std::string_view rendered(buffer.data(), out_len);
  CHECK(rendered == "Hello World!");
}

TEST_CASE("jinja_renderer_handles_loops_and_filters") {
  emel::jinja::program program{};
  parse_template("{% for x in items if x != 2 %}{{ x|upper }}{% endfor %}", program);

  std::array<emel::jinja::value, 3> items = {make_int(1), make_int(2), make_int(3)};
  emel::jinja::value items_val = make_array(items.data(), items.size());

  std::array<emel::jinja::object_entry, 1> entries = {};
  entries[0].key = make_string("items");
  entries[0].val = items_val;
  emel::jinja::object_value globals{entries.data(), entries.size(), entries.size(), false};

  std::array<char, 128> buffer = {};
  size_t out_len = 0;
  int32_t err = EMEL_OK;

  emel::jinja::renderer::action::context ctx{};
  emel::jinja::renderer::sm machine{ctx};
  emel::jinja::event::render ev{
    .program = &program,
    .globals = &globals,
    .output = buffer.data(),
    .output_capacity = buffer.size(),
    .output_length = &out_len,
    .error_out = &err,
  };

  machine.process_event(ev);

  CHECK(machine.is(boost::sml::state<emel::jinja::renderer::Done>));
  CHECK(err == EMEL_OK);
  std::string_view rendered(buffer.data(), out_len);
  CHECK(rendered == "13");
}

TEST_CASE("jinja_renderer_supports_set_and_member_access") {
  emel::jinja::program program{};
  parse_template("{% set name = 'Bob' %}{{ name }} {{ obj.key }}", program);

  std::array<emel::jinja::object_entry, 1> obj_entries = {};
  obj_entries[0].key = make_string("key");
  obj_entries[0].val = make_string("OK");
  emel::jinja::value obj_val = make_object(obj_entries.data(), obj_entries.size());

  std::array<emel::jinja::object_entry, 1> entries = {};
  entries[0].key = make_string("obj");
  entries[0].val = obj_val;
  emel::jinja::object_value globals{entries.data(), entries.size(), entries.size(), false};

  std::array<char, 128> buffer = {};
  size_t out_len = 0;
  int32_t err = EMEL_OK;

  emel::jinja::renderer::action::context ctx{};
  emel::jinja::renderer::sm machine{ctx};
  emel::jinja::event::render ev{
    .program = &program,
    .globals = &globals,
    .output = buffer.data(),
    .output_capacity = buffer.size(),
    .output_length = &out_len,
    .error_out = &err,
  };

  machine.process_event(ev);

  CHECK(machine.is(boost::sml::state<emel::jinja::renderer::Done>));
  CHECK(err == EMEL_OK);
  std::string_view rendered(buffer.data(), out_len);
  CHECK(rendered == "Bob OK");
}

TEST_CASE("jinja_renderer_invalid_request_errors") {
  emel::jinja::renderer::action::context ctx{};
  emel::jinja::renderer::sm machine{ctx};

  std::array<char, 16> buffer = {};
  int32_t err = EMEL_OK;
  emel::jinja::event::render ev{
    .program = nullptr,
    .output = buffer.data(),
    .output_capacity = buffer.size(),
    .error_out = &err,
  };

  machine.process_event(ev);

  CHECK(machine.is(boost::sml::state<emel::jinja::renderer::Errored>));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("jinja_renderer_ops_basic") {
  auto result = render_template("{{ 2 * 3 }}{{ 5 / 2 }}{{ 5 % 2 }}{{ \"a\" ~ \"b\" }}");
  CHECK(result.done);
  CHECK(result.err == EMEL_OK);
  CHECK_FALSE(result.output.empty());
}

TEST_CASE("jinja_renderer_filters_trim_replace") {
  auto result = render_template("{{ \"  hi \"|trim }}{{ \"aba\"|replace(\"a\", \"x\") }}");
  CAPTURE(result.error_pos);
  CAPTURE(result.output);
  CHECK(result.done);
  CHECK(result.err == EMEL_OK);
  CHECK_FALSE(result.output.empty());
}

TEST_CASE("jinja_renderer_in_ops") {
  auto result = render_template("{{ 1 in [1, 2] }}{{ 1 not in [2, 3] }}");
  CHECK(result.done);
  CHECK(result.err == EMEL_OK);
  CHECK_FALSE(result.output.empty());
}

TEST_CASE("jinja_renderer_test_ops") {
  auto result = render_template("{{ 1 is odd }}{{ 2 is even }}");
  CHECK(result.done);
  CHECK(result.err == EMEL_OK);
  CHECK_FALSE(result.output.empty());
}

TEST_CASE("jinja_renderer_range_join") {
  auto result = render_template("{{ range(0, 3)|join(\",\") }}");
  CHECK(result.done);
  CHECK(result.err == EMEL_OK);
  CHECK_FALSE(result.output.empty());
}

TEST_CASE("jinja_renderer_if_and_select") {
  std::array<emel::jinja::object_entry, 2> entries = {};
  entries[0].key = make_string("name");
  entries[0].val = make_string("Bob");
  entries[1].key = make_string("cond");
  entries[1].val = make_bool(true);
  emel::jinja::object_value globals{entries.data(), entries.size(), entries.size(), false};

  auto result = render_template(
      "{% if cond %}{{ name|upper }}{% else %}{{ name|lower }}{% endif %}"
      "{{ 1 if cond else 2 }}{{ 3 if cond }}",
      &globals);
  CHECK(result.done);
  CHECK(result.err == EMEL_OK);
  CHECK(result.output.find("BOB") != std::string::npos);
}

TEST_CASE("jinja_renderer_loops_and_members") {
  std::array<emel::jinja::value, 3> items = {make_int(1), make_int(2), make_int(3)};
  emel::jinja::value items_val = make_array(items.data(), items.size());

  std::array<emel::jinja::object_entry, 1> obj_entries = {};
  obj_entries[0].key = make_string("key");
  obj_entries[0].val = make_string("V");
  emel::jinja::value obj_val = make_object(obj_entries.data(), obj_entries.size());

  std::array<emel::jinja::value, 3> arr = {make_int(7), make_int(8), make_int(9)};
  emel::jinja::value arr_val = make_array(arr.data(), arr.size());

  std::array<emel::jinja::object_entry, 4> entries = {};
  entries[0].key = make_string("items");
  entries[0].val = items_val;
  entries[1].key = make_string("obj");
  entries[1].val = obj_val;
  entries[2].key = make_string("arr");
  entries[2].val = arr_val;
  entries[3].key = make_string("cond");
  entries[3].val = make_bool(true);
  emel::jinja::object_value globals{entries.data(), entries.size(), entries.size(), false};

  auto result = render_template(
      "{% for k, v in obj %}{{ k }}{{ v }}{% endfor %}"
      "{% for x in items if x > 1 %}{{ x }}{% endfor %}"
      "{{ obj.key }}{{ arr[1] }}{{ arr[0:2] }}",
      &globals);
  CHECK(result.done);
  CHECK(result.err == EMEL_OK);
  CHECK_FALSE(result.output.empty());
}

TEST_CASE("jinja_renderer_macro_and_call") {
  emel::jinja::program program{};
  parse_template(
      "{% macro wrap(name) %}{{ name }}{{ caller() }}{% endmacro %}"
      "{% call() wrap(\"Hi\") %}{{ \"there\"|upper }}{% endcall %}",
      program);

  std::array<char, 256> buffer = {};
  size_t out_len = 0;
  int32_t err = EMEL_OK;

  emel::jinja::renderer::action::context ctx{};
  emel::jinja::renderer::sm machine{ctx};
  emel::jinja::event::render ev{
    .program = &program,
    .output = buffer.data(),
    .output_capacity = buffer.size(),
    .output_length = &out_len,
    .error_out = &err,
  };

  machine.process_event(ev);

  CHECK(machine.is(boost::sml::state<emel::jinja::renderer::Done>));
  CHECK(err == EMEL_OK);
  std::string_view rendered(buffer.data(), out_len);
  CHECK(rendered == "HiTHERE");
}

TEST_CASE("jinja_renderer_filter_statement") {
  emel::jinja::program program{};
  parse_template("{% filter upper %}hello{% endfilter %}", program);

  std::array<char, 64> buffer = {};
  size_t out_len = 0;
  int32_t err = EMEL_OK;

  emel::jinja::renderer::action::context ctx{};
  emel::jinja::renderer::sm machine{ctx};
  emel::jinja::event::render ev{
    .program = &program,
    .output = buffer.data(),
    .output_capacity = buffer.size(),
    .output_length = &out_len,
    .error_out = &err,
  };

  machine.process_event(ev);

  CHECK(machine.is(boost::sml::state<emel::jinja::renderer::Done>));
  CHECK(err == EMEL_OK);
  std::string_view rendered(buffer.data(), out_len);
  CHECK(rendered == "HELLO");
}

TEST_CASE("jinja_renderer_truncates_on_small_buffer") {
  emel::jinja::program program{};
  parse_template("Hello {{ name }}!", program);

  std::array<emel::jinja::object_entry, 1> entries = {};
  entries[0].key = make_string("name");
  entries[0].val = make_string("World");
  emel::jinja::object_value globals{entries.data(), entries.size(), entries.size(), false};

  std::array<char, 4> buffer = {};
  size_t out_len = 0;
  bool truncated = false;
  int32_t err = EMEL_OK;

  emel::jinja::renderer::action::context ctx{};
  emel::jinja::renderer::sm machine{ctx};
  emel::jinja::event::render ev{
    .program = &program,
    .globals = &globals,
    .output = buffer.data(),
    .output_capacity = buffer.size(),
    .output_length = &out_len,
    .output_truncated = &truncated,
    .error_out = &err,
  };

  machine.process_event(ev);

  CHECK(machine.is(boost::sml::state<emel::jinja::renderer::Errored>));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(truncated);
}

TEST_CASE("jinja_renderer_dispatch_callbacks") {
  emel::jinja::program program{};
  parse_template("Hi", program);

  std::array<char, 32> buffer = {};
  size_t out_len = 0;
  int32_t err = EMEL_OK;

  struct tracker {
    bool done = false;
    bool error = false;
    size_t length = 0;

    bool on_done(const emel::jinja::events::rendering_done & ev) {
      done = true;
      length = ev.output_length;
      return true;
    }

    bool on_error(const emel::jinja::events::rendering_error &) {
      error = true;
      return true;
    }
  };

  tracker track{};
  using done_cb = emel::callback<bool(const emel::jinja::events::rendering_done &)>;
  using error_cb = emel::callback<bool(const emel::jinja::events::rendering_error &)>;

  emel::jinja::renderer::action::context ctx{};
  emel::jinja::renderer::sm machine{ctx};
  emel::jinja::event::render ev{
    .program = &program,
    .output = buffer.data(),
    .output_capacity = buffer.size(),
    .output_length = &out_len,
    .error_out = &err,
    .dispatch_done = done_cb::from<tracker, &tracker::on_done>(&track),
    .dispatch_error = error_cb::from<tracker, &tracker::on_error>(&track),
  };

  machine.process_event(ev);

  CHECK(machine.is(boost::sml::state<emel::jinja::renderer::Done>));
  CHECK(err == EMEL_OK);
  CHECK(track.done);
  CHECK_FALSE(track.error);
  CHECK(track.length == out_len);
}

TEST_CASE("jinja_renderer_rejects_invalid_and_dispatches_error") {
  std::array<char, 16> buffer = {};
  size_t out_len = 7;
  bool truncated = true;
  int32_t err = EMEL_OK;

  struct tracker {
    bool error = false;

    bool on_error(const emel::jinja::events::rendering_error & ev) {
      error = true;
      return ev.err == EMEL_ERR_INVALID_ARGUMENT;
    }
  };

  tracker track{};
  using error_cb = emel::callback<bool(const emel::jinja::events::rendering_error &)>;

  emel::jinja::renderer::action::context ctx{};
  emel::jinja::renderer::sm machine{ctx};
  emel::jinja::event::render ev{
    .program = nullptr,
    .output = buffer.data(),
    .output_capacity = buffer.size(),
    .output_length = &out_len,
    .output_truncated = &truncated,
    .error_out = &err,
    .dispatch_error = error_cb::from<tracker, &tracker::on_error>(&track),
  };

  machine.process_event(ev);

  CHECK(machine.is(boost::sml::state<emel::jinja::renderer::Errored>));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(out_len == 0);
  CHECK_FALSE(truncated);
  CHECK(track.error);
}

TEST_CASE("jinja_renderer_unexpected_event_sets_error") {
  struct unknown_event {
    int value = 0;
  };

  emel::jinja::renderer::action::context ctx{};
  emel::jinja::renderer::sm machine{ctx};
  machine.process_event(unknown_event{});

  CHECK(machine.is(boost::sml::state<emel::jinja::renderer::Unexpected>));
  CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
}

TEST_CASE("jinja_renderer_defaults_namespace_length") {
  std::array<emel::jinja::object_entry, 3> entries = {};
  entries[0].key = make_string("flag");
  entries[0].val = make_bool(false);
  entries[1].key = make_string("apply_default");
  entries[1].val = make_bool(true);
  entries[2].key = make_string("num");
  entries[2].val = make_int(7);
  emel::jinja::object_value globals{entries.data(), entries.size(), entries.size(), false};

  auto result = render_template(
      "{{ default(missing, \"x\") }}{{ default(flag, \"y\", apply_default) }}"
      "{{ namespace(a=1, b=2).b }}{{ \"abc\"|length }}{{ [1,2]|length }}"
      "{{ {\"k\":1}|length }}{{ flag }}",
      &globals);

  CHECK(result.done);
  CHECK(result.err == EMEL_OK);
  CHECK(result.output == "xy2321false");
}

TEST_CASE("jinja_renderer_range_spread_and_join") {
  auto result = render_template(
      "{{ range(*[1,4])|join(\"-\") }}"
      "{{ range(0,3)|join(\",\") }}"
      "{{ range(3,0,-1)|join(\",\") }}"
      "{{ [1,2] == [1,2] }}{{ {\"k\":1} == {\"k\":1} }}");

  CHECK(result.done);
  CHECK(result.err == EMEL_OK);
  CHECK_FALSE(result.output.empty());
}

TEST_CASE("jinja_renderer_member_and_set_variants") {
  std::array<emel::jinja::object_entry, 1> obj_entries = {};
  obj_entries[0].key = make_string("key");
  obj_entries[0].val = make_string("orig");
  emel::jinja::value obj_val = make_object(obj_entries.data(), obj_entries.size());

  std::array<emel::jinja::value, 3> arr_vals = {make_int(1), make_int(2), make_int(3)};
  emel::jinja::value arr_val = make_array(arr_vals.data(), arr_vals.size());

  std::array<emel::jinja::object_entry, 2> entries = {};
  entries[0].key = make_string("obj");
  entries[0].val = obj_val;
  entries[1].key = make_string("arr");
  entries[1].val = arr_val;
  emel::jinja::object_value globals{entries.data(), entries.size(), entries.size(), false};

  auto result = render_template(
      "{# comment #}{% generation %}"
      "{% set name %}Bob{% endset %}{{ name }}"
      "{% set a, b = [1,2] %}{{ a }}{{ b }}"
      "{% set obj.key = \"v\" %}{{ obj.key }}"
      "{{ obj[\"key\"] }}{{ \"hello\"[1] }}{{ \"hello\"[1:4] }}"
      "{{ arr[1] }}{{ arr[0:2] }}",
      &globals);

  CHECK(result.done);
  CHECK(result.err == EMEL_OK);
  CHECK_FALSE(result.output.empty());
}

TEST_CASE("jinja_renderer_control_flow_break_continue") {
  auto result = render_template(
      "{% for x in [1,2,3] %}"
      "{% if x == 2 %}{% continue %}{% endif %}"
      "{% if x == 3 %}{% break %}{% endif %}"
      "{{ x }}"
      "{% endfor %}"
      "{% for x in missing %}{{ x }}{% else %}EMPTY{% endfor %}");

  CHECK(result.done);
  CHECK(result.err == EMEL_OK);
  CHECK(result.output.find("1") != std::string::npos);
  CHECK(result.output.find("EMPTY") != std::string::npos);
}

TEST_CASE("jinja_renderer_filter_statement_with_args") {
  auto result = render_template("{% filter replace(\"a\", \"b\") %}a-a{% endfilter %}");

  CHECK(result.done);
  CHECK(result.err == EMEL_OK);
  CHECK(result.output == "b-b");
}

TEST_CASE("jinja_renderer_macro_defaults_and_caller_params") {
  auto result = render_template(
      "{% macro greet(name=\"Bob\") %}{{ name }}{% endmacro %}"
      "{% macro wrap(name) %}{{ caller(name) }}{% endmacro %}"
      "{% call(n) wrap(\"Hi\") %}{{ n|upper }}{% endcall %}"
      "{{ greet() }}{{ greet(\"Ana\") }}");

  CHECK(result.done);
  CHECK(result.err == EMEL_OK);
  CHECK_FALSE(result.output.empty());
}

TEST_CASE("jinja_renderer_tests_and_memberships") {
  std::array<emel::jinja::value, 2> arr_items = {make_int(1), make_int(2)};
  emel::jinja::value arr_val = make_array(arr_items.data(), arr_items.size());

  std::array<emel::jinja::object_entry, 1> obj_entries = {};
  obj_entries[0].key = make_string("a");
  obj_entries[0].val = make_int(1);
  emel::jinja::value obj_val = make_object(obj_entries.data(), obj_entries.size());

  std::array<emel::jinja::object_entry, 8> entries = {};
  entries[0].key = make_string("name");
  entries[0].val = make_string("Bob");
  entries[1].key = make_string("flag");
  entries[1].val = make_bool(false);
  entries[2].key = make_string("true_val");
  entries[2].val = make_bool(true);
  entries[3].key = make_string("none_val");
  entries[3].val = make_none();
  entries[4].key = make_string("num");
  entries[4].val = make_int(5);
  entries[5].key = make_string("float_val");
  entries[5].val = make_float(1.5);
  entries[6].key = make_string("arr");
  entries[6].val = arr_val;
  entries[7].key = make_string("obj");
  entries[7].val = obj_val;
  emel::jinja::object_value globals{entries.data(), entries.size(), entries.size(), false};

  auto result = render_template(
      "{{ missing is undefined }}{{ none_val is none }}{{ name is defined }}"
      "{{ name is string }}{{ flag is boolean }}{{ num is number }}"
      "{{ arr is iterable }}{{ obj is mapping }}{{ flag is false }}{{ true_val is true }}"
      "{{ \"a\" in obj }}{{ \"a\" in \"cat\" }}{{ float_val }}",
      &globals);

  CHECK(result.done);
  CHECK(result.err == EMEL_OK);
  CHECK_FALSE(result.output.empty());
}

TEST_CASE("jinja_renderer_errors_on_invalid_control_flow") {
  auto result = render_template("{% break %}");

  CHECK_FALSE(result.done);
  CHECK(result.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("jinja_renderer_errors_on_div_zero") {
  auto result = render_template("{{ 1 / 0 }}");

  CHECK_FALSE(result.done);
  CHECK(result.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("jinja_renderer_expression_matrix") {
  auto result = render_template(
      "{{ 1 or 0 }}{{ 0 or 2 }}{{ 0 and 1 }}{{ 1 and 2 }}{{ not 0 }}"
      "{{ 1 + 2 }}{{ 1 + 2.5 }}{{ \"a\" + \"b\" }}{{ 5 - 2 }}"
      "{{ 2 * 3 }}{{ \"ha\" * 2 }}{{ 5 / 2 }}{{ 5 % 2 }}"
      "{{ 1 < 2 }}{{ 2 >= 1 }}{{ \"a\" < \"b\" }}"
      "{{ 1 in [1,2] }}{{ 1 not in [2] }}{{ \"a\" in {\"a\":1} }}"
      "{{ \"a\" in \"cat\" }}");

  CHECK(result.done);
  CHECK(result.err == EMEL_OK);
  CHECK_FALSE(result.output.empty());
}

TEST_CASE("jinja_renderer_member_edge_cases") {
  std::array<emel::jinja::value, 3> arr_vals = {make_int(1), make_int(2), make_int(3)};
  emel::jinja::value arr_val = make_array(arr_vals.data(), arr_vals.size());

  std::array<emel::jinja::object_entry, 1> entries = {};
  entries[0].key = make_string("arr");
  entries[0].val = arr_val;
  emel::jinja::object_value globals{entries.data(), entries.size(), entries.size(), false};

  auto result = render_template(
      "{{ missing.key }}{{ arr[99] }}{{ arr[\"bad\"] }}{{ \"hi\"[99] }}"
      "{{ \"hi\".bad }}{{ {1:\"a\"}[1] }}"
      "{{ arr[1:] }}{{ \"hello\"[2:] }}{{ arr[0:3:2] }}",
      &globals);

  CHECK(result.done);
  CHECK(result.err == EMEL_OK);
  CHECK_FALSE(result.output.empty());
}

TEST_CASE("jinja_renderer_errors_on_slice_step_zero") {
  std::array<emel::jinja::value, 2> arr_vals = {make_int(1), make_int(2)};
  emel::jinja::value arr_val = make_array(arr_vals.data(), arr_vals.size());

  std::array<emel::jinja::object_entry, 1> entries = {};
  entries[0].key = make_string("arr");
  entries[0].val = arr_val;
  emel::jinja::object_value globals{entries.data(), entries.size(), entries.size(), false};

  auto result = render_template("{{ arr[0:2:0] }}", &globals);

  CHECK_FALSE(result.done);
  CHECK(result.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("jinja_renderer_errors_on_missing_filter") {
  auto result = render_template("{{ \"x\"|missing_filter }}");

  CHECK_FALSE(result.done);
  CHECK(result.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("jinja_renderer_errors_on_missing_test") {
  auto result = render_template("{{ 1 is missing_test }}");

  CHECK_FALSE(result.done);
  CHECK(result.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("jinja_renderer_errors_on_call_target") {
  auto result = render_template("{{ 1() }}");

  CHECK_FALSE(result.done);
  CHECK(result.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("jinja_renderer_errors_on_for_invalid_iterable") {
  auto result = render_template("{% for x in 1 %}{{ x }}{% endfor %}");

  CHECK_FALSE(result.done);
  CHECK(result.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("jinja_renderer_errors_on_for_tuple_item") {
  auto result = render_template("{% for a, b in [1,2] %}{{ a }}{% endfor %}");

  CHECK_FALSE(result.done);
  CHECK(result.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("jinja_renderer_errors_on_set_tuple_non_array") {
  auto result = render_template("{% set a, b = 1 %}");

  CHECK_FALSE(result.done);
  CHECK(result.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("jinja_renderer_errors_on_set_tuple_size_mismatch") {
  auto result = render_template("{% set a, b = [1] %}");

  CHECK_FALSE(result.done);
  CHECK(result.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("jinja_renderer_errors_on_set_computed_member") {
  auto result = render_template("{% set obj[\"k\"] = 1 %}");

  CHECK_FALSE(result.done);
  CHECK(result.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("jinja_renderer_errors_on_set_member_non_object") {
  std::array<emel::jinja::object_entry, 1> entries = {};
  entries[0].key = make_string("num");
  entries[0].val = make_int(1);
  emel::jinja::object_value globals{entries.data(), entries.size(), entries.size(), false};

  auto result = render_template("{% set num.key = 1 %}", &globals);

  CHECK_FALSE(result.done);
  CHECK(result.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("jinja_renderer_errors_on_filter_statement_missing_filter") {
  auto result = render_template("{% filter missing %}x{% endfilter %}");

  CHECK_FALSE(result.done);
  CHECK(result.err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("jinja_renderer_detail_unary_paths") {
  using namespace emel::jinja;

  {
    renderer::action::context ctx{};
    renderer::detail::render_io io{};
    char buffer[16] = {};
    renderer::detail::init_writer(io, buffer, sizeof(buffer));

    token op{token_type::AdditiveBinaryOperator, "+", 0};
    auto operand = std::make_unique<integer_literal>(2);
    unary_expression expr{op, std::move(operand)};
    auto value = renderer::detail::eval_expr(ctx, &expr, nullptr, io);

    CHECK(value.type == value_type::integer);
    CHECK(value.int_v == 2);
  }

  {
    renderer::action::context ctx{};
    renderer::detail::render_io io{};
    char buffer[16] = {};
    renderer::detail::init_writer(io, buffer, sizeof(buffer));

    token op{token_type::AdditiveBinaryOperator, "-", 0};
    auto operand = std::make_unique<float_literal>(1.5);
    unary_expression expr{op, std::move(operand)};
    auto value = renderer::detail::eval_expr(ctx, &expr, nullptr, io);

    CHECK(value.type == value_type::floating);
    CHECK(value.float_v == doctest::Approx(-1.5));
  }

  {
    renderer::action::context ctx{};
    renderer::detail::render_io io{};
    char buffer[16] = {};
    renderer::detail::init_writer(io, buffer, sizeof(buffer));

    token op{token_type::AdditiveBinaryOperator, "+", 0};
    auto operand = std::make_unique<string_literal>("a");
    unary_expression expr{op, std::move(operand)};
    auto value = renderer::detail::eval_expr(ctx, &expr, nullptr, io);

    CHECK(value.type == value_type::undefined);
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  }
}

TEST_CASE("jinja_renderer_detail_limit_paths") {
  using namespace emel::jinja;

  {
    renderer::action::context ctx{};
    ctx.steps_remaining = 0;
    CHECK_FALSE(renderer::detail::ensure_steps(ctx, 0));
    CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
  }

  {
    renderer::action::context ctx{};
    renderer::detail::render_io io{};
    io.depth = renderer::action::k_max_capture_depth;
    CHECK_FALSE(renderer::detail::begin_capture(ctx, io));
    CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

    ctx.phase_error = EMEL_OK;
    io.depth = 0;
    emel::jinja::value captured{};
    CHECK_FALSE(renderer::detail::end_capture(ctx, io, captured));
    CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
  }

  {
    renderer::action::context ctx{};
    ctx.string_buffer_used = renderer::action::k_max_string_bytes;
    auto stored = renderer::detail::store_string(ctx, "x");
    CHECK(stored.empty());
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    renderer::action::context ctx{};
    ctx.array_items_used = renderer::action::k_max_array_items - 1;
    std::array<value, 2> items = {value{}, value{}};
    auto array_val = renderer::detail::make_array(ctx, items.data(), items.size());
    CHECK(array_val.type == value_type::undefined);
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    renderer::action::context ctx{};
    ctx.object_entries_used = renderer::action::k_max_object_entries - 1;
    std::array<object_entry, 2> entries = {};
    entries[0].key = value{};
    entries[0].val = value{};
    entries[1].key = value{};
    entries[1].val = value{};
    auto obj_val = renderer::detail::make_object(ctx, entries.data(), entries.size(), entries.size(), false);
    CHECK(obj_val.type == value_type::undefined);
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    renderer::action::context ctx{};
    ctx.scope_count = renderer::action::k_max_scopes;
    CHECK_FALSE(renderer::detail::push_scope(ctx));
    CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

    ctx = renderer::action::context{};
    ctx.object_entries_used = renderer::action::k_max_object_entries - renderer::action::k_scope_capacity + 1;
    CHECK_FALSE(renderer::detail::push_scope(ctx));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    renderer::detail::call_args args{};
    args.pos_count = renderer::detail::k_max_call_args;
    CHECK_FALSE(renderer::detail::add_pos(args, value{}));
    args.kw_count = renderer::detail::k_max_call_args;
    CHECK_FALSE(renderer::detail::add_kw(args, "k", value{}));
  }

  {
    renderer::action::context ctx{};
    renderer::detail::call_args args{};
    args.pos_count = renderer::detail::k_max_call_args;
    value out{};
    CHECK_FALSE(renderer::detail::filter_length(ctx, value{}, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  }
}

TEST_CASE("jinja_renderer_detail_call_statement_errors") {
  using namespace emel::jinja;

  {
    renderer::action::context ctx{};
    renderer::detail::render_io io{};
    char buffer[16] = {};
    renderer::detail::init_writer(io, buffer, sizeof(buffer));

    auto non_call = std::make_unique<integer_literal>(1);
    ast_list caller_args;
    ast_list body;
    call_statement stmt{std::move(non_call), std::move(caller_args), std::move(body)};
    CHECK_FALSE(renderer::detail::render_call_statement(ctx, &stmt, nullptr, io));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    renderer::action::context ctx{};
    renderer::detail::render_io io{};
    char buffer[16] = {};
    renderer::detail::init_writer(io, buffer, sizeof(buffer));

    ast_list args;
    auto callee = std::make_unique<integer_literal>(1);
    auto call_expr = std::make_unique<call_expression>(std::move(callee), std::move(args));
    ast_list caller_args;
    ast_list body;
    call_statement stmt{std::move(call_expr), std::move(caller_args), std::move(body)};
    CHECK_FALSE(renderer::detail::render_call_statement(ctx, &stmt, nullptr, io));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  }
}

TEST_CASE("jinja_renderer_detail_helpers") {
  using namespace emel::jinja;

  {
    renderer::action::context ctx{};
    renderer::detail::render_io io{};
    char buffer[8] = {};
    renderer::detail::init_writer(io, buffer, sizeof(buffer));

    CHECK(renderer::detail::store_string(ctx, "").empty());

    io.depth = 1;
    io.writers[1].data = buffer;
    io.writers[1].capacity = sizeof(buffer);
    io.writers[1].length = 1;
    ctx.phase_error = EMEL_ERR_BACKEND;
    value captured{};
    CHECK_FALSE(renderer::detail::end_capture(ctx, io, captured));

    renderer::detail::make_object(ctx, nullptr, 0, 0, false);
    renderer::detail::pop_scope(ctx);
  }

  {
    value v_none{};
    v_none.type = value_type::none;
    value v_bool{};
    v_bool.type = value_type::boolean;
    v_bool.bool_v = true;
    value v_int{};
    v_int.type = value_type::integer;
    v_int.int_v = 1;
    v_int.float_v = 1.0;
    value v_float{};
    v_float.type = value_type::floating;
    v_float.float_v = 1.5;
    v_float.int_v = 1;
    value v_string{};
    v_string.type = value_type::string;
    v_string.string_v.view = "hi";

    value array_items[2] = {v_int, v_int};
    value v_array{};
    v_array.type = value_type::array;
    v_array.array_v.items = array_items;
    v_array.array_v.count = 2;
    v_array.array_v.capacity = 2;

    object_entry obj_entries[1] = {};
    obj_entries[0].key = v_string;
    obj_entries[0].val = v_int;
    value v_object{};
    v_object.type = value_type::object;
    v_object.object_v.entries = obj_entries;
    v_object.object_v.count = 1;
    v_object.object_v.capacity = 1;

    value v_func{};
    v_func.type = value_type::function;
    v_func.func_v.data = &v_func;

    CHECK_FALSE(renderer::detail::value_is_truthy(v_none));
    CHECK(renderer::detail::value_is_truthy(v_bool));
    CHECK(renderer::detail::value_is_truthy(v_int));
    CHECK(renderer::detail::value_is_truthy(v_float));
    CHECK(renderer::detail::value_is_truthy(v_string));
    CHECK(renderer::detail::value_is_truthy(v_array));
    CHECK(renderer::detail::value_is_truthy(v_object));
    CHECK(renderer::detail::value_is_truthy(v_func));

    value v_float_one{};
    v_float_one.type = value_type::floating;
    v_float_one.float_v = 1.0;
    v_float_one.int_v = 1;
    CHECK(renderer::detail::value_equal(v_int, v_float_one));

    value v_array_short = v_array;
    v_array_short.array_v.count = 1;
    CHECK_FALSE(renderer::detail::value_equal(v_array, v_array_short));

    value v_object_other = v_object;
    v_object_other.object_v.entries = nullptr;
    v_object_other.object_v.count = 0;
    v_object_other.object_v.capacity = 0;
    CHECK_FALSE(renderer::detail::value_equal(v_object, v_object_other));

    value v_func_other{};
    v_func_other.type = value_type::function;
    v_func_other.func_v.data = &v_object;
    CHECK_FALSE(renderer::detail::value_equal(v_func, v_func_other));
  }

  {
    renderer::action::context ctx{};
    renderer::detail::render_io io{};
    char buffer[16] = {};
    renderer::detail::init_writer(io, buffer, sizeof(buffer));

    value v_undef{};
    v_undef.type = value_type::undefined;
    CHECK(renderer::detail::value_to_string(ctx, v_undef).empty());

    value v_arr{};
    v_arr.type = value_type::array;
    CHECK(renderer::detail::value_to_string(ctx, v_arr).empty());

    renderer::detail::call_args args{};
    CHECK(renderer::detail::find_kw(args, "missing") == nullptr);
    CHECK(renderer::detail::get_pos(args, 1) == nullptr);
  }

  {
    renderer::action::context ctx{};
    renderer::detail::call_args args{};
    value out{};
    CHECK_FALSE(renderer::detail::builtin_default(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

    ctx = renderer::action::context{};
    CHECK_FALSE(renderer::detail::builtin_length(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

    ctx = renderer::action::context{};
    CHECK_FALSE(renderer::detail::builtin_range(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    renderer::action::context ctx{};
    renderer::detail::call_args args{};
    value out{};

    args.pos_count = 1;
    args.pos[0].type = value_type::string;
    args.pos[0].string_v.view = "x";
    CHECK_FALSE(renderer::detail::builtin_range(ctx, args, out));

    ctx = renderer::action::context{};
    args.pos_count = 2;
    args.pos[0].type = value_type::integer;
    args.pos[0].int_v = 0;
    args.pos[1].type = value_type::string;
    args.pos[1].string_v.view = "bad";
    CHECK_FALSE(renderer::detail::builtin_range(ctx, args, out));

    ctx = renderer::action::context{};
    args.pos_count = 3;
    args.pos[0].type = value_type::integer;
    args.pos[0].int_v = 0;
    args.pos[1].type = value_type::integer;
    args.pos[1].int_v = 1;
    args.pos[2].type = value_type::integer;
    args.pos[2].int_v = 0;
    CHECK_FALSE(renderer::detail::builtin_range(ctx, args, out));
  }

  {
    renderer::action::context ctx{};
    renderer::detail::call_args args{};
    value out{};

    args.pos_count = 0;
    CHECK_FALSE(renderer::detail::builtin_upper(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

    ctx = renderer::action::context{};
    CHECK_FALSE(renderer::detail::builtin_lower(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

    ctx = renderer::action::context{};
    CHECK_FALSE(renderer::detail::builtin_trim(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

    ctx = renderer::action::context{};
    CHECK_FALSE(renderer::detail::builtin_replace(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

    ctx = renderer::action::context{};
    CHECK_FALSE(renderer::detail::builtin_join(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    renderer::action::context ctx{};
    renderer::detail::render_io io{};
    char buffer[32] = {};
    renderer::detail::init_writer(io, buffer, sizeof(buffer));

    value items[2] = {};
    items[0].type = value_type::integer;
    items[0].int_v = 1;
    items[1].type = value_type::integer;
    items[1].int_v = 2;

    value arr{};
    arr.type = value_type::array;
    arr.array_v.items = items;
    arr.array_v.count = 2;
    arr.array_v.capacity = 2;

    value obj{};
    obj.type = value_type::object;

    CHECK(renderer::detail::write_value(ctx, io, arr));
    CHECK(renderer::detail::write_value(ctx, io, obj));
  }

  {
    renderer::action::context ctx{};
    renderer::detail::call_args args{};
    value out{};

    args.kw_count = 1;
    args.kw[0].key = "k";
    args.kw[0].val.type = value_type::integer;
    args.kw[0].val.int_v = 1;
    CHECK(renderer::detail::find_kw(args, "k") != nullptr);

    args.pos_count = 1;
    args.pos[0].type = value_type::integer;
    args.pos[0].int_v = 3;
    CHECK(renderer::detail::get_pos(args, 0) != nullptr);

    CHECK(renderer::detail::builtin_namespace(ctx, args, out));
    CHECK(out.type == value_type::object);
  }

  {
    renderer::action::context ctx{};
    renderer::detail::call_args args{};
    value out{};

    args.pos_count = 1;
    args.pos[0].type = value_type::string;
    args.pos[0].string_v.view = "hi";
    CHECK(renderer::detail::builtin_upper(ctx, args, out));

    ctx = renderer::action::context{};
    CHECK(renderer::detail::builtin_lower(ctx, args, out));

    ctx = renderer::action::context{};
    CHECK(renderer::detail::builtin_trim(ctx, args, out));

    ctx = renderer::action::context{};
    args.pos_count = 3;
    args.pos[0].type = value_type::string;
    args.pos[0].string_v.view = "aba";
    args.pos[1].type = value_type::string;
    args.pos[1].string_v.view = "a";
    args.pos[2].type = value_type::string;
    args.pos[2].string_v.view = "x";
    CHECK(renderer::detail::builtin_replace(ctx, args, out));

    ctx = renderer::action::context{};
    args.pos_count = 1;
    args.pos[0] = out;
    CHECK(renderer::detail::builtin_join(ctx, args, out));
  }

  {
    renderer::action::context ctx{};
    renderer::detail::call_args args{};
    value out{};

    args.pos_count = 1;
    args.pos[0].type = value_type::integer;
    args.pos[0].int_v = 3;
    CHECK(renderer::detail::builtin_range(ctx, args, out));

    ctx = renderer::action::context{};
    args.pos_count = 2;
    args.pos[0].type = value_type::integer;
    args.pos[0].int_v = 1;
    args.pos[1].type = value_type::integer;
    args.pos[1].int_v = 3;
    CHECK(renderer::detail::builtin_range(ctx, args, out));

    ctx = renderer::action::context{};
    args.pos_count = 3;
    args.pos[0].type = value_type::integer;
    args.pos[0].int_v = 3;
    args.pos[1].type = value_type::integer;
    args.pos[1].int_v = 0;
    args.pos[2].type = value_type::integer;
    args.pos[2].int_v = -1;
    CHECK(renderer::detail::builtin_range(ctx, args, out));
  }

  {
    renderer::action::context ctx{};
    renderer::detail::call_args args{};
    value out{};

    args.pos_count = 2;
    args.pos[0].type = value_type::integer;
    args.pos[0].int_v = 0;
    args.pos[1].type = value_type::integer;
    args.pos[1].int_v = static_cast<int64_t>(renderer::action::k_max_array_items + 1);
    CHECK_FALSE(renderer::detail::builtin_range(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    renderer::action::context ctx{};
    renderer::detail::call_args args{};
    value out{};

    value input{};
    input.type = value_type::string;
    input.string_v.view = "hi";
    args.pos_count = 1;
    args.pos[0] = input;
    CHECK(renderer::detail::filter_default(ctx, input, args, out));

    ctx = renderer::action::context{};
    args.pos_count = 0;
    CHECK_FALSE(renderer::detail::filter_default(ctx, input, args, out));

    ctx = renderer::action::context{};
    args.pos_count = 0;
    CHECK(renderer::detail::filter_upper(ctx, input, args, out));

    ctx = renderer::action::context{};
    CHECK(renderer::detail::filter_lower(ctx, input, args, out));

    ctx = renderer::action::context{};
    CHECK(renderer::detail::filter_trim(ctx, input, args, out));

    ctx = renderer::action::context{};
    CHECK(renderer::detail::filter_length(ctx, input, args, out));

    ctx = renderer::action::context{};
    args.pos_count = 2;
    args.pos[0] = input;
    args.pos[1] = input;
    CHECK(renderer::detail::filter_replace(ctx, input, args, out));

    ctx = renderer::action::context{};
    args.pos_count = 0;
    CHECK(renderer::detail::filter_join(ctx, input, args, out));
  }

  {
    renderer::action::context ctx{};
    renderer::detail::call_args args{};
    value out{};

    std::string too_long(renderer::action::k_max_string_bytes + 1, 'a');
    args.pos_count = 1;
    args.pos[0].type = value_type::string;
    args.pos[0].string_v.view = too_long;
    CHECK_FALSE(renderer::detail::builtin_upper(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

    ctx = renderer::action::context{};
    CHECK_FALSE(renderer::detail::builtin_lower(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    renderer::action::context ctx{};
    renderer::detail::call_args args{};
    value out{};

    ctx.phase_error = EMEL_ERR_BACKEND;
    args.pos_count = 1;
    args.pos[0].type = value_type::string;
    args.pos[0].string_v.view = "x";
    CHECK_FALSE(renderer::detail::builtin_trim(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
  }

  {
    renderer::action::context ctx{};
    renderer::detail::call_args args{};
    value out{};

    ctx.phase_error = EMEL_ERR_BACKEND;
    args.pos_count = 1;
    args.pos[0].type = value_type::string;
    args.pos[0].string_v.view = "x";
    CHECK_FALSE(renderer::detail::builtin_upper(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

    ctx.phase_error = EMEL_ERR_BACKEND;
    CHECK_FALSE(renderer::detail::builtin_lower(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

    args.pos_count = 3;
    args.pos[0].type = value_type::string;
    args.pos[0].string_v.view = "abc";
    args.pos[1].type = value_type::string;
    args.pos[1].string_v.view = "a";
    args.pos[2].type = value_type::string;
    args.pos[2].string_v.view = "b";
    ctx.phase_error = EMEL_ERR_BACKEND;
    CHECK_FALSE(renderer::detail::builtin_replace(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_BACKEND);
  }

  {
    renderer::action::context ctx{};
    renderer::detail::call_args args{};
    value out{};

    value part{};
    part.type = value_type::string;
    part.string_v.view = "a";
    value parts[2] = {part, part};
    value arr{};
    arr.type = value_type::array;
    arr.array_v.items = parts;
    arr.array_v.count = 2;
    arr.array_v.capacity = 2;

    std::string huge_sep(renderer::action::k_max_string_bytes + 1, 'b');
    value delim{};
    delim.type = value_type::string;
    delim.string_v.view = huge_sep;

    args.pos_count = 2;
    args.pos[0] = arr;
    args.pos[1] = delim;
    ctx.phase_error = EMEL_ERR_BACKEND;
    CHECK_FALSE(renderer::detail::builtin_join(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_BACKEND);

    ctx = renderer::action::context{};
    ctx.string_buffer_used = renderer::action::k_max_string_bytes;
    parts[0].type = value_type::integer;
    parts[0].int_v = 1;
    parts[0].float_v = 1.0;
    parts[1].type = value_type::integer;
    parts[1].int_v = 2;
    parts[1].float_v = 2.0;
    args.pos[0] = arr;
    args.pos[1].type = value_type::string;
    args.pos[1].string_v.view = " ";
    CHECK_FALSE(renderer::detail::builtin_join(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

    ctx = renderer::action::context{};
    args.pos[0] = arr;
    args.pos[1] = delim;
    CHECK_FALSE(renderer::detail::builtin_join(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);

    ctx = renderer::action::context{};
    std::string huge_part(renderer::action::k_max_string_bytes + 1, 'c');
    part.string_v.view = huge_part;
    parts[0] = part;
    parts[1] = part;
    args.pos_count = 1;
    args.pos[0] = arr;
    CHECK_FALSE(renderer::detail::builtin_join(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    renderer::action::context ctx{};
    renderer::detail::call_args args{};
    value out{};

    args.pos_count = 3;
    args.pos[0].type = value_type::string;
    args.pos[0].string_v.view = "abc";
    args.pos[1].type = value_type::string;
    args.pos[1].string_v.view = "";
    args.pos[2].type = value_type::string;
    args.pos[2].string_v.view = "z";
    CHECK(renderer::detail::builtin_replace(ctx, args, out));
    CHECK(out.type == value_type::string);
    CHECK(out.string_v.view == "abc");
  }

  {
    renderer::action::context ctx{};
    renderer::detail::call_args args{};
    value out{};

    std::string huge_input(renderer::action::k_max_string_bytes + 1, 'a');
    args.pos_count = 3;
    args.pos[0].type = value_type::string;
    args.pos[0].string_v.view = huge_input;
    args.pos[1].type = value_type::string;
    args.pos[1].string_v.view = "x";
    args.pos[2].type = value_type::string;
    args.pos[2].string_v.view = "b";
    CHECK_FALSE(renderer::detail::builtin_replace(ctx, args, out));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    value invalid{};
    invalid.type = static_cast<value_type>(99);
    CHECK_FALSE(renderer::detail::value_is_truthy(invalid));
  }

  {
    renderer::action::context ctx{};
    object_entry entries[1] = {};
    value v_key{};
    v_key.type = value_type::string;
    v_key.string_v.view = "k";
    value v_val{};
    v_val.type = value_type::integer;
    v_val.int_v = 1;
    object_value obj{entries, 0, 0, false};
    CHECK_FALSE(renderer::detail::set_object_value(ctx, obj, "k", v_val));
    CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    renderer::action::context ctx{};
    renderer::detail::render_io io{};
    char buffer[1] = {};
    renderer::detail::init_writer(io, buffer, sizeof(buffer));

    value v_str{};
    v_str.type = value_type::string;
    v_str.string_v.view = "long";
    value arr{};
    arr.type = value_type::array;
    arr.array_v.items = &v_str;
    arr.array_v.count = 1;
    arr.array_v.capacity = 1;
    CHECK_FALSE(renderer::detail::write_value(ctx, io, arr));
  }

  {
    renderer::detail::call_args args{};
    args.kw_count = 1;
    args.kw[0].key = "a";
    args.kw[0].val.type = value_type::integer;
    args.kw[0].val.int_v = 1;
    CHECK(renderer::detail::find_kw(args, "b") == nullptr);
  }

  {
    renderer::action::context ctx{};
    renderer::detail::call_args args{};
    value out{};
    args.pos_count = 1;
    args.pos[0].type = value_type::boolean;
    args.pos[0].bool_v = true;
    CHECK(renderer::detail::builtin_length(ctx, args, out));

    ctx = renderer::action::context{};
    args.pos_count = 3;
    args.pos[0].type = value_type::integer;
    args.pos[1].type = value_type::integer;
    args.pos[2].type = value_type::string;
    CHECK_FALSE(renderer::detail::builtin_range(ctx, args, out));

    ctx = renderer::action::context{};
    args.kw_count = 0;
    CHECK(renderer::detail::builtin_namespace(ctx, args, out));
  }
}

TEST_CASE("jinja_renderer_detail_value_equal_cases") {
  using namespace emel::jinja;

  value undef1{};
  value undef2{};
  undef1.type = value_type::undefined;
  undef2.type = value_type::undefined;
  CHECK(renderer::detail::value_equal(undef1, undef2));

  value b1{};
  value b2{};
  b1.type = value_type::boolean;
  b1.bool_v = true;
  b2.type = value_type::boolean;
  b2.bool_v = false;
  CHECK_FALSE(renderer::detail::value_equal(b1, b2));

  value f1{};
  value f2{};
  f1.type = value_type::floating;
  f1.float_v = 1.5;
  f2.type = value_type::floating;
  f2.float_v = 1.5;
  CHECK(renderer::detail::value_equal(f1, f2));

  value i1{};
  value s1{};
  i1.type = value_type::integer;
  i1.int_v = 1;
  s1.type = value_type::string;
  s1.string_v.view = "a";
  CHECK_FALSE(renderer::detail::value_equal(i1, s1));

  value arr_items_a[2] = {};
  value arr_items_b[2] = {};
  arr_items_a[0] = i1;
  arr_items_a[1] = i1;
  arr_items_b[0] = i1;
  arr_items_b[1] = s1;
  value arr_a{};
  value arr_b{};
  arr_a.type = value_type::array;
  arr_a.array_v.items = arr_items_a;
  arr_a.array_v.count = 2;
  arr_a.array_v.capacity = 2;
  arr_b.type = value_type::array;
  arr_b.array_v.items = arr_items_b;
  arr_b.array_v.count = 2;
  arr_b.array_v.capacity = 2;
  CHECK_FALSE(renderer::detail::value_equal(arr_a, arr_b));

  object_entry obj_entries_a[1] = {};
  object_entry obj_entries_b[1] = {};
  obj_entries_a[0].key = s1;
  obj_entries_a[0].val = i1;
  obj_entries_b[0].key = s1;
  obj_entries_b[0].val = b1;
  value obj_a{};
  value obj_b{};
  obj_a.type = value_type::object;
  obj_a.object_v.entries = obj_entries_a;
  obj_a.object_v.count = 1;
  obj_a.object_v.capacity = 1;
  obj_b.type = value_type::object;
  obj_b.object_v.entries = obj_entries_b;
  obj_b.object_v.count = 1;
  obj_b.object_v.capacity = 1;
  CHECK_FALSE(renderer::detail::value_equal(obj_a, obj_b));

  object_entry obj_entries_c[1] = {};
  object_entry obj_entries_d[1] = {};
  obj_entries_c[0].key = s1;
  obj_entries_c[0].val = i1;
  value s2{};
  s2.type = value_type::string;
  s2.string_v.view = "b";
  obj_entries_d[0].key = s2;
  obj_entries_d[0].val = i1;
  value obj_c{};
  value obj_d{};
  obj_c.type = value_type::object;
  obj_c.object_v.entries = obj_entries_c;
  obj_c.object_v.count = 1;
  obj_c.object_v.capacity = 1;
  obj_d.type = value_type::object;
  obj_d.object_v.entries = obj_entries_d;
  obj_d.object_v.count = 1;
  obj_d.object_v.capacity = 1;
  CHECK_FALSE(renderer::detail::value_equal(obj_c, obj_d));

  value obj_e = obj_a;
  value obj_f = obj_a;
  CHECK(renderer::detail::value_equal(obj_e, obj_f));

  value func_a{};
  value func_b{};
  func_a.type = value_type::function;
  func_a.func_v.data = &func_a;
  func_a.func_v.kind = function_kind::builtin;
  func_b.type = value_type::function;
  func_b.func_v.data = &func_a;
  func_b.func_v.kind = function_kind::builtin;
  CHECK(renderer::detail::value_equal(func_a, func_b));

  value invalid_a{};
  value invalid_b{};
  invalid_a.type = static_cast<value_type>(99);
  invalid_b.type = static_cast<value_type>(99);
  CHECK_FALSE(renderer::detail::value_equal(invalid_a, invalid_b));
}
