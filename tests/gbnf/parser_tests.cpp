#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include <string>
#include <string_view>

#include "emel/gbnf/rule_parser/detail.hpp"
#include "emel/gbnf/rule_parser/errors.hpp"
#include "emel/gbnf/rule_parser/events.hpp"
#include "emel/gbnf/rule_parser/guards.hpp"
#include "emel/gbnf/rule_parser/sm.hpp"

namespace {

struct done_capture {
  bool called = false;
  emel::gbnf::grammar *grammar = nullptr;
};

struct error_capture {
  bool called = false;
  emel::gbnf::grammar *grammar = nullptr;
  int32_t err = 0;
};

bool dispatch_done_test(void *owner, const emel::gbnf::rule_parser::events::parsing_done &ev) {
  auto *capture = static_cast<done_capture *>(owner);
  capture->called = true;
  capture->grammar = &ev.grammar;
  return true;
}

bool dispatch_error_test(void *owner, const emel::gbnf::rule_parser::events::parsing_error &ev) {
  auto *capture = static_cast<error_capture *>(owner);
  capture->called = true;
  capture->grammar = &ev.grammar;
  capture->err = ev.err;
  return true;
}

int32_t parser_error_code(const emel::gbnf::rule_parser::error err) {
  return static_cast<int32_t>(emel::error::cast(err));
}

emel::gbnf::rule_parser::event::parse make_parse_event(std::string_view grammar_text,
                                                   emel::gbnf::grammar *grammar,
                                                   done_capture *done,
                                                   error_capture *error,
                                                   const bool include_done,
                                                   const bool include_error) {
  emel::gbnf::rule_parser::event::parse ev{};
  ev.grammar_text = grammar_text;
  ev.grammar_out = grammar;
  if (include_done) {
    ev.dispatch_done = emel::callback<bool(const emel::gbnf::rule_parser::events::parsing_done &)>{
        done,
        dispatch_done_test};
  }
  if (include_error) {
    ev.dispatch_error = emel::callback<bool(const emel::gbnf::rule_parser::events::parsing_error &)>{
        error,
        dispatch_error_test};
  }
  return ev;
}

} // namespace

TEST_CASE("gbnf_parser_starts_ready") {
  emel::gbnf::rule_parser::sm machine{};
  CHECK(machine.is(boost::sml::state<emel::gbnf::rule_parser::ready>));
}

TEST_CASE("gbnf_parser_valid_parse_dispatches_done") {
  emel::gbnf::rule_parser::sm machine{};
  emel::gbnf::grammar grammar{};
  done_capture done{};
  error_capture error{};

  const auto ev = make_parse_event(
      "root ::= [a-z]+", &grammar, &done, &error, true, true);

  CHECK(machine.process_event(ev));
  CHECK(done.called);
  CHECK(done.grammar == &grammar);
  CHECK_FALSE(error.called);
  CHECK(grammar.rule_count > 0);
  CHECK(grammar.element_count > 0);
  CHECK(machine.is(boost::sml::state<emel::gbnf::rule_parser::ready>));
}

TEST_CASE("gbnf_parser_invalid_request_dispatches_invalid_request_error") {
  emel::gbnf::rule_parser::sm machine{};
  emel::gbnf::grammar grammar{};
  done_capture done{};
  error_capture error{};

  const auto ev = make_parse_event(
      "root ::= \"a\"", &grammar, &done, &error, false, true);

  CHECK_FALSE(machine.process_event(ev));
  CHECK_FALSE(done.called);
  CHECK(error.called);
  CHECK(error.grammar == &grammar);
  CHECK(error.err == parser_error_code(emel::gbnf::rule_parser::error::invalid_request));
  CHECK(grammar.rule_count == 0);
  CHECK(grammar.element_count == 0);
}

TEST_CASE("gbnf_parser_invalid_request_without_error_callback_returns_false") {
  emel::gbnf::rule_parser::sm machine{};
  emel::gbnf::grammar grammar{};
  done_capture done{};
  error_capture error{};

  const auto ev = make_parse_event(
      "root ::= \"a\"", &grammar, &done, &error, true, false);

  CHECK_FALSE(machine.process_event(ev));
  CHECK_FALSE(done.called);
  CHECK_FALSE(error.called);
  CHECK(grammar.rule_count == 0);
  CHECK(grammar.element_count == 0);
}

TEST_CASE("gbnf_parser_syntax_error_dispatches_parse_failed") {
  emel::gbnf::rule_parser::sm machine{};
  emel::gbnf::grammar grammar{};
  done_capture done{};
  error_capture error{};

  const auto ev = make_parse_event(
      "root ::= <abc>", &grammar, &done, &error, true, true);

  CHECK_FALSE(machine.process_event(ev));
  CHECK_FALSE(done.called);
  CHECK(error.called);
  CHECK(error.grammar == &grammar);
  CHECK(error.err == parser_error_code(emel::gbnf::rule_parser::error::parse_failed));
}

TEST_CASE("gbnf_parser_state_machine_parses_complex_grammar") {
  const std::string_view grammar_text = R"g(# comment
root ::= ("ab" | [a-z] | [^0-9] | <[3]> | !<[4]> | . | name-ref)+
name-ref ::= "\x41\u0042\u00000043" {2,3} "t\n\r\t\\"?
range-rule ::= [a-zA-z0-9]*
)g";

  emel::gbnf::rule_parser::sm machine{};
  emel::gbnf::grammar grammar{};
  done_capture done{};
  error_capture error{};

  const auto ev = make_parse_event(grammar_text, &grammar, &done, &error, true, true);

  CHECK(machine.process_event(ev));
  CHECK(done.called);
  CHECK(done.grammar == &grammar);
  CHECK_FALSE(error.called);
  CHECK(grammar.rule_count >= 3);
  CHECK(grammar.element_count > 0);
}

TEST_CASE("gbnf_parser_reuse_resets_output_grammar") {
  emel::gbnf::rule_parser::sm machine{};
  emel::gbnf::grammar grammar{};
  done_capture done{};
  error_capture error{};

  const auto first = make_parse_event("root ::= [a-z]+", &grammar, &done, &error, true, true);
  CHECK(machine.process_event(first));
  CHECK(done.called);
  CHECK_FALSE(error.called);
  const uint32_t first_element_count = grammar.element_count;
  CHECK(first_element_count > 0);

  done = {};
  error = {};
  const auto second = make_parse_event(
      "root ::= \"x\"", &grammar, &done, &error, true, true);
  CHECK(machine.process_event(second));
  CHECK(done.called);
  CHECK_FALSE(error.called);
  CHECK(grammar.element_count > 0);
  CHECK(grammar.element_count <= first_element_count + 2u);
}

TEST_CASE("gbnf_rule_builder_overflow_and_resize") {
  emel::gbnf::rule_parser::detail::rule_builder builder{};
  const emel::gbnf::element elem{emel::gbnf::element_type::end, 0};
  builder.size = emel::gbnf::k_max_gbnf_rule_elements;
  CHECK_FALSE(builder.push(elem));

  builder.size = emel::gbnf::k_max_gbnf_rule_elements - 1;
  CHECK_FALSE(builder.append(&elem, 2));
  CHECK(builder.append(&elem, 0));

  builder.size = 1;
  CHECK_FALSE(builder.resize(2));
}

TEST_CASE("gbnf_symbol_table_insert_find_and_full_probe") {
  emel::gbnf::rule_parser::detail::symbol_table table{};
  table.clear();
  const std::string_view name = "root";
  const uint32_t hash = emel::gbnf::rule_parser::detail::symbol_table::hash_name(name);
  uint32_t id = 0;
  CHECK_FALSE(table.find(name, hash, id));
  CHECK(table.insert(name, hash, 1));
  CHECK(table.find(name, hash, id));
  CHECK(id == 1);
  CHECK(table.insert(name, hash, 2));

  for (auto &entry : table.entries) {
    entry.name = "x";
    entry.hash = 1;
    entry.id = 3;
    entry.occupied = true;
  }
  const std::string_view missing = "missing";
  const uint32_t missing_hash = emel::gbnf::rule_parser::detail::symbol_table::hash_name(missing);
  uint32_t missing_id = 0;
  CHECK_FALSE(table.find(missing, missing_hash, missing_id));
  CHECK_FALSE(table.insert(missing, missing_hash, 4));
}

TEST_CASE("gbnf_parser_helpers_error_paths") {
  namespace detail = emel::gbnf::rule_parser::detail;
  uint64_t value = 0;
  const char *next = nullptr;

  const char *bad_digit = "x";
  CHECK_FALSE(detail::parse_uint64(bad_digit, bad_digit + 1, value, &next));

  constexpr char overflow[] = "18446744073709551616";
  CHECK_FALSE(detail::parse_uint64(overflow, overflow + sizeof(overflow) - 1, value, &next));

  const char *hex_short = "A";
  auto hex_res = detail::parse_hex(hex_short, hex_short + 1, 2);
  CHECK(hex_res.second == nullptr);

  const char *hex_ok = "aF";
  auto hex_ok_res = detail::parse_hex(hex_ok, hex_ok + 2, 2);
  CHECK(hex_ok_res.second == hex_ok + 2);

  const char *utf8_short = "\xF0";
  auto utf8_res = detail::decode_utf8(utf8_short, utf8_short + 1);
  CHECK(utf8_res.second == nullptr);

  const char *utf8_empty = "";
  auto utf8_empty_res = detail::decode_utf8(utf8_empty, utf8_empty);
  CHECK(utf8_empty_res.second == nullptr);

  const char *utf8_two = "\xC2\xA9";
  auto utf8_two_res = detail::decode_utf8(utf8_two, utf8_two + 2);
  CHECK(utf8_two_res.second == utf8_two + 2);

  const char *name_bad = "!";
  CHECK(detail::parse_name(name_bad, name_bad + 1) == nullptr);

  const char *escape_short = "\\";
  auto esc_short = detail::parse_char(escape_short, escape_short + 1);
  CHECK(esc_short.second == nullptr);

  const char *char_empty = "";
  auto char_empty_res = detail::parse_char(char_empty, char_empty);
  CHECK(char_empty_res.second == nullptr);

  const char *escape_bad = "\\q";
  auto esc_bad = detail::parse_char(escape_bad, escape_bad + 2);
  CHECK(esc_bad.second == nullptr);
}

TEST_CASE("gbnf_grammar_rule_view_bounds") {
  emel::gbnf::grammar grammar{};
  grammar.rule_count = 1;
  grammar.rule_offsets[0] = 0;
  grammar.rule_lengths[0] = 0;
  grammar.element_count = 0;
  CHECK(grammar.rule(1).length == 0);
  CHECK(grammar.rule(0).length == 0);

  grammar.rule_lengths[0] = 2;
  grammar.element_count = 1;
  CHECK(grammar.rule(0).length == 0);
}

TEST_CASE("gbnf_parser_error_guards_classify_explicit_errors") {
  emel::gbnf::grammar grammar{};
  grammar.rule_count = 1;
  emel::gbnf::rule_parser::event::parse request{};
  request.grammar_out = &grammar;
  emel::gbnf::rule_parser::event::parse_rules_ctx parse_ctx{};
  emel::gbnf::rule_parser::event::parse_rules ev{request, parse_ctx};
  emel::gbnf::rule_parser::action::context ctx{};

  parse_ctx.err = emel::error::cast(emel::gbnf::rule_parser::error::none);
  CHECK(emel::gbnf::rule_parser::guard::parse_error_none{}(ev, ctx));

  parse_ctx.err = emel::error::cast(emel::gbnf::rule_parser::error::invalid_request);
  CHECK(emel::gbnf::rule_parser::guard::parse_error_invalid_request{}(ev, ctx));

  parse_ctx.err = emel::error::cast(emel::gbnf::rule_parser::error::parse_failed);
  CHECK(emel::gbnf::rule_parser::guard::parse_error_parse_failed{}(ev, ctx));

  parse_ctx.err = emel::error::cast(emel::gbnf::rule_parser::error::internal_error);
  CHECK(emel::gbnf::rule_parser::guard::parse_error_internal_error{}(ev, ctx));

  parse_ctx.err = emel::error::cast(emel::gbnf::rule_parser::error::untracked);
  CHECK(emel::gbnf::rule_parser::guard::parse_error_untracked{}(ev, ctx));

  parse_ctx.err = static_cast<emel::error::type>(0x40000000u);
  CHECK(emel::gbnf::rule_parser::guard::parse_error_unknown{}(ev, ctx));

  parse_ctx.err = emel::error::cast(emel::gbnf::rule_parser::error::none);
  ctx.next_symbol_id = 1u;
  ctx.rule_defined[0] = true;
  CHECK(emel::gbnf::rule_parser::guard::eof_can_finalize_symbols{}(ev, ctx));
  CHECK_FALSE(emel::gbnf::rule_parser::guard::eof_cannot_finalize_symbols{}(ev, ctx));

  grammar.rule_count = 0u;
  CHECK_FALSE(emel::gbnf::rule_parser::guard::eof_can_finalize_symbols{}(ev, ctx));
  CHECK(emel::gbnf::rule_parser::guard::eof_cannot_finalize_symbols{}(ev, ctx));
}
