#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/gbnf/parser/detail.hpp"
#include "emel/gbnf/parser/events.hpp"
#include "emel/gbnf/parser/sm.hpp"

namespace {

bool dispatch_done_test(void *owner, const emel::gbnf::events::parsing_done &) {
  *static_cast<bool *>(owner) = true;
  return true;
}

bool dispatch_error_test(void *owner,
                         const emel::gbnf::events::parsing_error &) {
  *static_cast<bool *>(owner) = true;
  return true;
}

} // namespace

TEST_CASE("gbnf_parser_starts_initialized") {
  emel::gbnf::parser::action::context ctx{};
  emel::gbnf::parser::sm machine{ctx};
  CHECK(machine.is(boost::sml::state<emel::gbnf::parser::initialized>));
}

TEST_CASE("gbnf_parser_valid_parse_reaches_done") {
  emel::gbnf::parser::action::context ctx{};
  emel::gbnf::parser::sm machine{ctx};
  int32_t error = -1;
  emel::gbnf::grammar grammar{};
  bool done_called = false;

  ::emel::gbnf::event::parse ev{
      .grammar_text = "root ::= [a-z]+",
      .grammar_out = &grammar,
      .error_out = &error,
      .owner_sm = &done_called,
      .dispatch_done =
          ::emel::callback<bool(const ::emel::gbnf::events::parsing_done &)>(
              &done_called, dispatch_done_test)};

  CHECK(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::gbnf::parser::done>));
  CHECK(done_called);
  CHECK(error == EMEL_OK);
  CHECK(grammar.rule_count > 0);
  CHECK(grammar.element_count > 0);
  CHECK(grammar.rule(0).length > 0);
}

TEST_CASE("gbnf_parser_invalid_parse_reaches_errored") {
  emel::gbnf::parser::action::context ctx{};
  emel::gbnf::parser::sm machine{ctx};
  int32_t error = EMEL_OK;
  emel::gbnf::grammar grammar{};
  bool error_called = false;

  ::emel::gbnf::event::parse ev{
      .grammar_text = "", // invalid: empty text
      .grammar_out = &grammar,
      .error_out = &error,
      .owner_sm = &error_called,
      .dispatch_error =
          ::emel::callback<bool(const ::emel::gbnf::events::parsing_error &)>(
              &error_called, dispatch_error_test)};

  CHECK_FALSE(machine.process_event(ev));
  CHECK(machine.is(boost::sml::state<emel::gbnf::parser::errored>));
  CHECK(error_called);
  CHECK(error == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(grammar.rule_count == 0);
  CHECK(grammar.element_count == 0);
}

TEST_CASE("gbnf_detail_parser_parses_complex_grammar") {
  emel::gbnf::parser::action::context ctx{};
  emel::gbnf::grammar grammar_out{};
  emel::gbnf::parser::detail::recursive_descent_parser parser{ctx, &grammar_out};
  const std::string_view grammar = R"g(# comment
root ::= ("ab" | [a-z] | [^0-9] | <[3]> | !<[4]> | . | name-ref)+
name-ref ::= "\x41\u0042\u00000043" {2,3} "t\n\r\t\\"?
range-rule ::= [a-zA-z0-9]*
)g";
  CHECK(parser.parse(grammar));
  CHECK(grammar_out.rule_count >= 3);
  CHECK(grammar_out.element_count > 0);
}

TEST_CASE("gbnf_detail_parser_rejects_invalid_hex_escape") {
  emel::gbnf::parser::action::context ctx{};
  emel::gbnf::grammar grammar_out{};
  emel::gbnf::parser::detail::recursive_descent_parser parser{ctx, &grammar_out};
  ctx.phase_error = EMEL_OK;
  CHECK_FALSE(parser.parse("root ::= \"\\xZZ\""));
  CHECK(ctx.phase_error == EMEL_ERR_PARSE_FAILED);
}

TEST_CASE("gbnf_detail_parser_rejects_invalid_token_reference") {
  emel::gbnf::parser::action::context ctx{};
  emel::gbnf::grammar grammar_out{};
  emel::gbnf::parser::detail::recursive_descent_parser parser{ctx, &grammar_out};
  ctx.phase_error = EMEL_OK;
  CHECK_FALSE(parser.parse("root ::= <abc>"));
  CHECK(ctx.phase_error == EMEL_ERR_PARSE_FAILED);
}

TEST_CASE("gbnf_detail_parser_rejects_large_repetitions") {
  emel::gbnf::parser::action::context ctx{};
  emel::gbnf::grammar grammar_out{};
  emel::gbnf::parser::detail::recursive_descent_parser parser{ctx, &grammar_out};
  ctx.phase_error = EMEL_OK;
  CHECK_FALSE(parser.parse("root ::= \"a\"{2001}"));
  CHECK(ctx.phase_error == EMEL_ERR_PARSE_FAILED);
}

TEST_CASE("gbnf_parser_guards_and_actions_cover_branches") {
  emel::gbnf::parser::action::context ctx{};
  emel::gbnf::grammar grammar{};
  int32_t err = EMEL_OK;
  bool done_called = false;
  bool error_called = false;

  ::emel::gbnf::event::parse ev{
      .grammar_text = "root ::= \"a\"",
      .grammar_out = &grammar,
      .error_out = &err,
      .owner_sm = &done_called,
      .dispatch_done =
          ::emel::callback<bool(const ::emel::gbnf::events::parsing_done &)>(
              &done_called, dispatch_done_test),
      .dispatch_error =
          ::emel::callback<bool(const ::emel::gbnf::events::parsing_error &)>(
              &error_called, dispatch_error_test)};

  CHECK(emel::gbnf::parser::guard::valid_parse{}(ev, ctx));
  ::emel::gbnf::event::parse missing_outputs{
      .grammar_text = "root ::= \"a\"",
      .grammar_out = nullptr,
      .error_out = &err,
  };
  CHECK_FALSE(emel::gbnf::parser::guard::valid_parse{}(missing_outputs, ctx));
  ::emel::gbnf::event::parse invalid_ev{
      .grammar_text = "",
      .grammar_out = nullptr,
      .error_out = nullptr,
  };
  CHECK(emel::gbnf::parser::guard::invalid_parse{}(invalid_ev, ctx));

  ctx.phase_error = EMEL_OK;
  CHECK(emel::gbnf::parser::guard::phase_ok{}(ctx));
  ctx.phase_error = EMEL_ERR_PARSE_FAILED;
  CHECK(emel::gbnf::parser::guard::phase_failed{}(ctx));

  emel::gbnf::parser::action::run_parse(ev, ctx);
  CHECK(done_called);

  ::emel::gbnf::event::parse invalid_ev_parse{
      .grammar_text = "root ::= <abc>",
      .grammar_out = &grammar,
      .error_out = &err,
      .owner_sm = &error_called,
      .dispatch_error =
          ::emel::callback<bool(const ::emel::gbnf::events::parsing_error &)>(
              &error_called, dispatch_error_test)};
  emel::gbnf::parser::action::run_parse(invalid_ev_parse, ctx);
  CHECK(error_called);

  emel::gbnf::parser::action::on_unexpected(ev, ctx);
  CHECK(ctx.last_error == EMEL_ERR_BACKEND);
}

TEST_CASE("gbnf_rule_builder_overflow_and_resize") {
  emel::gbnf::parser::detail::rule_builder builder{};
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
  emel::gbnf::parser::detail::symbol_table table{};
  table.clear();
  const std::string_view name = "root";
  const uint32_t hash = emel::gbnf::parser::detail::symbol_table::hash_name(name);
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
  const uint32_t missing_hash =
      emel::gbnf::parser::detail::symbol_table::hash_name(missing);
  uint32_t missing_id = 0;
  CHECK_FALSE(table.find(missing, missing_hash, missing_id));
  CHECK_FALSE(table.insert(missing, missing_hash, 4));
}

TEST_CASE("gbnf_parser_helpers_error_paths") {
  using parser = emel::gbnf::parser::detail::recursive_descent_parser;
  uint64_t value = 0;
  const char *next = nullptr;

  const char *bad_digit = "x";
  CHECK_FALSE(parser::parse_uint64(bad_digit, bad_digit + 1, value, &next));

  constexpr char overflow[] = "18446744073709551616";
  CHECK_FALSE(parser::parse_uint64(overflow, overflow + sizeof(overflow) - 1, value, &next));

  const char *hex_short = "A";
  auto hex_res = parser::parse_hex(hex_short, hex_short + 1, 2);
  CHECK(hex_res.second == nullptr);

  const char *hex_ok = "aF";
  auto hex_ok_res = parser::parse_hex(hex_ok, hex_ok + 2, 2);
  CHECK(hex_ok_res.second == hex_ok + 2);

  const char *utf8_short = "\xF0";
  auto utf8_res = parser::decode_utf8(utf8_short, utf8_short + 1);
  CHECK(utf8_res.second == nullptr);

  const char *utf8_empty = "";
  auto utf8_empty_res = parser::decode_utf8(utf8_empty, utf8_empty);
  CHECK(utf8_empty_res.second == nullptr);

  const char *utf8_two = "\xC2\xA9";
  auto utf8_two_res = parser::decode_utf8(utf8_two, utf8_two + 2);
  CHECK(utf8_two_res.second == utf8_two + 2);

  const char *name_bad = "!";
  CHECK(parser::parse_name(name_bad, name_bad + 1) == nullptr);

  const char *escape_short = "\\";
  auto esc_short = parser::parse_char(escape_short, escape_short + 1);
  CHECK(esc_short.second == nullptr);

  const char *char_empty = "";
  auto char_empty_res = parser::parse_char(char_empty, char_empty);
  CHECK(char_empty_res.second == nullptr);

  const char *escape_bad = "\\q";
  auto esc_bad = parser::parse_char(escape_bad, escape_bad + 2);
  CHECK(esc_bad.second == nullptr);

  const char *token_bad = "<[";
  auto token_res = parser::parse_token(token_bad, token_bad + 2);
  CHECK(token_res.second == nullptr);

  const char *token_no_open = "abc";
  auto token_no = parser::parse_token(token_no_open, token_no_open + 3);
  CHECK(token_no.second == nullptr);

  const char *token_no_bracket = "<1>";
  auto token_bracket = parser::parse_token(token_no_bracket, token_no_bracket + 3);
  CHECK(token_bracket.second == nullptr);

  const char *token_no_gt = "<[1]";
  auto token_gt = parser::parse_token(token_no_gt, token_no_gt + 4);
  CHECK(token_gt.second == nullptr);

  constexpr char token_big[] = "<[4294967296]>";
  auto token_big_res = parser::parse_token(token_big, token_big + sizeof(token_big) - 1);
  CHECK(token_big_res.second == nullptr);

  constexpr char token_bad_close[] = "<[12x]>";
  auto token_bad_res =
      parser::parse_token(token_bad_close, token_bad_close + sizeof(token_bad_close) - 1);
  CHECK(token_bad_res.second == nullptr);
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

TEST_CASE("gbnf_parser_add_rule_errors") {
  emel::gbnf::parser::action::context ctx{};
  emel::gbnf::grammar grammar{};
  emel::gbnf::parser::detail::recursive_descent_parser parser{ctx, &grammar};
  emel::gbnf::parser::detail::rule_builder rule{};
  CHECK(rule.push({emel::gbnf::element_type::end, 0}));
  CHECK(parser.add_rule(0, rule));
  CHECK_FALSE(parser.add_rule(0, rule));
  CHECK(ctx.phase_error == EMEL_ERR_PARSE_FAILED);
}

TEST_CASE("gbnf_parser_requires_grammar_out") {
  emel::gbnf::parser::action::context ctx{};
  emel::gbnf::parser::detail::recursive_descent_parser parser{ctx, nullptr};
  CHECK_FALSE(parser.parse("root ::= \"a\""));
  CHECK(ctx.phase_error == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("gbnf_parser_symbol_and_rule_error_paths") {
  emel::gbnf::parser::action::context ctx{};
  emel::gbnf::grammar grammar{};
  emel::gbnf::parser::detail::recursive_descent_parser parser{ctx, &grammar};
  emel::gbnf::parser::detail::rule_builder rule{};
  CHECK(rule.push({emel::gbnf::element_type::end, 0}));

  parser.get_symbol_id(nullptr, 0);
  CHECK(ctx.phase_error == EMEL_ERR_PARSE_FAILED);

  ctx.phase_error = EMEL_OK;
  parser.next_symbol_id = emel::gbnf::k_max_gbnf_rules;
  parser.get_symbol_id("a", 1);
  CHECK(ctx.phase_error == EMEL_ERR_PARSE_FAILED);

  ctx.phase_error = EMEL_OK;
  parser.next_symbol_id = emel::gbnf::k_max_gbnf_rules;
  parser.generate_symbol_id("x");
  CHECK(ctx.phase_error == EMEL_ERR_PARSE_FAILED);

  ctx.phase_error = EMEL_OK;
  emel::gbnf::parser::detail::recursive_descent_parser parser_null{ctx, nullptr};
  CHECK_FALSE(parser_null.add_rule(0, rule));
  CHECK(ctx.phase_error == EMEL_ERR_PARSE_FAILED);

  ctx.phase_error = EMEL_OK;
  CHECK_FALSE(parser.add_rule(emel::gbnf::k_max_gbnf_rules, rule));
  CHECK(ctx.phase_error == EMEL_ERR_PARSE_FAILED);

  ctx.phase_error = EMEL_OK;
  grammar.element_count = emel::gbnf::k_max_gbnf_elements;
  CHECK_FALSE(parser.add_rule(1, rule));
  CHECK(ctx.phase_error == EMEL_ERR_PARSE_FAILED);

  ctx.phase_error = EMEL_OK;
  parser.symbols.entries.fill({});
  for (auto &entry : parser.symbols.entries) {
    entry.name = "x";
    entry.hash = 1;
    entry.id = 7;
    entry.occupied = true;
  }
  parser.symbols.count = 0;
  parser.next_symbol_id = 0;
  parser.get_symbol_id("y", 1);
  CHECK(ctx.phase_error == EMEL_ERR_PARSE_FAILED);
}
