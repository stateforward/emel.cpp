#include "bench_cases.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

#include "emel/gbnf/rule_parser/events.hpp"
#include "emel/gbnf/rule_parser/sm.hpp"

#include "llama-grammar.h"

namespace {

constexpr char k_basic_grammar[] = "root ::= [a-z]+";
constexpr std::string_view k_basic_grammar_view{k_basic_grammar, sizeof(k_basic_grammar) - 1};

constexpr char k_complex_grammar[] =
    "# comment\n"
    "root ::= (\"ab\" | [a-z] | [^0-9] | <[3]> | !<[4]> | . | name-ref)+\n"
    "name-ref ::= \"\\x41\\u0042\\u00000043\" {2,3} \"t\\n\\r\\t\\\\\"?\n"
    "range-rule ::= [a-zA-z0-9]*";
constexpr std::string_view k_complex_grammar_view{
    k_complex_grammar,
    sizeof(k_complex_grammar) - 1};

struct parser_done_capture {
  bool called = false;
  const emel::gbnf::grammar * grammar = nullptr;
};

struct parser_error_capture {
  bool called = false;
  const emel::gbnf::grammar * grammar = nullptr;
  int32_t err = 0;
};

bool on_gbnf_done(void * owner, const emel::gbnf::rule_parser::events::parsing_done & ev) noexcept {
  auto * capture = static_cast<parser_done_capture *>(owner);
  capture->called = true;
  capture->grammar = &ev.grammar;
  return true;
}

bool on_gbnf_error(void * owner, const emel::gbnf::rule_parser::events::parsing_error & ev) noexcept {
  auto * capture = static_cast<parser_error_capture *>(owner);
  capture->called = true;
  capture->grammar = &ev.grammar;
  capture->err = ev.err;
  return true;
}

struct emel_parse_state {
  parser_done_capture done_capture = {};
  parser_error_capture error_capture = {};
  emel::callback<bool(const emel::gbnf::rule_parser::events::parsing_done &)> done_cb;
  emel::callback<bool(const emel::gbnf::rule_parser::events::parsing_error &)> error_cb;
  emel::gbnf::grammar grammar = {};
  emel::gbnf::rule_parser::event::parse parse_ev = {};
  emel::gbnf::rule_parser::sm parser = {};

  explicit emel_parse_state(const std::string_view grammar_text) noexcept
      : done_cb(&done_capture, on_gbnf_done),
        error_cb(&error_capture, on_gbnf_error),
        parse_ev{
            .grammar_text = grammar_text,
            .grammar_out = &grammar,
            .dispatch_done = done_cb,
            .dispatch_error = error_cb,
        } {}
};

bool parse_emel_gbnf(emel_parse_state & state) noexcept {
  state.done_capture.called = false;
  state.done_capture.grammar = nullptr;
  state.error_capture.called = false;
  state.error_capture.grammar = nullptr;
  state.error_capture.err = 0;

  const bool accepted = state.parser.process_event(state.parse_ev);
  if (!accepted) {
    return false;
  }
  if (!state.done_capture.called || state.error_capture.called) {
    return false;
  }
  return state.done_capture.grammar == state.parse_ev.grammar_out;
}

struct reference_parse_state {
  llama_grammar_parser parser{nullptr};
  const char * grammar_text = nullptr;

  explicit reference_parse_state(const char * text) noexcept : parser{nullptr}, grammar_text(text) {
  }
};

bool parse_reference_gbnf(reference_parse_state & state) {
  state.parser.symbol_ids.clear();
  state.parser.rules.clear();
  if (!state.parser.parse(state.grammar_text)) {
    return false;
  }
  return !state.parser.rules.empty();
}

bool compare_grammars(const emel::gbnf::grammar & emel_grammar,
                      const llama_grammar_rules & llama_rules) {
  if (emel_grammar.rule_count != llama_rules.size()) {
    return false;
  }

  for (uint32_t rule_id = 0; rule_id < emel_grammar.rule_count; ++rule_id) {
    const emel::gbnf::rule_view emel_rule = emel_grammar.rule(rule_id);
    const llama_grammar_rule & llama_rule = llama_rules[rule_id];
    const uint32_t llama_len = static_cast<uint32_t>(llama_rule.size());
    if (emel_rule.length != llama_len) {
      return false;
    }

    for (uint32_t i = 0; i < emel_rule.length; ++i) {
      const emel::gbnf::element & emel_elem = emel_rule.elements[i];
      const llama_grammar_element & llama_elem = llama_rule[i];
      const uint32_t emel_type = static_cast<uint32_t>(emel_elem.type);
      const uint32_t llama_type = static_cast<uint32_t>(llama_elem.type);
      if (emel_type != llama_type || emel_elem.value != llama_elem.value) {
        return false;
      }
    }
  }
  return true;
}

void ensure_case_parity(const char * case_name,
                        const char * grammar_text,
                        std::string_view grammar_view) {
  emel_parse_state emel_state{grammar_view};
  reference_parse_state reference_state{grammar_text};

  if (!parse_emel_gbnf(emel_state)) {
    std::fprintf(stderr, "error: emel parse failed for %s\n", case_name);
    std::abort();
  }
  if (!parse_reference_gbnf(reference_state)) {
    std::fprintf(stderr, "error: llama parse failed for %s\n", case_name);
    std::abort();
  }
  if (!compare_grammars(emel_state.grammar, reference_state.parser.rules)) {
    std::fprintf(stderr, "error: grammar parity mismatch for %s\n", case_name);
    std::abort();
  }
}

void ensure_gbnf_rule_parser_parity() {
  static bool checked = false;
  if (checked) {
    return;
  }
  checked = true;

  ensure_case_parity("gbnf/rule_parser_basic", k_basic_grammar, k_basic_grammar_view);
  ensure_case_parity("gbnf/rule_parser_complex", k_complex_grammar, k_complex_grammar_view);
}

}  // namespace

namespace emel::bench {

void append_emel_gbnf_rule_parser_cases(std::vector<result> & results, const config & cfg) {
  ensure_gbnf_rule_parser_parity();

  {
    emel_parse_state state{k_basic_grammar_view};
    auto basic_fn = [&]() {
      if (!parse_emel_gbnf(state)) {
        std::abort();
      }
    };
    results.push_back(measure_case("gbnf/rule_parser_basic", cfg, basic_fn));
  }

  {
    emel_parse_state state{k_complex_grammar_view};
    auto complex_fn = [&]() {
      if (!parse_emel_gbnf(state)) {
        std::abort();
      }
    };
    results.push_back(measure_case("gbnf/rule_parser_complex", cfg, complex_fn));
  }
}

void append_reference_gbnf_rule_parser_cases(std::vector<result> & results, const config & cfg) {
  ensure_gbnf_rule_parser_parity();

  {
    reference_parse_state state{k_basic_grammar};
    auto basic_fn = [&]() {
      if (!parse_reference_gbnf(state)) {
        std::abort();
      }
    };
    results.push_back(measure_case("gbnf/rule_parser_basic", cfg, basic_fn));
  }

  {
    reference_parse_state state{k_complex_grammar};
    auto complex_fn = [&]() {
      if (!parse_reference_gbnf(state)) {
        std::abort();
      }
    };
    results.push_back(measure_case("gbnf/rule_parser_complex", cfg, complex_fn));
  }
}

}  // namespace emel::bench
