#include "parity_runner.hpp"

#include <cstdio>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "emel/gbnf/rule_parser/events.hpp"
#include "emel/gbnf/rule_parser/sm.hpp"

#include "llama-grammar.h"

namespace {

constexpr int32_t k_error_ok = 0;
constexpr int32_t k_error_internal = 3;

struct parser_done_capture {
  bool called = false;
  const emel::gbnf::grammar * grammar = nullptr;
};

struct parser_error_capture {
  bool called = false;
  const emel::gbnf::grammar * grammar = nullptr;
  int32_t err = 0;
};

bool on_gbnf_done(void * owner, const emel::gbnf::rule_parser::events::parsing_done & ev) {
  auto * capture = static_cast<parser_done_capture *>(owner);
  capture->called = true;
  capture->grammar = &ev.grammar;
  return true;
}

bool on_gbnf_error(void * owner, const emel::gbnf::rule_parser::events::parsing_error & ev) {
  auto * capture = static_cast<parser_error_capture *>(owner);
  capture->called = true;
  capture->grammar = &ev.grammar;
  capture->err = ev.err;
  return true;
}

bool run_emel_gbnf_parse(std::string_view grammar_text,
                         emel::gbnf::grammar & grammar_out,
                         int32_t & err_out) {
  parser_done_capture done_capture{};
  parser_error_capture error_capture{};

  emel::callback<bool(const emel::gbnf::rule_parser::events::parsing_done &)> done_cb{
      &done_capture,
      on_gbnf_done};
  emel::callback<bool(const emel::gbnf::rule_parser::events::parsing_error &)> error_cb{
      &error_capture,
      on_gbnf_error};

  emel::gbnf::rule_parser::event::parse parse_ev{
      .grammar_text = grammar_text,
      .grammar_out = &grammar_out,
      .dispatch_done = done_cb,
      .dispatch_error = error_cb,
  };

  emel::gbnf::rule_parser::sm parser{};
  const bool accepted = parser.process_event(parse_ev);
  if (accepted && done_capture.called && !error_capture.called) {
    err_out = k_error_ok;
    return true;
  }
  if (error_capture.called) {
    err_out = error_capture.err;
    return false;
  }
  err_out = k_error_internal;
  return false;
}

bool run_llama_gbnf_parse(std::string_view grammar_text, llama_grammar_rules & rules_out) {
  std::string grammar(grammar_text);
  llama_grammar_parser parser{nullptr};
  if (!parser.parse(grammar.c_str())) {
    return false;
  }
  rules_out = std::move(parser.rules);
  return true;
}

bool compare_grammars(const emel::gbnf::grammar & emel_grammar,
                      const llama_grammar_rules & llama_rules) {
  if (emel_grammar.rule_count != llama_rules.size()) {
    std::fprintf(stderr,
                 "rule count mismatch: emel=%u llama=%zu\n",
                 emel_grammar.rule_count,
                 llama_rules.size());
    return false;
  }

  for (uint32_t rule_id = 0; rule_id < emel_grammar.rule_count; ++rule_id) {
    const emel::gbnf::rule_view emel_rule = emel_grammar.rule(rule_id);
    const llama_grammar_rule & llama_rule = llama_rules[rule_id];
    const uint32_t llama_len = static_cast<uint32_t>(llama_rule.size());
    if (emel_rule.length != llama_len) {
      std::fprintf(stderr,
                   "rule length mismatch at rule %u: emel=%u llama=%u\n",
                   rule_id,
                   emel_rule.length,
                   llama_len);
      return false;
    }
    for (uint32_t i = 0; i < emel_rule.length; ++i) {
      const emel::gbnf::element & emel_elem = emel_rule.elements[i];
      const llama_grammar_element & llama_elem = llama_rule[i];
      const uint32_t emel_type = static_cast<uint32_t>(emel_elem.type);
      const uint32_t llama_type = static_cast<uint32_t>(llama_elem.type);
      if (emel_type != llama_type || emel_elem.value != llama_elem.value) {
        std::fprintf(stderr,
                     "element mismatch at rule %u index %u: "
                     "emel(type=%u,value=%u) llama(type=%u,value=%u)\n",
                     rule_id,
                     i,
                     emel_type,
                     emel_elem.value,
                     llama_type,
                     llama_elem.value);
        return false;
      }
    }
  }
  return true;
}

void dump_emel_grammar(const emel::gbnf::grammar & grammar) {
  std::fprintf(stdout,
               "emel grammar: rules=%u elements=%u\n",
               grammar.rule_count,
               grammar.element_count);
  for (uint32_t rule_id = 0; rule_id < grammar.rule_count; ++rule_id) {
    const emel::gbnf::rule_view rule = grammar.rule(rule_id);
    std::fprintf(stdout, "  rule[%u] len=%u:", rule_id, rule.length);
    for (uint32_t i = 0; i < rule.length; ++i) {
      const emel::gbnf::element & elem = rule.elements[i];
      std::fprintf(stdout,
                   " (%u,%u)",
                   static_cast<uint32_t>(elem.type),
                   elem.value);
    }
    std::fprintf(stdout, "\n");
  }
}

void dump_llama_grammar(const llama_grammar_rules & rules) {
  std::fprintf(stdout, "llama grammar: rules=%zu\n", rules.size());
  for (size_t rule_id = 0; rule_id < rules.size(); ++rule_id) {
    const llama_grammar_rule & rule = rules[rule_id];
    std::fprintf(stdout, "  rule[%zu] len=%zu:", rule_id, rule.size());
    for (const auto & elem : rule) {
      std::fprintf(stdout,
                   " (%u,%u)",
                   static_cast<unsigned int>(elem.type),
                   static_cast<unsigned int>(elem.value));
    }
    std::fprintf(stdout, "\n");
  }
}

int run_tokenizer_parity(const emel::paritychecker::parity_options &) {
  std::fprintf(stderr, "tokenizer parity is scaffolded\n");
  return 1;
}

int run_gbnf_parser_parity(const emel::paritychecker::parity_options & opts) {
  emel::gbnf::grammar emel_grammar{};
  int32_t emel_err = k_error_ok;
  const bool emel_ok = run_emel_gbnf_parse(opts.text, emel_grammar, emel_err);

  llama_grammar_rules llama_rules;
  const bool llama_ok = run_llama_gbnf_parse(opts.text, llama_rules);

  if (emel_ok != llama_ok) {
    std::fprintf(stderr,
                 "parse outcome mismatch: emel=%s llama=%s (emel_err=%d)\n",
                 emel_ok ? "ok" : "error",
                 llama_ok ? "ok" : "error",
                 emel_err);
    if (opts.dump) {
      if (emel_ok) {
        dump_emel_grammar(emel_grammar);
      }
      if (llama_ok) {
        dump_llama_grammar(llama_rules);
      }
    }
    return 1;
  }

  if (!emel_ok) {
    std::fprintf(stdout, "parity ok (both parsers rejected grammar)\n");
    return 0;
  }

  const bool matched = compare_grammars(emel_grammar, llama_rules);
  if (!matched) {
    if (opts.dump) {
      dump_emel_grammar(emel_grammar);
      dump_llama_grammar(llama_rules);
    }
    return 1;
  }

  if (opts.dump) {
    dump_emel_grammar(emel_grammar);
  }
  std::fprintf(stdout,
               "parity ok (%u rules, %u elements)\n",
               emel_grammar.rule_count,
               emel_grammar.element_count);
  return 0;
}

}  // namespace

namespace emel::paritychecker {

int run_parity(const parity_options & opts) {
  switch (opts.mode) {
    case parity_mode::gbnf_parser:
      return run_gbnf_parser_parity(opts);
    case parity_mode::tokenizer:
    default:
      return run_tokenizer_parity(opts);
  }
}

}  // namespace emel::paritychecker
