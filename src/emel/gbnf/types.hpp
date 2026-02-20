#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace emel::gbnf {

/**
 * GBNF element type mappings from llama.cpp `llama_gretype`
 */
enum class element_type : uint32_t {
  end = 0,
  alt = 1,
  rule_ref = 2,
  character = 3,
  char_not = 4,
  char_rng_upper = 5,
  char_alt = 6,
  char_any = 7,
  token = 8,
  token_not = 9
};

/**
 * GBNF AST element equivalent to `llama_grammar_element`
 */
struct element {
  element_type type;
  uint32_t value; // Unicode code point, rule ID, or token ID
};

constexpr size_t k_max_gbnf_rules = 2048;
constexpr size_t k_max_gbnf_elements = 65536;
constexpr size_t k_max_gbnf_rule_elements = 4096;
constexpr size_t k_max_gbnf_symbols = 2048;
constexpr size_t k_gbnf_symbol_table_slots = 4096;

static_assert((k_gbnf_symbol_table_slots &
               (k_gbnf_symbol_table_slots - 1)) == 0,
              "gbnf symbol table slots must be power-of-two");
static_assert(k_max_gbnf_rule_elements <= k_max_gbnf_elements,
              "gbnf rule elements must fit within total elements");

struct rule_view {
  const element *elements = nullptr;
  uint32_t length = 0;
};

struct grammar {
  std::array<element, k_max_gbnf_elements> elements = {};
  std::array<uint32_t, k_max_gbnf_rules> rule_offsets = {};
  std::array<uint32_t, k_max_gbnf_rules> rule_lengths = {};
  uint32_t rule_count = 0;
  uint32_t element_count = 0;

  void reset() noexcept {
    for (size_t i = 0; i < rule_count; ++i) {
      rule_offsets[i] = 0;
      rule_lengths[i] = 0;
    }
    rule_count = 0;
    element_count = 0;
  }

  rule_view rule(uint32_t rule_id) const noexcept {
    if (rule_id >= rule_count) {
      return {};
    }
    const uint32_t length = rule_lengths[rule_id];
    if (length == 0) {
      return {};
    }
    const uint32_t offset = rule_offsets[rule_id];
    if (offset + length > element_count) {
      return {};
    }
    return {elements.data() + offset, length};
  }
};

} // namespace emel::gbnf
