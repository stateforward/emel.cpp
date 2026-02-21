#pragma once

#include <cstddef>

#include "emel/parser/gguf/sm.hpp"
#include "emel/parser/map.hpp"

namespace emel::parser {

template <class sm>
inline bool dispatch_parse_sm(void * parser_sm,
                              const emel::parser::event::parse_model & ev) {
  if (parser_sm == nullptr) {
    return false;
  }
  return static_cast<sm *>(parser_sm)->process_event(ev);
}

inline dispatch_parse_fn dispatch_for_kind(const kind kind_id) {
  static constexpr dispatch_parse_fn k_table[] = {
    dispatch_parse_sm<emel::parser::gguf::sm>
  };
  static_assert(sizeof(k_table) / sizeof(k_table[0]) ==
                  static_cast<size_t>(kind::count),
                "parser dispatch table must cover all kinds");
  const size_t index = static_cast<size_t>(kind_id);
  if (index >= static_cast<size_t>(kind::count)) {
    return nullptr;
  }
  return k_table[index];
}

}  // namespace emel::parser
