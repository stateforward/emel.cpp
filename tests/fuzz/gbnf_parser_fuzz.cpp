#include <cstddef>
#include <cstdint>
#include <string_view>

#include "emel/gbnf/parser/context.hpp"
#include "emel/gbnf/parser/events.hpp"
#include "emel/gbnf/parser/sm.hpp"
#include "emel/gbnf/types.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  emel::gbnf::grammar grammar{};
  emel::gbnf::parser::action::context ctx{};
  emel::gbnf::parser::sm machine{ctx};

  std::string_view input(reinterpret_cast<const char *>(data), size);
  int32_t err = 0;
  emel::gbnf::event::parse ev{};
  ev.grammar_text = input;
  ev.grammar_out = &grammar;
  ev.error_out = &err;

  (void)machine.process_event(ev);
  return 0;
}
