#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string_view>

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/parser/events.hpp"
#include "emel/parser/gguf/actions.hpp"
#include "emel/parser/gguf/context.hpp"
#include "emel/parser/gguf/sm.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
#if defined(_WIN32)
  (void)data;
  (void)size;
  return 0;
#else
  std::FILE *file = std::tmpfile();
  if (file == nullptr) {
    return 0;
  }
  if (size > 0) {
    (void)std::fwrite(data, 1, size, file);
  }
  std::rewind(file);

  auto gguf_ctx = std::make_unique<emel::parser::gguf::context>();
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::event::load load_ev{*model};
  load_ev.file_handle = file;
  load_ev.format_ctx = gguf_ctx.get();

  int32_t map_err = EMEL_OK;
  (void)emel::parser::gguf::map_parser(load_ev, &map_err);

  emel::parser::event::parse_model parse_ev{};
  parse_ev.model = model.get();
  parse_ev.file_handle = file;
  parse_ev.format_ctx = gguf_ctx.get();
  parse_ev.map_tensors = false;
  parse_ev.architectures = nullptr;
  parse_ev.n_architectures = 0;
  parse_ev.loader_request = &load_ev;

  emel::parser::gguf::sm machine{};
  (void)machine.process_event(parse_ev);

  std::fclose(file);
  return 0;
#endif
}
