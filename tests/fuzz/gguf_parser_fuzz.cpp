#include <cstddef>
#include <cstdint>

#include "emel/model/data.hpp"
#include "emel/parser/gguf/sm.hpp"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
  if (data == nullptr || size == 0) {
    return 0;
  }

  emel::parser::gguf::sm machine{};
  emel::parser::gguf::requirements req{};
  emel::parser::gguf::event::probe probe{
    .file_image = data,
    .size = size,
    .requirements_out = &req,
  };
  (void)machine.process_event(probe);

  uint8_t kv_arena[8] = {};
  emel::model::data::tensor_record tensors[1] = {};
  emel::parser::gguf::event::bind_storage bind{
    .kv_arena = kv_arena,
    .kv_arena_size = sizeof(kv_arena),
    .tensors = tensors,
    .tensor_capacity = 1,
  };
  (void)machine.process_event(bind);

  emel::parser::gguf::event::parse parse{
    .file_image = data,
    .size = size,
  };
  (void)machine.process_event(parse);

  return 0;
}
