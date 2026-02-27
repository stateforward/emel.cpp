#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "emel/gguf/loader/sm.hpp"
#include "emel/model/data.hpp"

namespace {

void on_probe_done(const emel::gguf::loader::events::probe_done &) {}
void on_probe_error(const emel::gguf::loader::events::probe_error &) {}
void on_bind_done(const emel::gguf::loader::events::bind_done &) {}
void on_bind_error(const emel::gguf::loader::events::bind_error &) {}
void on_parse_done(const emel::gguf::loader::events::parse_done &) {}
void on_parse_error(const emel::gguf::loader::events::parse_error &) {}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
  if (data == nullptr || size == 0) {
    return 0;
  }

  emel::gguf::loader::sm machine{};
  emel::gguf::loader::requirements req = {};
  const emel::gguf::loader::event::probe_done_fn probe_done_cb =
      emel::gguf::loader::event::probe_done_fn::from<&on_probe_done>();
  const emel::gguf::loader::event::probe_error_fn probe_error_cb =
      emel::gguf::loader::event::probe_error_fn::from<&on_probe_error>();
  const emel::gguf::loader::event::bind_done_fn bind_done_cb =
      emel::gguf::loader::event::bind_done_fn::from<&on_bind_done>();
  const emel::gguf::loader::event::bind_error_fn bind_error_cb =
      emel::gguf::loader::event::bind_error_fn::from<&on_bind_error>();
  const emel::gguf::loader::event::parse_done_fn parse_done_cb =
      emel::gguf::loader::event::parse_done_fn::from<&on_parse_done>();
  const emel::gguf::loader::event::parse_error_fn parse_error_cb =
      emel::gguf::loader::event::parse_error_fn::from<&on_parse_error>();

  const emel::gguf::loader::event::probe probe{
    std::span<const uint8_t>{data, size},
    req,
    probe_done_cb,
    probe_error_cb,
  };
  (void)machine.process_event(probe);

  std::array<uint8_t, 8> kv_arena = {};
  std::array<emel::gguf::loader::kv_entry, 1> kv_entries = {};
  std::array<emel::model::data::tensor_record, 1> tensors = {};
  const emel::gguf::loader::event::bind_storage bind{
    std::span<uint8_t>{kv_arena},
    std::span<emel::gguf::loader::kv_entry>{kv_entries},
    std::span<emel::model::data::tensor_record>{tensors},
    bind_done_cb,
    bind_error_cb,
  };
  (void)machine.process_event(bind);

  const emel::gguf::loader::event::parse parse{
    std::span<const uint8_t>{data, size},
    parse_done_cb,
    parse_error_cb,
  };
  (void)machine.process_event(parse);

  return 0;
}
