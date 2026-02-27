#include <cstdio>

#include "emel/model/loader/sm.hpp"
#include "emel/model/weight_loader/sm.hpp"
#include "emel/gguf/loader/sm.hpp"

namespace {

void print_step(const char * step, const bool accepted) {
  std::printf("[%s] accepted=%s\n", step, accepted ? "true" : "false");
}

bool dispatch_probe(void * parser_sm, const emel::gguf::loader::event::probe & ev) {
  auto * machine = static_cast<emel::gguf::loader::sm *>(parser_sm);
  return machine != nullptr && machine->process_event(ev);
}

bool dispatch_bind_storage(void * parser_sm, const emel::gguf::loader::event::bind_storage & ev) {
  auto * machine = static_cast<emel::gguf::loader::sm *>(parser_sm);
  return machine != nullptr && machine->process_event(ev);
}

bool dispatch_parse(void * parser_sm, const emel::gguf::loader::event::parse & ev) {
  auto * machine = static_cast<emel::gguf::loader::sm *>(parser_sm);
  return machine != nullptr && machine->process_event(ev);
}

bool dispatch_bind_weights(void * weight_loader_sm,
                           const emel::model::weight_loader::event::bind_storage & ev) {
  auto * machine = static_cast<emel::model::weight_loader::sm *>(weight_loader_sm);
  return machine != nullptr && machine->process_event(ev);
}

bool dispatch_plan_load(void * weight_loader_sm,
                        const emel::model::weight_loader::event::plan_load & ev) {
  auto * machine = static_cast<emel::model::weight_loader::sm *>(weight_loader_sm);
  return machine != nullptr && machine->process_event(ev);
}

bool dispatch_apply_results(void * weight_loader_sm,
                            const emel::model::weight_loader::event::apply_effect_results & ev) {
  auto * machine = static_cast<emel::model::weight_loader::sm *>(weight_loader_sm);
  return machine != nullptr && machine->process_event(ev);
}

bool map_layers_ok(const emel::model::loader::event::load &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

}  // namespace

int main() {
  {
    std::printf("=== model_load happy path ===\n");
    emel::model::weight_loader::sm weight_sm;
    emel::gguf::loader::sm parser_sm;
    emel::model::loader::sm loader_sm;
    emel::model::data model_data = {};
    uint8_t file_bytes[4] = {0};
    uint8_t kv_arena[8] = {};
    emel::model::data::tensor_record tensors[1] = {};
    emel::model::weight_loader::effect_request effects[1] = {};

    int32_t err = EMEL_OK;
    print_step(
      "load",
      loader_sm.process_event(emel::model::loader::event::load{
        .model_data = model_data,
        .file_image = file_bytes,
        .file_size = sizeof(file_bytes),
        .check_tensors = false,
        .validate_architecture = false,
        .parser_sm = &parser_sm,
        .dispatch_probe = dispatch_probe,
        .dispatch_bind_storage = dispatch_bind_storage,
        .dispatch_parse = dispatch_parse,
        .parser_kv_arena = kv_arena,
        .parser_kv_arena_size = sizeof(kv_arena),
        .parser_tensors = tensors,
        .parser_tensor_capacity = 1,
        .weight_loader_sm = &weight_sm,
        .dispatch_bind_weights = dispatch_bind_weights,
        .dispatch_plan_load = dispatch_plan_load,
        .dispatch_apply_results = dispatch_apply_results,
        .effect_requests = effects,
        .effect_capacity = 1,
        .map_layers = map_layers_ok,
        .error_out = &err,
      })
    );
    std::printf("load error_out=%d\n", err);
  }

  {
    std::printf("=== model_load missing parser ===\n");
    emel::model::weight_loader::sm weight_sm;
    emel::model::loader::sm loader_sm;
    emel::model::data model_data = {};
    uint8_t file_bytes[4] = {0};

    int32_t err = EMEL_OK;
    print_step(
      "load",
      loader_sm.process_event(emel::model::loader::event::load{
        .model_data = model_data,
        .file_image = file_bytes,
        .file_size = sizeof(file_bytes),
        .weight_loader_sm = &weight_sm,
        .error_out = &err,
      })
    );
    std::printf("load error_out=%d\n", err);
  }

  return 0;
}
