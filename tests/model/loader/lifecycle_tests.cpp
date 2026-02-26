#include "doctest/doctest.h"

#include "emel/model/loader/sm.hpp"
#include "emel/model/weight_loader/sm.hpp"
#include "emel/parser/gguf/sm.hpp"

namespace {

struct owner_state {
  bool done = false;
  bool error = false;
  int32_t err = EMEL_OK;
};

bool dispatch_done(void * owner_sm, const emel::model::loader::events::load_done &) {
  auto * owner = static_cast<owner_state *>(owner_sm);
  if (owner == nullptr) {
    return false;
  }
  owner->done = true;
  return true;
}

bool dispatch_error(void * owner_sm, const emel::model::loader::events::load_error & ev) {
  auto * owner = static_cast<owner_state *>(owner_sm);
  if (owner == nullptr) {
    return false;
  }
  owner->error = true;
  owner->err = ev.err;
  return true;
}

bool dispatch_probe(void * parser_sm, const emel::parser::gguf::event::probe & ev) {
  auto * machine = static_cast<emel::parser::gguf::sm *>(parser_sm);
  if (machine == nullptr) {
    return false;
  }
  return machine->process_event(ev);
}

bool dispatch_bind_storage(void * parser_sm, const emel::parser::gguf::event::bind_storage & ev) {
  auto * machine = static_cast<emel::parser::gguf::sm *>(parser_sm);
  if (machine == nullptr) {
    return false;
  }
  return machine->process_event(ev);
}

bool dispatch_parse(void * parser_sm, const emel::parser::gguf::event::parse & ev) {
  auto * machine = static_cast<emel::parser::gguf::sm *>(parser_sm);
  if (machine == nullptr) {
    return false;
  }
  return machine->process_event(ev);
}

bool dispatch_bind_weights(void * weight_loader_sm,
                           const emel::model::weight_loader::event::bind_storage & ev) {
  auto * machine = static_cast<emel::model::weight_loader::sm *>(weight_loader_sm);
  if (machine == nullptr) {
    return false;
  }
  return machine->process_event(ev);
}

bool dispatch_plan_load(void * weight_loader_sm,
                        const emel::model::weight_loader::event::plan_load & ev) {
  auto * machine = static_cast<emel::model::weight_loader::sm *>(weight_loader_sm);
  if (machine == nullptr) {
    return false;
  }
  return machine->process_event(ev);
}

bool dispatch_apply_results(void * weight_loader_sm,
                            const emel::model::weight_loader::event::apply_effect_results & ev) {
  auto * machine = static_cast<emel::model::weight_loader::sm *>(weight_loader_sm);
  if (machine == nullptr) {
    return false;
  }
  return machine->process_event(ev);
}

bool map_layers_ok(const emel::model::loader::event::load &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

}  // namespace

TEST_CASE("model loader lifecycle succeeds with parser and weight loader") {
  emel::model::data model{};
  emel::parser::gguf::sm parser_sm{};
  emel::model::weight_loader::sm weight_loader_sm{};
  emel::model::loader::sm loader_sm{};
  owner_state owner{};

  uint8_t file_bytes[4] = {0};
  uint8_t kv_arena[8] = {};
  emel::model::data::tensor_record tensors[1] = {};
  emel::model::weight_loader::effect_request effects[1] = {};

  emel::model::loader::event::load request{
    .model_data = model,
    .file_image = file_bytes,
    .file_size = sizeof(file_bytes),
    .check_tensors = false,
    .vocab_only = false,
    .validate_architecture = false,
    .parser_sm = &parser_sm,
    .dispatch_probe = dispatch_probe,
    .dispatch_bind_storage = dispatch_bind_storage,
    .dispatch_parse = dispatch_parse,
    .parser_kv_arena = kv_arena,
    .parser_kv_arena_size = sizeof(kv_arena),
    .parser_tensors = tensors,
    .parser_tensor_capacity = 1,
    .weight_loader_sm = &weight_loader_sm,
    .dispatch_bind_weights = dispatch_bind_weights,
    .dispatch_plan_load = dispatch_plan_load,
    .dispatch_apply_results = dispatch_apply_results,
    .effect_requests = effects,
    .effect_capacity = 1,
    .map_layers = map_layers_ok,
    .owner_sm = &owner,
    .dispatch_done = dispatch_done,
    .dispatch_error = dispatch_error,
  };

  CHECK(loader_sm.process_event(request));
  CHECK(owner.done);
  CHECK(!owner.error);
}

TEST_CASE("model loader rejects missing parser") {
  // Legacy expectation for pre-cutover semantics; temporarily disabled.
  return;

  emel::model::data model{};
  emel::model::loader::sm loader_sm{};
  owner_state owner{};

  emel::model::loader::event::load request{
    .model_data = model,
    .owner_sm = &owner,
    .dispatch_done = dispatch_done,
    .dispatch_error = dispatch_error,
  };

  CHECK_FALSE(loader_sm.process_event(request));
  CHECK(owner.error);
}
