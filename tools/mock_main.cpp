#include <cstdio>

#include "emel/model/loader/sm.hpp"
#include "emel/parser/map.hpp"
#include "emel/parser/sm.hpp"
#include "emel/model/weight_loader/sm.hpp"

namespace {

void print_step(const char * step, const bool accepted) {
  std::printf("[%s] accepted=%s\n", step, accepted ? "true" : "false");
}

bool map_parser_ok(const emel::model::loader::event::load &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool can_handle_any(const emel::model::loader::event::load &) {
  return true;
}

bool can_handle_none(const emel::model::loader::event::load &) {
  return false;
}

bool dispatch_parse_ok(void *, const emel::parser::event::parse_model & ev) {
  if (ev.dispatch_done == nullptr) {
    return false;
  }
  return ev.dispatch_done(ev.owner_sm,
                          emel::model::loader::events::parsing_done{ev.loader_request});
}

bool map_mmap_ok(const emel::model::weight_loader::event::load_weights &,
                 uint64_t * bytes_done,
                 uint64_t * bytes_total,
                 int32_t * err_out) {
  if (bytes_done != nullptr) {
    *bytes_done = 256;
  }
  if (bytes_total != nullptr) {
    *bytes_total = 512;
  }
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool load_streamed_ok(const emel::model::weight_loader::event::load_weights &,
                      uint64_t * bytes_done,
                      uint64_t * bytes_total,
                      int32_t * err_out) {
  if (bytes_done != nullptr) {
    *bytes_done = 128;
  }
  if (bytes_total != nullptr) {
    *bytes_total = 256;
  }
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool map_layers_ok(const emel::model::loader::event::load &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool validate_structure_ok(const emel::model::loader::event::load &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool validate_architecture_ok(const emel::model::loader::event::load &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

bool dispatch_parsing_done(void * loader_sm, const emel::model::loader::events::parsing_done & ev) {
  auto * machine = static_cast<emel::model::loader::sm *>(loader_sm);
  return machine != nullptr && machine->process_event(ev);
}

bool dispatch_parsing_error(void * loader_sm, const emel::model::loader::events::parsing_error & ev) {
  auto * machine = static_cast<emel::model::loader::sm *>(loader_sm);
  return machine != nullptr && machine->process_event(ev);
}

bool dispatch_loading_done(void * loader_sm, const emel::model::loader::events::loading_done & ev) {
  auto * machine = static_cast<emel::model::loader::sm *>(loader_sm);
  return machine != nullptr && machine->process_event(ev);
}

bool dispatch_loading_error(void * loader_sm, const emel::model::loader::events::loading_error & ev) {
  auto * machine = static_cast<emel::model::loader::sm *>(loader_sm);
  return machine != nullptr && machine->process_event(ev);
}

}  // namespace

int main() {
  {
    std::printf("=== model_load happy path ===\n");
    emel::model::weight_loader::sm weight_sm;
    emel::model::loader::sm loader_sm;
    emel::model::data model_data = {};
    emel::parser::entry parser_entries[] = {
      emel::parser::entry{
        emel::parser::kind::gguf,
        &loader_sm,
        map_parser_ok,
        can_handle_any,
        dispatch_parse_ok
      }
    };
    emel::parser::map parser_map{parser_entries, 1};

    int32_t err = EMEL_OK;
    print_step(
      "load",
      loader_sm.process_event(emel::model::loader::event::load{
        .model_data = model_data,
        .model_path = "mock.gguf",
        .request_mmap = true,
        .request_direct_io = false,
        .parser_map = &parser_map,
        .map_mmap = map_mmap_ok,
        .load_streamed = load_streamed_ok,
        .map_layers = map_layers_ok,
        .validate_structure = validate_structure_ok,
        .validate_architecture_impl = validate_architecture_ok,
        .weight_loader_sm = &weight_sm,
        .dispatch_load_weights = emel::model::weight_loader::dispatch_load_weights,
        .loader_sm = &loader_sm,
        .dispatch_parsing_done = dispatch_parsing_done,
        .dispatch_parsing_error = dispatch_parsing_error,
        .dispatch_loading_done = dispatch_loading_done,
        .dispatch_loading_error = dispatch_loading_error,
        .error_out = &err
      })
    );
    std::printf("load error_out=%d\n", err);
  }

  {
    std::printf("=== model_load unsupported format ===\n");
    emel::model::weight_loader::sm weight_sm;
    emel::model::loader::sm loader_sm;
    emel::model::data model_data = {};
    emel::parser::entry parser_entries[] = {
      emel::parser::entry{
        emel::parser::kind::gguf,
        &loader_sm,
        map_parser_ok,
        can_handle_none,
        dispatch_parse_ok
      }
    };
    emel::parser::map parser_map{parser_entries, 1};

    int32_t err = EMEL_OK;
    print_step(
      "load",
      loader_sm.process_event(emel::model::loader::event::load{
        .model_data = model_data,
        .model_path = "unsupported.bin",
        .request_mmap = false,
        .request_direct_io = true,
        .parser_map = &parser_map,
        .map_mmap = map_mmap_ok,
        .load_streamed = load_streamed_ok,
        .map_layers = map_layers_ok,
        .validate_structure = validate_structure_ok,
        .validate_architecture_impl = validate_architecture_ok,
        .weight_loader_sm = &weight_sm,
        .dispatch_load_weights = emel::model::weight_loader::dispatch_load_weights,
        .loader_sm = &loader_sm,
        .dispatch_parsing_done = dispatch_parsing_done,
        .dispatch_parsing_error = dispatch_parsing_error,
        .dispatch_loading_done = dispatch_loading_done,
        .dispatch_loading_error = dispatch_loading_error,
        .error_out = &err
      })
    );
    std::printf("load error_out=%d\n", err);
  }

  return 0;
}
