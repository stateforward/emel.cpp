#include <cstdio>

#include "emel/model/loader/sm.hpp"
#include "emel/model/parser/events.hpp"
#include "emel/model/weight_loader/events.hpp"

namespace {

void print_step(const char * step, const bool accepted) {
  std::printf("[%s] accepted=%s\n", step, accepted ? "true" : "false");
}

}  // namespace

int main() {
  {
    std::printf("=== model_load happy path ===\n");
    emel::model::loader::sm loader_sm;
    emel::model::data model_data = {};

    print_step(
      "load",
      emel::model::loader::load(
        loader_sm,
        emel::model::loader::event::load{
          .model_data = model_data,
          .model_path = "mock.gguf",
          .request_mmap = true,
          .request_direct_io = false,
        }
      )
    );
    print_step(
      "mapping_parser_done",
      loader_sm.process_event(emel::model::loader::event::mapping_parser_done{})
    );
    print_step(
      "parsing_done",
      loader_sm.process_event(emel::model::parser::events::parsing_done{})
    );
    print_step(
      "loading_done",
      loader_sm.process_event(emel::model::weight_loader::events::loading_done{})
    );
    print_step(
      "layers_mapped",
      loader_sm.process_event(emel::model::loader::event::layers_mapped{})
    );
    print_step(
      "structure_validated",
      loader_sm.process_event(emel::model::loader::event::structure_validated{})
    );

    // Depending on guard::has_arch_validate, this may be accepted (needs arch validation)
    // or rejected (already terminal done).
    print_step(
      "architecture_validated",
      loader_sm.process_event(emel::model::loader::event::architecture_validated{})
    );
  }

  {
    std::printf("=== model_load unsupported format ===\n");
    emel::model::loader::sm loader_sm;
    emel::model::data model_data = {};

    print_step(
      "load",
      emel::model::loader::load(
        loader_sm,
        emel::model::loader::event::load{
          .model_data = model_data,
          .model_path = "unsupported.bin",
          .request_mmap = false,
          .request_direct_io = true,
        }
      )
    );
    print_step(
      "unsupported_format_error",
      loader_sm.process_event(emel::model::loader::event::unsupported_format_error{})
    );
    print_step(
      "parsing_done_after_error",
      loader_sm.process_event(emel::model::parser::events::parsing_done{})
    );
  }

  return 0;
}
