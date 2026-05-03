#include <array>
#include <cstdio>
#include <span>

#include "emel/error/error.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/model/tensor/sm.hpp"

namespace {

void print_step(const char *step, const bool accepted) {
  std::printf("[%s] accepted=%s\n", step, accepted ? "true" : "false");
}

struct load_probe_state {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::model::loader::error::none);
};

emel::error::type
parse_model_ok(void *, const emel::model::loader::event::load &req) noexcept {
  req.model_data.n_tensors = 1u;
  req.model_data.n_layers = 1;
  auto &tensor = req.model_data.tensors[0];
  tensor.file_offset = 0u;
  tensor.data_size = req.file_size;
  tensor.data = const_cast<void *>(req.file_image);
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type
parse_model_fail(void *, const emel::model::loader::event::load &) noexcept {
  return emel::error::cast(emel::model::loader::error::parse_failed);
}

emel::error::type
map_layers_ok(void *, const emel::model::loader::event::load &) noexcept {
  return emel::error::cast(emel::model::loader::error::none);
}

void on_load_done(void *object,
                  const emel::model::loader::events::load_done &) noexcept {
  auto *state = static_cast<load_probe_state *>(object);
  state->done = true;
  state->error = false;
}

void on_load_error(void *object,
                   const emel::model::loader::events::load_error &ev) noexcept {
  auto *state = static_cast<load_probe_state *>(object);
  state->done = false;
  state->error = true;
  state->err = ev.err;
}

} // namespace

int main() {
  {
    std::printf("=== model_load happy path ===\n");
    emel::model::tensor::sm tensor_sm;
    emel::model::loader::sm loader_sm;
    emel::model::data model_data = {};
    std::array<uint8_t, 4> file_bytes = {};
    std::array<emel::model::tensor::effect_request, 1> effect_requests = {};
    std::array<emel::model::tensor::effect_result, 1> effect_results = {};
    load_probe_state state{};

    emel::model::loader::event::parse_model_fn parse_model{nullptr,
                                                           parse_model_ok};
    emel::model::loader::event::load request{model_data, parse_model};
    request.file_image = file_bytes.data();
    request.file_size = file_bytes.size();
    request.check_tensors = false;
    request.validate_architecture = false;
    request.tensor_loader = &tensor_sm;
    request.effect_requests = std::span{effect_requests};
    request.effect_results = std::span{effect_results};
    request.map_layers = {nullptr, map_layers_ok};
    request.on_done = {&state, on_load_done};
    request.on_error = {&state, on_load_error};

    print_step("load", loader_sm.process_event(request));
    std::printf("load done=%s error=%s err=%d\n",
                state.done ? "true" : "false",
                state.error ? "true" : "false", static_cast<int>(state.err));
  }

  {
    std::printf("=== model_load parse failure ===\n");
    emel::model::loader::sm loader_sm;
    emel::model::data model_data = {};
    std::array<uint8_t, 4> file_bytes = {};
    load_probe_state state{};

    emel::model::loader::event::parse_model_fn parse_model{nullptr,
                                                           parse_model_fail};
    emel::model::loader::event::load request{model_data, parse_model};
    request.file_image = file_bytes.data();
    request.file_size = file_bytes.size();
    request.on_done = {&state, on_load_done};
    request.on_error = {&state, on_load_error};

    print_step("load", loader_sm.process_event(request));
    std::printf("load done=%s error=%s err=%d\n",
                state.done ? "true" : "false",
                state.error ? "true" : "false", static_cast<int>(state.err));
  }

  return 0;
}
