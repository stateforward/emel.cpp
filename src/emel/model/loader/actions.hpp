#pragma once

#include "emel/model/loader/context.hpp"

namespace emel::model::loader::action {

inline void clear_request(context & ctx) noexcept {
  ctx.request = nullptr;
}

inline void set_error(context & ctx, const int32_t err) noexcept {
  ctx.phase_error = err;
  ctx.last_error = err;
}

struct begin_load {
  void operator()(const event::load & ev, context & ctx) const noexcept {
    ctx.request = &ev;
    ctx.bytes_total = 0;
    ctx.bytes_done = 0;
    ctx.used_mmap = false;
    ctx.parser_requirements = {};
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    if (ev.error_out != nullptr) {
      *ev.error_out = EMEL_OK;
    }
  }
};

struct set_invalid_argument {
  void operator()(context & ctx) const noexcept { set_error(ctx, EMEL_ERR_INVALID_ARGUMENT); }
};

struct set_backend_error {
  void operator()(context & ctx) const noexcept { set_error(ctx, EMEL_ERR_BACKEND); }
};

struct run_map_parser {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    const event::load * request = ctx.request;
    if (request == nullptr || request->parser_sm == nullptr ||
        request->dispatch_probe == nullptr ||
        request->dispatch_bind_storage == nullptr ||
        request->dispatch_parse == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
  }
};

struct run_parse {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    const event::load * request = ctx.request;
    if (request == nullptr || request->parser_sm == nullptr ||
        request->dispatch_probe == nullptr ||
        request->dispatch_bind_storage == nullptr ||
        request->dispatch_parse == nullptr ||
        request->file_image == nullptr || request->file_size == 0) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    emel::parser::gguf::event::probe probe_request{
      .file_image = request->file_image,
      .size = request->file_size,
      .requirements_out = &ctx.parser_requirements,
    };
    if (!request->dispatch_probe(request->parser_sm, probe_request)) {
      set_error(ctx, EMEL_ERR_BACKEND);
      return;
    }

    emel::parser::gguf::event::bind_storage bind_request{
      .kv_arena = request->parser_kv_arena,
      .kv_arena_size = request->parser_kv_arena_size,
      .kv_entries = request->parser_kv_entries,
      .kv_entry_capacity = request->parser_kv_entry_capacity,
      .tensors = request->parser_tensors,
      .tensor_capacity = request->parser_tensor_capacity,
    };
    if (!request->dispatch_bind_storage(request->parser_sm, bind_request)) {
      set_error(ctx, EMEL_ERR_BACKEND);
      return;
    }

    emel::parser::gguf::event::parse parse_request{
      .file_image = request->file_image,
      .size = request->file_size,
    };
    if (!request->dispatch_parse(request->parser_sm, parse_request)) {
      set_error(ctx, EMEL_ERR_BACKEND);
      return;
    }
  }
};

struct run_load_weights {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    ctx.bytes_total = 0;
    ctx.bytes_done = 0;
    ctx.used_mmap = false;
    const event::load * request = ctx.request;
    if (request == nullptr || request->weight_loader_sm == nullptr ||
        request->dispatch_bind_weights == nullptr ||
        request->dispatch_plan_load == nullptr ||
        request->dispatch_apply_results == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }

    emel::model::weight_loader::event::bind_storage bind_request{
      .tensors = request->parser_tensors,
      .tensor_count = request->parser_tensor_capacity,
    };
    if (!request->dispatch_bind_weights(request->weight_loader_sm, bind_request)) {
      set_error(ctx, EMEL_ERR_BACKEND);
      return;
    }

    uint32_t planned_count = 0;
    emel::model::weight_loader::event::plan_load plan_request{
      .effects_out = request->effect_requests,
      .effect_capacity = request->effect_capacity,
      .effect_count_out = &planned_count,
    };
    if (!request->dispatch_plan_load(request->weight_loader_sm, plan_request)) {
      set_error(ctx, EMEL_ERR_BACKEND);
      return;
    }

    emel::model::weight_loader::event::apply_effect_results apply_request{
      .results = request->effect_results,
      .result_count = planned_count,
    };
    if (!request->dispatch_apply_results(request->weight_loader_sm, apply_request)) {
      set_error(ctx, EMEL_ERR_BACKEND);
      return;
    }
  }
};

struct run_map_layers {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    const event::load * request = ctx.request;
    if (request == nullptr || request->map_layers == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    int32_t err = EMEL_OK;
    const bool ok = request->map_layers(*request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_BACKEND;
      }
      set_error(ctx, err);
    }
  }
};

struct run_validate_structure {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    const event::load * request = ctx.request;
    if (request == nullptr || request->validate_structure == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    int32_t err = EMEL_OK;
    const bool ok = request->validate_structure(*request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_MODEL_INVALID;
      }
      set_error(ctx, err);
    }
  }
};

struct skip_validate_structure {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct run_validate_architecture {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    const event::load * request = ctx.request;
    if (request == nullptr || request->validate_architecture_impl == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    int32_t err = EMEL_OK;
    const bool ok = request->validate_architecture_impl(*request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_MODEL_INVALID;
      }
      set_error(ctx, err);
    }
  }
};

struct publish_done {
  void operator()(context & ctx) const noexcept {
    const event::load * request = ctx.request;
    if (request == nullptr) {
      return;
    }
    if (request->dispatch_done != nullptr && request->owner_sm != nullptr) {
      request->dispatch_done(request->owner_sm, emel::model::loader::events::load_done{
        request,
        ctx.bytes_total,
        ctx.bytes_done,
        ctx.used_mmap,
      });
    }
  }
};

struct publish_error {
  void operator()(context & ctx) const noexcept {
    const event::load * request = ctx.request;
    if (request == nullptr) {
      return;
    }
    if (request->dispatch_error != nullptr && request->owner_sm != nullptr) {
      request->dispatch_error(request->owner_sm, emel::model::loader::events::load_error{
        request,
        ctx.last_error,
      });
    }
  }
};

struct on_unexpected {
  void operator()(context & ctx) const noexcept { set_error(ctx, EMEL_ERR_BACKEND); }
};

inline constexpr begin_load begin_load{};
inline constexpr set_invalid_argument set_invalid_argument{};
inline constexpr set_backend_error set_backend_error{};
inline constexpr run_map_parser run_map_parser{};
inline constexpr run_parse run_parse{};
inline constexpr run_load_weights run_load_weights{};
inline constexpr run_map_layers run_map_layers{};
inline constexpr run_validate_structure run_validate_structure{};
inline constexpr skip_validate_structure skip_validate_structure{};
inline constexpr run_validate_architecture run_validate_architecture{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::model::loader::action
