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

inline bool store_parsing_done(void * owner_sm,
                               const emel::model::loader::events::parsing_done &) {
  auto * ctx = static_cast<context *>(owner_sm);
  if (ctx == nullptr) {
    return false;
  }
  ctx->phase_error = EMEL_OK;
  ctx->last_error = EMEL_OK;
  return true;
}

inline bool store_parsing_error(void * owner_sm,
                                const emel::model::loader::events::parsing_error & ev) {
  auto * ctx = static_cast<context *>(owner_sm);
  if (ctx == nullptr) {
    return false;
  }
  ctx->phase_error = ev.err;
  ctx->last_error = ev.err;
  return true;
}

inline bool store_loading_done(void * owner_sm,
                               const emel::model::loader::events::loading_done & ev) {
  auto * ctx = static_cast<context *>(owner_sm);
  if (ctx == nullptr) {
    return false;
  }
  ctx->bytes_total = ev.bytes_total;
  ctx->bytes_done = ev.bytes_done;
  ctx->used_mmap = ev.used_mmap;
  ctx->phase_error = EMEL_OK;
  ctx->last_error = EMEL_OK;
  return true;
}

inline bool store_loading_error(void * owner_sm,
                                const emel::model::loader::events::loading_error & ev) {
  auto * ctx = static_cast<context *>(owner_sm);
  if (ctx == nullptr) {
    return false;
  }
  ctx->phase_error = ev.err;
  ctx->last_error = ev.err;
  return true;
}

struct begin_load {
  void operator()(const event::load & ev, context & ctx) const noexcept {
    ctx.request = &ev;
    ctx.bytes_total = 0;
    ctx.bytes_done = 0;
    ctx.used_mmap = false;
    ctx.parser_kind = emel::parser::kind::count;
    ctx.parser_sm = nullptr;
    ctx.parser_dispatch = nullptr;
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
    if (request == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    const emel::parser::selection selection = emel::parser::select(request->parser_map, *request);
    if (selection.entry == nullptr) {
      set_error(ctx, EMEL_ERR_FORMAT_UNSUPPORTED);
      return;
    }
    if (selection.entry->map_parser == nullptr || selection.entry->parser_sm == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    ctx.parser_kind = selection.kind_id;
    ctx.parser_sm = selection.entry->parser_sm;
    ctx.parser_dispatch = selection.entry->dispatch_parse;
    if (ctx.parser_dispatch == nullptr) {
      ctx.parser_dispatch = emel::parser::dispatch_for_kind(ctx.parser_kind);
    }
    if (ctx.parser_dispatch == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    int32_t err = EMEL_OK;
    const bool ok = selection.entry->map_parser(*request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_BACKEND;
      }
      set_error(ctx, err);
      return;
    }
  }
};

struct run_parse {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    const event::load * request = ctx.request;
    if (request == nullptr || ctx.parser_sm == nullptr || ctx.parser_dispatch == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    emel::parser::event::parse_model parse_request{
      .model = &request->model_data,
      .model_path = request->model_path,
      .architectures = request->architectures,
      .n_architectures = request->n_architectures,
      .file_handle = request->file_handle,
      .format_ctx = request->format_ctx,
      .map_tensors = !request->vocab_only,
      .loader_request = request,
      .owner_sm = &ctx,
      .dispatch_done = store_parsing_done,
      .dispatch_error = store_parsing_error
    };
    const bool ok = ctx.parser_dispatch(ctx.parser_sm, parse_request);
    if (!ok && ctx.phase_error == EMEL_OK) {
      set_error(ctx, EMEL_ERR_BACKEND);
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
    if (request == nullptr || request->dispatch_load_weights == nullptr ||
        request->weight_loader_sm == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    emel::model::weight_loader::event::load_weights load_request{
      .request_mmap = request->request_mmap,
      .request_direct_io = request->request_direct_io,
      .check_tensors = request->check_tensors,
      .no_alloc = request->no_alloc,
      .mmap_supported = request->mmap_supported,
      .direct_io_supported = request->direct_io_supported,
      .buffer_allocator_sm = request->buffer_allocator_sm,
      .init_mappings = request->init_mappings,
      .map_mmap = request->map_mmap,
      .load_streamed = request->load_streamed,
      .validate = request->validate_weights,
      .clean_up = request->clean_up_weights,
      .upload_ctx = request->upload_ctx,
      .upload_begin = request->upload_begin,
      .upload_chunk = request->upload_chunk,
      .upload_end = request->upload_end,
      .progress_callback = request->progress_callback,
      .progress_user_data = request->progress_user_data,
      .loader_request = request,
      .owner_sm = &ctx,
      .dispatch_done = store_loading_done,
      .dispatch_error = store_loading_error
    };
    const bool ok = request->dispatch_load_weights(request->weight_loader_sm, load_request);
    if (!ok && ctx.phase_error == EMEL_OK) {
      set_error(ctx, EMEL_ERR_BACKEND);
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
    if (request->error_out != nullptr) {
      *request->error_out = EMEL_OK;
    }
    if (request->dispatch_done != nullptr && request->owner_sm != nullptr) {
      request->dispatch_done(request->owner_sm, events::load_done{
        request,
        ctx.bytes_total,
        ctx.bytes_done,
        ctx.used_mmap
      });
    }
    clear_request(ctx);
  }
};

struct publish_error {
  void operator()(context & ctx) const noexcept {
    const event::load * request = ctx.request;
    if (request == nullptr) {
      return;
    }
    int32_t err = ctx.last_error;
    if (err == EMEL_OK) {
      err = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
    }
    ctx.last_error = err;
    if (request->error_out != nullptr) {
      *request->error_out = err;
    }
    if (request->dispatch_error != nullptr && request->owner_sm != nullptr) {
      request->dispatch_error(request->owner_sm, events::load_error{request, err});
    }
    clear_request(ctx);
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
