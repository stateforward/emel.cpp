#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/model/loader/events.hpp"
#include "emel/model/weight_loader/events.hpp"

namespace emel::model::weight_loader::action {

struct context {
  const event::load_weights * request = nullptr;
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
  bool use_mmap = false;
  bool use_direct_io = false;
  int32_t phase_error = EMEL_OK;
  int32_t last_error = EMEL_OK;
};

inline void clear_request(context & ctx) noexcept {
  ctx.request = nullptr;
}

inline void set_error(context & ctx, const int32_t err) noexcept {
  ctx.phase_error = err;
  ctx.last_error = err;
}

struct begin_load {
  void operator()(const event::load_weights & ev, context & ctx) const noexcept {
    ctx.request = &ev;
    ctx.bytes_total = 0;
    ctx.bytes_done = 0;
    ctx.used_mmap = false;
    ctx.use_mmap = false;
    ctx.use_direct_io = false;
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct set_invalid_argument {
  void operator()(context & ctx) const noexcept { set_error(ctx, EMEL_ERR_INVALID_ARGUMENT); }
};

struct set_backend_error {
  void operator()(context & ctx) const noexcept { set_error(ctx, EMEL_ERR_BACKEND); }
};

struct select_strategy {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    ctx.use_mmap = false;
    ctx.use_direct_io = false;
    const event::load_weights * request = ctx.request;
    if (request == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    const bool use_direct_io = request->request_direct_io && request->direct_io_supported;
    bool use_mmap = false;
    if (!request->no_alloc && request->request_mmap && request->mmap_supported && !use_direct_io) {
      use_mmap = true;
    }
    ctx.use_direct_io = use_direct_io;
    ctx.use_mmap = use_mmap;
  }
};

struct run_init_mappings {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    const event::load_weights * request = ctx.request;
    if (request == nullptr || request->init_mappings == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    int32_t err = EMEL_OK;
    const bool ok = request->init_mappings(*request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_BACKEND;
      }
      set_error(ctx, err);
    }
  }
};

struct skip_init_mappings {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct run_load_mmap {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    ctx.bytes_total = 0;
    ctx.bytes_done = 0;
    ctx.used_mmap = true;
    const event::load_weights * request = ctx.request;
    if (request == nullptr || request->map_mmap == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    uint64_t bytes_done = 0;
    uint64_t bytes_total = 0;
    int32_t err = EMEL_OK;
    const bool ok = request->map_mmap(*request, &bytes_done, &bytes_total, &err);
    ctx.bytes_done = bytes_done;
    ctx.bytes_total = bytes_total;
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_BACKEND;
      }
      set_error(ctx, err);
    }
  }
};

struct run_load_streamed {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    ctx.bytes_total = 0;
    ctx.bytes_done = 0;
    ctx.used_mmap = false;
    const event::load_weights * request = ctx.request;
    if (request == nullptr || request->load_streamed == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    uint64_t bytes_done = 0;
    uint64_t bytes_total = 0;
    int32_t err = EMEL_OK;
    const bool ok = request->load_streamed(*request, &bytes_done, &bytes_total, &err);
    ctx.bytes_done = bytes_done;
    ctx.bytes_total = bytes_total;
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_BACKEND;
      }
      set_error(ctx, err);
    }
  }
};

struct run_validate {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    const event::load_weights * request = ctx.request;
    if (request == nullptr || request->validate == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    int32_t err = EMEL_OK;
    const bool ok = request->validate(*request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_MODEL_INVALID;
      }
      set_error(ctx, err);
    }
  }
};

struct skip_validate {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct run_clean_up {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
    const event::load_weights * request = ctx.request;
    if (request == nullptr || request->clean_up == nullptr) {
      set_error(ctx, EMEL_ERR_INVALID_ARGUMENT);
      return;
    }
    int32_t err = EMEL_OK;
    const bool ok = request->clean_up(*request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_BACKEND;
      }
      set_error(ctx, err);
    }
  }
};

struct skip_clean_up {
  void operator()(context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    ctx.last_error = EMEL_OK;
  }
};

struct publish_done {
  void operator()(context & ctx) const noexcept {
    const event::load_weights * request = ctx.request;
    if (request == nullptr) {
      return;
    }
    if (request->dispatch_done != nullptr && request->owner_sm != nullptr) {
      request->dispatch_done(request->owner_sm, emel::model::loader::events::loading_done{
        request->loader_request,
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
    const event::load_weights * request = ctx.request;
    if (request == nullptr) {
      return;
    }
    int32_t err = ctx.last_error;
    if (err == EMEL_OK) {
      err = ctx.phase_error == EMEL_OK ? EMEL_ERR_BACKEND : ctx.phase_error;
    }
    ctx.last_error = err;
    if (request->dispatch_error != nullptr && request->owner_sm != nullptr) {
      request->dispatch_error(request->owner_sm, emel::model::loader::events::loading_error{
        request->loader_request,
        err
      });
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
inline constexpr select_strategy select_strategy{};
inline constexpr run_init_mappings run_init_mappings{};
inline constexpr skip_init_mappings skip_init_mappings{};
inline constexpr run_load_mmap run_load_mmap{};
inline constexpr run_load_streamed run_load_streamed{};
inline constexpr run_validate run_validate{};
inline constexpr skip_validate skip_validate{};
inline constexpr run_clean_up run_clean_up{};
inline constexpr skip_clean_up skip_clean_up{};
inline constexpr publish_done publish_done{};
inline constexpr publish_error publish_error{};
inline constexpr on_unexpected on_unexpected{};

}  // namespace emel::model::weight_loader::action
