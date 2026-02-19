#pragma once

#include "emel/emel.h"
#include "emel/model/weight_loader/actions.hpp"

namespace emel::model::weight_loader::guard {

struct has_request {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr;
  }
};

struct phase_ok {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error == EMEL_OK;
  }
};

struct phase_failed {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.phase_error != EMEL_OK;
  }
};

struct use_mmap_selected {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.use_mmap;
  }
};

struct use_stream_selected {
  bool operator()(const action::context & ctx) const noexcept {
    return !ctx.use_mmap;
  }
};

struct can_init_mappings {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr && ctx.request->init_mappings != nullptr;
  }
};

struct skip_init_mappings {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr && ctx.request->init_mappings == nullptr;
  }
};

struct cannot_init_mappings {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request == nullptr;
  }
};

struct can_load_streamed {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr && ctx.request->load_streamed != nullptr;
  }
};

struct cannot_load_streamed {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request == nullptr || ctx.request->load_streamed == nullptr;
  }
};

struct can_load_mmap {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr && ctx.request->map_mmap != nullptr;
  }
};

struct cannot_load_mmap {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request == nullptr || ctx.request->map_mmap == nullptr;
  }
};

struct can_validate {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr && ctx.request->check_tensors && ctx.request->validate != nullptr;
  }
};

struct skip_validate {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr && (!ctx.request->check_tensors || ctx.request->validate == nullptr);
  }
};

struct cannot_validate {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request == nullptr;
  }
};

struct can_clean_up {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr && ctx.used_mmap && ctx.request->clean_up != nullptr;
  }
};

struct skip_clean_up {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request != nullptr && (!ctx.used_mmap || ctx.request->clean_up == nullptr);
  }
};

struct cannot_clean_up {
  bool operator()(const action::context & ctx) const noexcept {
    return ctx.request == nullptr;
  }
};

struct phase_ok_and_use_mmap_and_can_init_mappings {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && use_mmap_selected{}(ctx) && can_init_mappings{}(ctx);
  }
};

struct phase_ok_and_use_mmap_and_skip_init_mappings {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && use_mmap_selected{}(ctx) && skip_init_mappings{}(ctx);
  }
};

struct phase_ok_and_use_mmap_and_cannot_init_mappings {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && use_mmap_selected{}(ctx) && cannot_init_mappings{}(ctx);
  }
};

struct phase_ok_and_use_stream_and_can_load_streamed {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && use_stream_selected{}(ctx) && can_load_streamed{}(ctx);
  }
};

struct phase_ok_and_use_stream_and_cannot_load_streamed {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && use_stream_selected{}(ctx) && cannot_load_streamed{}(ctx);
  }
};

struct phase_ok_and_can_load_mmap {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && can_load_mmap{}(ctx);
  }
};

struct phase_ok_and_cannot_load_mmap {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && cannot_load_mmap{}(ctx);
  }
};

struct phase_ok_and_can_validate {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && can_validate{}(ctx);
  }
};

struct phase_ok_and_skip_validate {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && skip_validate{}(ctx);
  }
};

struct phase_ok_and_cannot_validate {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && cannot_validate{}(ctx);
  }
};

struct phase_ok_and_can_clean_up {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && can_clean_up{}(ctx);
  }
};

struct phase_ok_and_skip_clean_up {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && skip_clean_up{}(ctx);
  }
};

struct phase_ok_and_cannot_clean_up {
  bool operator()(const action::context & ctx) const noexcept {
    return phase_ok{}(ctx) && cannot_clean_up{}(ctx);
  }
};

}  // namespace emel::model::weight_loader::guard
