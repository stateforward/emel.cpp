#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>

#include "emel/speech/codec/mimi/context.hpp"
#include "emel/speech/codec/mimi/detail.hpp"
#include "emel/speech/codec/mimi/events.hpp"

namespace emel::speech::codec::mimi::guard {

struct bind_model_identity {
  uint64_t low = 14695981039346656037ull;
  uint64_t high = 7809847782465536322ull;
};

inline void guard_mix_model_identity_byte(bind_model_identity &identity,
                                          const uint8_t value) noexcept {
  identity.low = (identity.low ^ value) * 1099511628211ull;
  identity.high = (identity.high ^ value) * 14029467366897019727ull;
}

inline void guard_mix_model_identity_word(bind_model_identity &identity,
                                          const uint64_t value) noexcept {
  for (uint32_t shift = 0u; shift < 64u; shift += 8u) {
    guard_mix_model_identity_byte(
        identity, static_cast<uint8_t>((value >> shift) & 0xffu));
  }
}

inline bind_model_identity
guard_compute_model_identity(const emel::model::data &model) noexcept {
  bind_model_identity identity{};
  const auto mix_signed = [&identity](const int64_t value) noexcept {
    guard_mix_model_identity_word(identity, static_cast<uint64_t>(value));
  };
  const auto mix_unsigned = [&identity](const uint64_t value) noexcept {
    guard_mix_model_identity_word(identity, value);
  };

  mix_signed(model.mimi.sample_rate);
  mix_unsigned(std::bit_cast<uint32_t>(model.mimi.frame_rate));
  mix_signed(model.mimi.n_q);
  mix_signed(model.mimi.card);
  mix_signed(model.mimi.dim);
  mix_signed(model.mimi.semantic_n_q);
  mix_signed(model.mimi.codebook_dim);
  mix_signed(model.mimi.transformer_num_layers);
  mix_signed(model.mimi.transformer_num_heads);
  mix_signed(model.mimi.transformer_context);
  mix_signed(model.mimi.transformer_max_period);

  mix_unsigned(model.n_tensors);
  mix_unsigned(model.name_bytes_used);
  const uint32_t name_bytes = std::min<uint32_t>(
      model.name_bytes_used, static_cast<uint32_t>(model.name_storage.size()));
  for (uint32_t index = 0u; index < name_bytes; ++index) {
    guard_mix_model_identity_byte(
        identity, static_cast<uint8_t>(model.name_storage[index]));
  }

  const uint32_t tensor_count = std::min<uint32_t>(
      model.n_tensors, static_cast<uint32_t>(model.tensors.size()));
  for (uint32_t index = 0u; index < tensor_count; ++index) {
    const auto &tensor = model.tensors[index];
    mix_unsigned(tensor.name_offset);
    mix_unsigned(tensor.name_length);
    mix_signed(tensor.type);
    mix_signed(tensor.n_dims);
    for (const int64_t extent : tensor.dims) {
      mix_signed(extent);
    }
    mix_unsigned(tensor.data_offset);
    mix_unsigned(tensor.file_offset);
    mix_unsigned(tensor.data_size);
    mix_unsigned(reinterpret_cast<uintptr_t>(tensor.data));
    mix_unsigned(tensor.file_index);
  }
  return identity;
}

// Pre-dispatch contract factory. It runs before process_event, captures all
// dry-run validation and sizing facts immutably, and leaves route choice to
// the explicit facade guards below.
inline event::bind_contract
make_bind_contract(const emel::model::data &model) noexcept {
  const bind_model_identity identity = guard_compute_model_identity(model);
  return event::bind_contract{model,
                              identity.low,
                              identity.high,
                              detail::required_prepared_floats(model),
                              detail::required_state_floats(model),
                              detail::required_workspace_floats(model),
                              detail::required_frame_floats(model),
                              detail::validate_codec_contract_f32(model),
                              detail::validate_codec_contract_native(model),
                              detail::validate_codec_contract_q8(model)};
}

inline bool
guard_bind_contract_matches_model(const event::bind_contract &contract,
                                  const emel::model::data &model) noexcept {
  if (contract.model() != &model) {
    return false;
  }
  const bind_model_identity identity = guard_compute_model_identity(model);
  return contract.model_identity_low() == identity.low &&
         contract.model_identity_high() == identity.high;
}

inline constexpr std::string_view k_first_projection_name =
    "mimi.encoder_transformer.transformer.layers.0.self_attn.in_projs.0."
    "weight";

inline bool guard_first_projection_has_dtype(const emel::model::data &model,
                                             const uint8_t dtype) noexcept {
  if (model.n_tensors > model.tensors.size()) {
    return false;
  }
  for (uint32_t index = 0u; index < model.n_tensors; ++index) {
    const auto &tensor = model.tensors[index];
    if (emel::model::tensor_name_view(model, tensor) ==
        k_first_projection_name) {
      return static_cast<uint8_t>(tensor.type) == dtype;
    }
  }
  return false;
}

inline bool
guard_is_transformer_projection_name(const std::string_view name) noexcept {
  const bool transformer =
      name.starts_with("mimi.encoder_transformer.transformer.layers.") ||
      name.starts_with("mimi.decoder_transformer.transformer.layers.");
  const bool projection = name.ends_with(".self_attn.in_projs.0.weight") ||
                          name.ends_with(".self_attn.out_projs.0.weight") ||
                          name.ends_with(".linear1.weight") ||
                          name.ends_with(".linear2.weight");
  return transformer && projection;
}

inline bool
guard_transformer_projections_have_dtype(const emel::model::data &model,
                                         const uint8_t dtype,
                                         const uint64_t alignment) noexcept {
  constexpr uint32_t k_projection_kinds_per_layer = 4u;
  constexpr uint32_t k_transformer_directions = 2u;
  if (model.n_tensors > model.tensors.size() ||
      model.mimi.transformer_num_layers <= 0 ||
      static_cast<uint64_t>(model.mimi.transformer_num_layers) >
          std::numeric_limits<uint32_t>::max() /
              (k_projection_kinds_per_layer * k_transformer_directions)) {
    return false;
  }
  const uint32_t expected_projection_count =
      static_cast<uint32_t>(model.mimi.transformer_num_layers) *
      k_projection_kinds_per_layer * k_transformer_directions;
  uint32_t projection_count = 0u;
  for (uint32_t index = 0u; index < model.n_tensors; ++index) {
    const auto &tensor = model.tensors[index];
    if (!guard_is_transformer_projection_name(
            emel::model::tensor_name_view(model, tensor))) {
      continue;
    }
    if (static_cast<uint8_t>(tensor.type) != dtype || tensor.data == nullptr ||
        reinterpret_cast<uintptr_t>(tensor.data) % alignment != 0u) {
      return false;
    }
    ++projection_count;
  }
  return projection_count == expected_projection_count;
}

inline bool guard_range_valid(const void *data, const uint64_t bytes,
                              uintptr_t &begin, uintptr_t &end) noexcept {
  begin = reinterpret_cast<uintptr_t>(data);
  if (data == nullptr || bytes == 0u ||
      begin > std::numeric_limits<uintptr_t>::max() - bytes) {
    return false;
  }
  end = begin + bytes;
  return true;
}

inline bool guard_ranges_disjoint(const void *first, const uint64_t first_bytes,
                                  const void *second,
                                  const uint64_t second_bytes) noexcept {
  uintptr_t first_begin = 0u;
  uintptr_t first_end = 0u;
  uintptr_t second_begin = 0u;
  uintptr_t second_end = 0u;
  return guard_range_valid(first, first_bytes, first_begin, first_end) &&
         guard_range_valid(second, second_bytes, second_begin, second_end) &&
         (first_end <= second_begin || second_end <= first_begin);
}

inline bool guard_arena_bytes(const std::span<float> arena,
                              uint64_t &bytes) noexcept {
  if (arena.size() > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
    return false;
  }
  bytes = static_cast<uint64_t>(arena.size()) * sizeof(float);
  return bytes != 0u;
}

inline bool
guard_arena_disjoint_from_model(const emel::model::data &model,
                                const std::span<float> arena,
                                const uint64_t arena_bytes) noexcept {
  if (model.n_tensors > model.tensors.size()) {
    return false;
  }
  for (uint32_t index = 0u; index < model.n_tensors; ++index) {
    const auto &tensor = model.tensors[index];
    if (tensor.data == nullptr || tensor.data_size == 0u) {
      continue;
    }
    if (!guard_ranges_disjoint(arena.data(), arena_bytes, tensor.data,
                               tensor.data_size)) {
      return false;
    }
  }
  return true;
}

inline bool guard_arenas_disjoint(const event::initialize &request) noexcept {
  uint64_t prepared_bytes = 0u;
  uint64_t state_bytes = 0u;
  uint64_t workspace_bytes = 0u;
  uint64_t frame_bytes = 0u;
  if (!guard_arena_bytes(request.prepared, prepared_bytes) ||
      !guard_arena_bytes(request.state_arena, state_bytes) ||
      !guard_arena_bytes(request.workspace, workspace_bytes) ||
      !guard_arena_bytes(request.frame, frame_bytes)) {
    return false;
  }
  return guard_ranges_disjoint(request.prepared.data(), prepared_bytes,
                               request.state_arena.data(), state_bytes) &&
         guard_ranges_disjoint(request.prepared.data(), prepared_bytes,
                               request.workspace.data(), workspace_bytes) &&
         guard_ranges_disjoint(request.prepared.data(), prepared_bytes,
                               request.frame.data(), frame_bytes) &&
         guard_ranges_disjoint(request.state_arena.data(), state_bytes,
                               request.workspace.data(), workspace_bytes) &&
         guard_ranges_disjoint(request.state_arena.data(), state_bytes,
                               request.frame.data(), frame_bytes) &&
         guard_ranges_disjoint(request.workspace.data(), workspace_bytes,
                               request.frame.data(), frame_bytes) &&
         guard_arena_disjoint_from_model(request.model, request.prepared,
                                         prepared_bytes) &&
         guard_arena_disjoint_from_model(request.model, request.state_arena,
                                         state_bytes) &&
         guard_arena_disjoint_from_model(request.model, request.workspace,
                                         workspace_bytes) &&
         guard_arena_disjoint_from_model(request.model, request.frame,
                                         frame_bytes);
}

// Bind validation facts are computed before process_event and carried as an
// immutable event value. Guards own route selection and capacity acceptance;
// no detail helper result decides an RTC transition.
struct guard_bind_f32_contract_valid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.contract.model() == &runtime_ev.request.model &&
           runtime_ev.request.contract.f32_valid() &&
           guard_transformer_projections_have_dtype(
               runtime_ev.request.model, emel::kernel::detail::dtype_f32,
               alignof(float)) &&
           guard_bind_contract_matches_model(runtime_ev.request.contract,
                                             runtime_ev.request.model);
  }
};

struct guard_bind_native_contract_valid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.contract.model() == &runtime_ev.request.model &&
           runtime_ev.request.contract.native_valid() &&
           guard_transformer_projections_have_dtype(
               runtime_ev.request.model, emel::kernel::detail::dtype_f16,
               alignof(uint16_t)) &&
           guard_bind_contract_matches_model(runtime_ev.request.contract,
                                             runtime_ev.request.model);
  }
};

struct guard_bind_q8_contract_valid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.contract.model() == &runtime_ev.request.model &&
           runtime_ev.request.contract.q8_valid() &&
           guard_first_projection_has_dtype(runtime_ev.request.model, 8u) &&
           guard_bind_contract_matches_model(runtime_ev.request.contract,
                                             runtime_ev.request.model);
  }
};

struct guard_bind_contract_invalid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_bind_f32_contract_valid{}(runtime_ev, ctx) &&
           !guard_bind_native_contract_valid{}(runtime_ev, ctx) &&
           !guard_bind_q8_contract_valid{}(runtime_ev, ctx);
  }
};

struct guard_arena_capacity_valid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    const auto &request = runtime_ev.request;
    const auto &model = request.model;
    // The bind walk and reset_streaming_state write through every arena, so
    // a sized span without storage must reject here, not crash in the bind.
    return model.mimi.dim <= action::k_max_latent_floats &&
           request.prepared.data() != nullptr &&
           request.state_arena.data() != nullptr &&
           request.workspace.data() != nullptr &&
           request.frame.data() != nullptr &&
           reinterpret_cast<uintptr_t>(request.prepared.data()) %
                   alignof(float) ==
               0u &&
           reinterpret_cast<uintptr_t>(request.state_arena.data()) %
                   alignof(float) ==
               0u &&
           reinterpret_cast<uintptr_t>(request.workspace.data()) %
                   alignof(float) ==
               0u &&
           reinterpret_cast<uintptr_t>(request.frame.data()) % alignof(float) ==
               0u &&
           request.prepared.size() >= request.contract.prepared_floats() &&
           request.state_arena.size() >= request.contract.state_floats() &&
           request.workspace.size() >= request.contract.workspace_floats() &&
           request.frame.size() >= request.contract.frame_floats() &&
           guard_arenas_disjoint(request);
  }
};

struct guard_arena_capacity_invalid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_arena_capacity_valid{}(runtime_ev, ctx);
  }
};

// Per-frame request validation against the bound runtime (O(1) reads plus a
// bounded n_q code scan); the compute actions selected behind these guards
// are non-failing by contract.
struct guard_encode_request_valid {
  bool operator()(const event::encode_frame_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &request = runtime_ev.request;
    // The compute actions read pcm and write codes_out; sized spans without
    // storage must reject here.
    return request.pcm.data() != nullptr &&
           request.codes_out.data() != nullptr &&
           request.pcm.size() ==
               static_cast<size_t>(ctx.runtime.frame_samples) &&
           request.codes_out.size() >= static_cast<size_t>(ctx.runtime.n_q);
  }
};

struct guard_encode_request_invalid {
  bool operator()(const event::encode_frame_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_encode_request_valid{}(runtime_ev, ctx);
  }
};

struct guard_decode_request_valid {
  bool operator()(const event::decode_frame_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &request = runtime_ev.request;
    // Mirror of the encode guard: codes are read and pcm_out is written.
    const size_t code_count = request.codes.size();
    return request.codes.data() != nullptr &&
           request.pcm_out.data() != nullptr &&
           code_count >= static_cast<size_t>(
                             ctx.runtime.quantizer.semantic.level_count) &&
           code_count <= static_cast<size_t>(ctx.runtime.n_q) &&
           request.pcm_out.size() >=
               static_cast<size_t>(ctx.runtime.frame_samples);
  }
};

struct guard_decode_request_invalid {
  bool operator()(const event::decode_frame_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_decode_request_valid{}(runtime_ev, ctx);
  }
};

struct guard_decode_codes_valid {
  bool operator()(const event::decode_frame_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return detail::validate_codes_in_range(ctx.runtime,
                                           runtime_ev.request.codes);
  }
};

struct guard_decode_codes_invalid {
  bool operator()(const event::decode_frame_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_decode_codes_valid{}(runtime_ev, ctx);
  }
};

template <class runtime_event_type> struct guard_has_error_out {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.error_out != nullptr;
  }
};

template <class runtime_event_type> struct guard_no_error_out {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.error_out == nullptr;
  }
};

// Unexpected external events: the error-out channel choice is a guarded
// transition, mirroring guard_has_error_out on the modeled routes. Events
// without a runtime ctx or without an error_out channel take the absent
// route.
struct guard_unexpected_error_out_present {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    if constexpr (requires {
                    ev.ctx.err;
                    ev.request.error_out;
                  }) {
      return ev.request.error_out != nullptr;
    } else {
      return false;
    }
  }
};

struct guard_unexpected_error_out_absent {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !guard_unexpected_error_out_present{}(ev, ctx);
  }
};

template <class runtime_event_type> struct guard_has_done_callback {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_done);
  }
};

template <class runtime_event_type> struct guard_no_done_callback {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_done_callback<runtime_event_type>{}(runtime_ev, ctx);
  }
};

template <class runtime_event_type> struct guard_has_error_callback {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_error);
  }
};

template <class runtime_event_type> struct guard_no_error_callback {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_error_callback<runtime_event_type>{}(runtime_ev, ctx);
  }
};

} // namespace emel::speech::codec::mimi::guard
